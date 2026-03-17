"""Tests for tuned_lens.py - tuned lens visualizer and training."""

import math
import warnings

warnings.filterwarnings("ignore")

import logging

logging.disable(logging.WARNING)

import contextlib
import io
import pathlib
import re
import tempfile

import torch
import torch.nn as nn
import pytest

from tuned_lens import (
    TunedLens,
    load_hf_model_and_tuned_lens,
    run_tuned_lens,
    train_tuned_lens,
    plot_loss_curves,
    prob_to_bg_fg,
    sanitize_token,
    display_lens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_rgb(bg: str) -> tuple[int, int, int]:
    """Parse 'rgb(r,g,b)' -> (r, g, b)."""
    parts = bg[4:-1].split(",")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _make_console_and_buf():
    """Create a Rich Console that writes to a StringIO buffer."""
    buf = io.StringIO()
    from rich.console import Console
    return Console(file=buf, force_terminal=True, width=200), buf


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from a string."""
    return _ANSI_RE.sub("", text)


def _with_test_console(fn, *args, **kwargs):
    """Run fn while the module-level console is swapped out."""
    import tuned_lens
    test_console, buf = _make_console_and_buf()
    orig = tuned_lens.console
    tuned_lens.console = test_console
    try:
        fn(*args, **kwargs)
    finally:
        tuned_lens.console = orig
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hf_tuned():
    """Load HuggingFace GPT-2 + TunedLens (identity-initialized) once for all tests."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(
        io.StringIO()
    ):
        hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hf_model.eval()
    for param in hf_model.parameters():
        param.requires_grad = False
    tuned = TunedLens.from_model(hf_model)
    tuned.eval()
    return hf_model, tokenizer, tuned


# ===================================================================
# prob_to_bg_fg -- unit tests
# ===================================================================

class TestProbToBgFg:
    def test_returns_tuple_of_two_strings(self):
        bg, fg = prob_to_bg_fg(0.5)
        assert isinstance(bg, str)
        assert isinstance(fg, str)

    def test_zero_prob_is_red(self):
        bg, _ = prob_to_bg_fg(0.0)
        r, g, b = _parse_rgb(bg)
        assert r > 100
        assert g == 0

    def test_full_prob_is_green(self):
        bg, _ = prob_to_bg_fg(1.0)
        r, g, b = _parse_rgb(bg)
        assert r == 0
        assert g >= 150

    def test_mid_prob_is_yellowish(self):
        bg, _ = prob_to_bg_fg(0.5)
        r, g, b = _parse_rgb(bg)
        assert r > 50
        assert g > 50

    def test_gradient_green_monotonic_increasing(self):
        prev_g = -1
        for pct in range(0, 101):
            bg, _ = prob_to_bg_fg(pct / 100)
            g = _parse_rgb(bg)[1]
            assert g >= prev_g, (
                f"Green not monotonic at {pct}%: got {g}, prev {prev_g}"
            )
            prev_g = g

    def test_gradient_red_monotonic_decreasing(self):
        prev_r = 999
        for pct in range(0, 101):
            bg, _ = prob_to_bg_fg(pct / 100)
            r = _parse_rgb(bg)[0]
            assert r <= prev_r, (
                f"Red not monotonic at {pct}%: got {r}, prev {prev_r}"
            )
            prev_r = r

    def test_fg_is_black_or_white(self):
        for pct in range(0, 101):
            _, fg = prob_to_bg_fg(pct / 100)
            assert fg in ("black", "white")

    def test_clamps_negative(self):
        assert prob_to_bg_fg(-0.5) == prob_to_bg_fg(0.0)

    def test_clamps_above_one(self):
        assert prob_to_bg_fg(1.5) == prob_to_bg_fg(1.0)

    def test_dark_bg_gets_white_fg(self):
        _, fg = prob_to_bg_fg(0.0)
        assert fg == "white"

    def test_bright_bg_gets_black_fg(self):
        _, fg = prob_to_bg_fg(0.5)
        assert fg == "black"

    # --- Edge cases ---

    def test_nan_does_not_crash(self):
        bg, fg = prob_to_bg_fg(float("nan"))
        assert bg.startswith("rgb(")
        assert fg in ("black", "white")

    def test_nan_treated_as_zero(self):
        assert prob_to_bg_fg(float("nan")) == prob_to_bg_fg(0.0)

    def test_positive_inf_treated_as_one(self):
        assert prob_to_bg_fg(float("inf")) == prob_to_bg_fg(1.0)

    def test_negative_inf_treated_as_zero(self):
        assert prob_to_bg_fg(float("-inf")) == prob_to_bg_fg(0.0)

    def test_boundary_continuity_at_half(self):
        """No large discontinuity at the p=0.5 boundary between the two branches."""
        bg_below, _ = prob_to_bg_fg(0.499)
        bg_at, _ = prob_to_bg_fg(0.5)
        bg_above, _ = prob_to_bg_fg(0.501)

        r1, g1, _ = _parse_rgb(bg_below)
        r2, g2, _ = _parse_rgb(bg_at)
        r3, g3, _ = _parse_rgb(bg_above)

        # Red channel: should be ~200 on both sides of boundary
        assert abs(r1 - r2) <= 5
        # Green: should be ~160 on both sides
        assert abs(g1 - g2) <= 2
        assert abs(g2 - g3) <= 2

    def test_blue_is_always_zero(self):
        for pct in range(0, 101, 5):
            bg, _ = prob_to_bg_fg(pct / 100)
            _, _, b = _parse_rgb(bg)
            assert b == 0

    def test_rgb_values_in_valid_range(self):
        for pct in range(0, 101):
            bg, _ = prob_to_bg_fg(pct / 100)
            r, g, b = _parse_rgb(bg)
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255


# ===================================================================
# sanitize_token -- unit tests
# ===================================================================

class TestSanitizeToken:
    def test_normal_word(self):
        assert sanitize_token("hello") == "hello"

    def test_leading_space(self):
        result = sanitize_token(" Tower")
        assert "Tower" in result

    def test_newline_escaped(self):
        assert "\\n" in sanitize_token("\n")

    def test_tab_escaped(self):
        assert "\\t" in sanitize_token("\t")

    def test_carriage_return_escaped(self):
        assert "\\r" in sanitize_token("\r")

    def test_null_byte_escaped(self):
        result = sanitize_token("\x00")
        assert "\x00" not in result  # should be escaped

    def test_truncates_long_token(self):
        result = sanitize_token("a" * 50)
        assert len(result) == 12
        assert result.endswith("\u2026")

    def test_short_token_not_truncated(self):
        assert sanitize_token("hi") == "hi"

    def test_empty_string(self):
        result = sanitize_token("")
        assert result == ""

    def test_exactly_12_chars_not_truncated(self):
        tok = "a" * 12
        result = sanitize_token(tok)
        assert result == tok
        assert "\u2026" not in result

    def test_exactly_13_chars_truncated(self):
        tok = "a" * 13
        result = sanitize_token(tok)
        assert len(result) == 12
        assert result.endswith("\u2026")

    def test_unicode_chars(self):
        result = sanitize_token("caf\u00e9")
        assert "caf" in result

    def test_backslash(self):
        result = sanitize_token("a\\b")
        # repr will escape the backslash
        assert "\\\\" in result


# ===================================================================
# run_tuned_lens -- integration tests
# ===================================================================

class TestRunTunedLens:
    def test_basic_output_shape(self, hf_tuned):
        hf_model, tokenizer, tuned = hf_tuned
        input_tokens, layer_preds = run_tuned_lens(
            hf_model, tokenizer, tuned, "Hello", top_k=1
        )
        assert len(input_tokens) >= 1
        n_layers = hf_model.config.n_layer
        assert len(layer_preds) == n_layers + 1
        for row in layer_preds:
            assert len(row) == len(input_tokens)

    def test_top_k_predictions(self, hf_tuned):
        hf_model, tokenizer, tuned = hf_tuned
        _, layer_preds = run_tuned_lens(
            hf_model, tokenizer, tuned, "Hello", top_k=3
        )
        for row in layer_preds:
            for cell in row:
                assert len(cell) == 3
                for tok_str, prob in cell:
                    assert isinstance(tok_str, str)
                    assert 0.0 <= prob <= 1.0

    def test_output_row_matches_model(self, hf_tuned):
        hf_model, tokenizer, tuned = hf_tuned
        prompt = "The cat"
        _, layer_preds = run_tuned_lens(
            hf_model, tokenizer, tuned, prompt, top_k=1
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_model(**inputs)
        probs = torch.softmax(outputs.logits[0, -1], dim=-1)
        _, top_idx = probs.max(dim=-1)
        model_pred = tokenizer.decode(top_idx.item())
        lens_pred = layer_preds[-1][-1][0][0]
        assert lens_pred == model_pred

    def test_single_token_prompt(self, hf_tuned):
        hf_model, tokenizer, tuned = hf_tuned
        input_tokens, layer_preds = run_tuned_lens(
            hf_model, tokenizer, tuned, "A", top_k=1
        )
        assert len(input_tokens) >= 1
        for row in layer_preds:
            assert len(row) == len(input_tokens)

    def test_probs_are_valid(self, hf_tuned):
        hf_model, tokenizer, tuned = hf_tuned
        _, layer_preds = run_tuned_lens(
            hf_model, tokenizer, tuned, "Test", top_k=1
        )
        for row in layer_preds:
            for cell in row:
                _, prob = cell[0]
                assert 0.0 <= prob <= 1.0
                assert not math.isnan(prob)

    def test_different_prompts_differ(self, hf_tuned):
        hf_model, tokenizer, tuned = hf_tuned
        _, preds_a = run_tuned_lens(
            hf_model, tokenizer, tuned, "The dog", top_k=1
        )
        _, preds_b = run_tuned_lens(
            hf_model, tokenizer, tuned, "import os", top_k=1
        )
        assert preds_a[-1][-1][0][0] != preds_b[-1][-1][0][0]


# ===================================================================
# TunedLens class -- unit tests
# ===================================================================

class TestTunedLensModule:
    def test_identity_initialization(self, hf_tuned):
        """Probes should start as identity transforms."""
        _, _, tuned = hf_tuned
        for probe in tuned.probes:
            weight_diff = (probe.weight - torch.eye(tuned.d_model)).abs().max()
            assert weight_diff < 1e-6, "Probe weight should be identity"
            bias_max = probe.bias.abs().max()
            assert bias_max < 1e-6, "Probe bias should be zeros"

    def test_identity_probe_produces_valid_logits(self, hf_tuned):
        """Identity probe on intermediate hidden state should produce valid logits."""
        hf_model, tokenizer, tuned = hf_tuned
        prompt = "The cat"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_model(**inputs, output_hidden_states=True)
            # Use an intermediate layer (pre-LN) -- hidden_states[-1] is post-LN
            h = outputs.hidden_states[6]  # After block 5
            logits = tuned.forward(h, 5)
        probs = torch.softmax(logits[0], dim=-1)
        # Should produce valid probability distributions
        assert torch.all(probs >= 0)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]), atol=1e-4)

    def test_forward_output_shape(self, hf_tuned):
        hf_model, tokenizer, tuned = hf_tuned
        inputs = tokenizer("Hello", return_tensors="pt")
        with torch.no_grad():
            outputs = hf_model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[1]
            logits = tuned.forward(h, 0)
        assert logits.shape == (1, h.shape[1], hf_model.config.vocab_size)

    def test_save_and_load_roundtrip(self, hf_tuned):
        """Save probes, load them, verify weights match."""
        hf_model, _, tuned = hf_tuned
        # Perturb one probe so it's not identity
        with torch.no_grad():
            tuned.probes[0].weight.add_(0.01 * torch.randn_like(tuned.probes[0].weight))

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tuned.save(pathlib.Path(tmpdir))
                loaded = TunedLens.from_pretrained(hf_model, tmpdir)

            for orig_probe, loaded_probe in zip(tuned.probes, loaded.probes):
                assert torch.allclose(orig_probe.weight, loaded_probe.weight)
                assert torch.allclose(orig_probe.bias, loaded_probe.bias)
        finally:
            # Reset the probe back to identity for other tests
            with torch.no_grad():
                nn.init.eye_(tuned.probes[0].weight)
                nn.init.zeros_(tuned.probes[0].bias)

    def test_probe_parameters_are_trainable(self, hf_tuned):
        _, _, tuned = hf_tuned
        trainable = list(tuned.probe_parameters())
        assert len(trainable) == 2 * tuned.n_layers  # weight + bias per layer
        for p in trainable:
            assert p.requires_grad

    def test_ln_and_unembed_frozen(self, hf_tuned):
        _, _, tuned = hf_tuned
        for p in tuned.layer_norm.parameters():
            assert not p.requires_grad
        for p in tuned.unembed.parameters():
            assert not p.requires_grad


# ===================================================================
# display_lens -- rendering tests
# ===================================================================

class TestDisplayLens:
    def test_display_tuned_lens_runs(self, hf_tuned):
        hf_model, tokenizer, tuned = hf_tuned
        input_tokens, layer_preds = run_tuned_lens(
            hf_model, tokenizer, tuned, "Hi", top_k=1
        )
        output = _with_test_console(
            display_lens, input_tokens, layer_preds, "Hi",
            lens_type="Tuned", has_embed_layer=False,
        )
        assert "Tuned Lens" in output
        assert "Output" in output
        assert "Embed" not in output

    def test_no_percentages_in_table_cells(self, hf_tuned):
        hf_model, tokenizer, tuned = hf_tuned
        input_tokens, layer_preds = run_tuned_lens(
            hf_model, tokenizer, tuned, "Hi", top_k=1
        )
        output = _with_test_console(
            display_lens, input_tokens, layer_preds, "Hi",
            lens_type="Tuned", has_embed_layer=False,
        )
        table_part = output.split("Confidence")[0]
        for line in table_part.strip().split("\n"):
            if "Layer" in line or "\u2501" in line or "\u2500" in line:
                continue
            matches = re.findall(r"\d{1,3}%", line)
            assert len(matches) == 0, (
                f"Found percentage in table data: {matches} in: {line}"
            )

    def test_top_k_data_structure(self, hf_tuned):
        hf_model, tokenizer, tuned = hf_tuned
        _, layer_preds = run_tuned_lens(
            hf_model, tokenizer, tuned, "Hi", top_k=2
        )
        for row in layer_preds:
            for cell in row:
                assert len(cell) == 2

    def test_label_embed_when_single_layer_embed(self):
        """Edge case: 1 row with has_embed_layer -> label should be 'Embed'."""
        fake_preds = [[[("tok", 0.5)]]]
        output = _with_test_console(
            display_lens, ["A"], fake_preds, "A",
            lens_type="Logit", has_embed_layer=True,
        )
        assert "Embed" in output

    def test_label_output_when_single_layer_tuned(self):
        """Edge case: 1 row without embed -> label should be 'Output'."""
        fake_preds = [[[("tok", 0.5)]]]
        output = _with_test_console(
            display_lens, ["A"], fake_preds, "A",
            lens_type="Tuned", has_embed_layer=False,
        )
        assert "Output" in output

    def test_logit_label_two_rows(self):
        """Embed + 1 layer -> labels: 'Embed', 'L0 (out)'."""
        fake_preds = [
            [[("emb", 0.1)]],
            [[("out", 0.9)]],
        ]
        output = _with_test_console(
            display_lens, ["X"], fake_preds, "X",
            lens_type="Logit", has_embed_layer=True,
        )
        assert "Embed" in output
        assert "L0 (out)" in output

    def test_display_with_empty_token(self):
        """Display should not crash with an empty token string."""
        fake_preds = [[[("", 0.3)]]]
        # Should not raise
        _with_test_console(
            display_lens, ["test"], fake_preds, "test",
            lens_type="Logit", has_embed_layer=True,
        )

    def test_display_prompt_with_rich_markup_chars(self):
        """Prompt containing [ ] shouldn't break Rich rendering."""
        fake_preds = [[[("a", 0.5)]]]
        output = _with_test_console(
            display_lens, ["["], fake_preds, "test [bracket]",
            lens_type="Logit", has_embed_layer=True,
        )
        # Should not crash and should contain the lens name
        assert "Logit Lens" in output


# ===================================================================
# Model loading -- integration tests
# ===================================================================

class TestModelLoading:
    def test_load_hf_tuned_lens(self):
        import tuned_lens
        test_console, buf = _make_console_and_buf()
        orig = tuned_lens.console
        tuned_lens.console = test_console
        try:
            hf_model, tokenizer, tuned = load_hf_model_and_tuned_lens("gpt2")
        finally:
            tuned_lens.console = orig
        assert hf_model.config.n_layer == 12
        assert tokenizer is not None
        assert len(tuned) == 12


# ===================================================================
# Training -- integration tests
# ===================================================================

class TestTraining:
    def test_train_and_load_roundtrip(self):
        """Train a tiny lens, save it, load it, verify it produces output."""
        import tuned_lens
        test_console, _ = _make_console_and_buf()
        orig = tuned_lens.console
        tuned_lens.console = test_console

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                train_tuned_lens(
                    model_name="gpt2",
                    out_dir=tmpdir,
                    epochs=1,
                    batch_size=2,
                    lr=1e-3,
                    max_length=64,
                    num_samples=4,
                )
            finally:
                tuned_lens.console = orig

            # Verify files exist
            assert (pathlib.Path(tmpdir) / "tuned_lens_probes.pt").exists()

            # Load and run inference with the trained lens
            tuned_lens.console = test_console
            try:
                hf_model, tokenizer, tuned = load_hf_model_and_tuned_lens(
                    "gpt2", lens_path=tmpdir
                )
            finally:
                tuned_lens.console = orig

            input_tokens, layer_preds = run_tuned_lens(
                hf_model, tokenizer, tuned, "Hello", top_k=1
            )
            assert len(input_tokens) >= 1
            assert len(layer_preds) == hf_model.config.n_layer + 1

    def test_train_loss_decreases(self):
        """Training for 2 epochs on tiny data -- loss in epoch 2 < epoch 1."""
        import tuned_lens
        test_console, buf = _make_console_and_buf()
        orig = tuned_lens.console
        tuned_lens.console = test_console

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                train_tuned_lens(
                    model_name="gpt2",
                    out_dir=tmpdir,
                    epochs=2,
                    batch_size=2,
                    lr=1e-3,
                    max_length=64,
                    num_samples=8,
                )
            finally:
                tuned_lens.console = orig

            output = _strip_ansi(buf.getvalue())
            losses = re.findall(r"avg KL loss: ([\d.]+)", output)
            assert len(losses) == 2, f"Expected 2 loss values, got: {losses}"
            loss1, loss2 = float(losses[0]), float(losses[1])
            assert loss2 < loss1, (
                f"Loss should decrease: epoch1={loss1}, epoch2={loss2}"
            )

    def test_train_num_samples_limits_data(self):
        """--num-samples should limit the number of training samples used."""
        import tuned_lens
        test_console, buf = _make_console_and_buf()
        orig = tuned_lens.console
        tuned_lens.console = test_console

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                train_tuned_lens(
                    model_name="gpt2",
                    out_dir=tmpdir,
                    epochs=1,
                    batch_size=2,
                    lr=1e-3,
                    max_length=64,
                    num_samples=6,
                )
            finally:
                tuned_lens.console = orig

            output = _strip_ansi(buf.getvalue())
            assert "6 samples" in output

    def test_train_generates_loss_plot(self):
        """Training with >1 batch should generate a loss_curves.png."""
        import tuned_lens
        test_console, _ = _make_console_and_buf()
        orig = tuned_lens.console
        tuned_lens.console = test_console

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                train_tuned_lens(
                    model_name="gpt2",
                    out_dir=tmpdir,
                    epochs=2,
                    batch_size=2,
                    lr=1e-3,
                    max_length=64,
                    num_samples=8,
                )
            finally:
                tuned_lens.console = orig

            plot_path = pathlib.Path(tmpdir) / "loss_curves.png"
            assert plot_path.exists(), "loss_curves.png should be generated"
            assert plot_path.stat().st_size > 1000, "Plot file should not be trivially small"
