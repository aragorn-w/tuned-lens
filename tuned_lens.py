#!/usr/bin/env python3
"""
Tuned Lens — From-Scratch Implementation

A from-scratch PyTorch implementation of the Tuned Lens technique introduced by
Belrose et al. (2023) in "Eliciting Latent Predictions from Transformers with
the Tuned Lens." This module provides training, inference, and terminal-based
visualization of learned affine probes that map intermediate transformer
representations to output-vocabulary predictions.

Unlike the logit lens (nostalgebraist, 2020), which applies the unembedding
matrix directly to intermediate residual-stream states, the tuned lens learns a
per-layer affine transformation (initialized to identity) that compensates for
the distributional shift across layers:

    logits_l = W_U * LN(W_l * h_l + b_l)

where W_l, b_l are the learned per-layer affine probes, LN is the frozen final
LayerNorm, and W_U is the frozen unembedding matrix. The probes are trained to
minimize the KL divergence between their output distribution and the model's
final output distribution.

Usage:
    uv run python tuned_lens.py visualize "The capital of France is"
    uv run python tuned_lens.py visualize --lens-path ./tuned_lens_weights "Hello world"
    uv run python tuned_lens.py train --epochs 4 --out ./tuned_lens_weights
"""

import contextlib
import io
import logging
import math
import pathlib
import warnings

# Suppress noisy deprecation warnings before any library imports
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
logging.disable(logging.WARNING)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

console = Console(soft_wrap=True)
app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Tuned Lens (from-scratch implementation)
# ---------------------------------------------------------------------------

class TunedLens(nn.Module):
    """
    Tuned Lens (Belrose et al., 2023) — from-scratch implementation.

    Per-layer affine probes initialized to identity transform, plus frozen
    final LayerNorm and unembedding from the model. Forward pass:
        logits = unembed(LN(W_i @ h_i + b_i))
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        layer_norm: nn.LayerNorm,
        unembed: nn.Linear,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model

        # Per-layer trainable affine probes: initialized to identity
        self.probes = nn.ModuleList()
        for _ in range(n_layers):
            probe = nn.Linear(d_model, d_model)
            nn.init.eye_(probe.weight)
            nn.init.zeros_(probe.bias)
            self.probes.append(probe)

        # Frozen model components (not saved as part of probe weights)
        self.layer_norm = layer_norm
        self.unembed = unembed
        for p in self.layer_norm.parameters():
            p.requires_grad = False
        for p in self.unembed.parameters():
            p.requires_grad = False

    def forward(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply probe at layer_idx to hidden state h -> logits."""
        transformed = self.probes[layer_idx](h)  # (batch, seq, d_model)
        normed = self.layer_norm(transformed)     # (batch, seq, d_model)
        logits = self.unembed(normed)             # (batch, seq, vocab)
        return logits

    def probe_parameters(self):
        """Return only the trainable probe parameters (not LN/unembed)."""
        return self.probes.parameters()

    def save(self, path: pathlib.Path) -> None:
        """Save probe weights to disk."""
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "probes": self.probes.state_dict(),
        }
        torch.save(state, path / "tuned_lens_probes.pt")

    @classmethod
    def from_model(cls, hf_model) -> "TunedLens":
        """Create a TunedLens with identity-initialized probes from a HuggingFace GPT-2 model.

        Currently supports GPT-2 family models. Other architectures would require
        mapping model-specific attribute names (layer norm, unembedding, config keys).
        """
        n_layers = hf_model.config.n_layer
        d_model = hf_model.config.n_embd

        # Extract frozen LN and unembed from model
        layer_norm = hf_model.transformer.ln_f
        # GPT-2 uses a tied weight matrix (no bias) for lm_head
        unembed = hf_model.lm_head

        lens = cls(n_layers, d_model, layer_norm, unembed)
        return lens

    @classmethod
    def from_pretrained(cls, hf_model, path: str | pathlib.Path) -> "TunedLens":
        """Load trained probe weights from disk."""
        path = pathlib.Path(path)
        lens = cls.from_model(hf_model)
        state = torch.load(path / "tuned_lens_probes.pt", weights_only=True)
        lens.probes.load_state_dict(state["probes"])
        return lens

    def __len__(self):
        return self.n_layers


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_hf_model_and_tuned_lens(
    model_name: str,
    lens_path: str | None = None,
) -> tuple:
    """Load a HuggingFace model + TunedLens probes (trained from disk or identity)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = f"local ({lens_path})" if lens_path else "identity-initialized"
    console.print(
        f"[bold]Loading {model_name} + tuned lens ({source})...[/bold]"
    )
    with (
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model.eval()
    for param in hf_model.parameters():
        param.requires_grad = False

    if lens_path:
        tuned = TunedLens.from_pretrained(hf_model, lens_path)
    else:
        tuned = TunedLens.from_model(hf_model)
    tuned.eval()

    n_layers = hf_model.config.n_layer
    d_model = hf_model.config.n_embd
    console.print(
        f"[green]Loaded {model_name} + tuned lens "
        f"({n_layers} layers, d_model={d_model})[/green]\n"
    )
    return hf_model, tokenizer, tuned


# ---------------------------------------------------------------------------
# Tuned lens (inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_tuned_lens(
    hf_model, tokenizer, tuned_lens, prompt: str, top_k: int = 1
) -> tuple[list[str], list[list[list[tuple[str, float]]]]]:
    """
    Run the tuned lens: apply learned affine probes at each layer.

    Returns:
        input_tokens: list of string tokens from the prompt
        layer_predictions: [n_layers+1][seq_len] -> list of (token_str, prob) top-k
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = hf_model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # (n_layers+1,) each (1, seq, d_model)

    layer_predictions: list[list[list[tuple[str, float]]]] = []
    n_layers = hf_model.config.n_layer

    for layer_idx in range(n_layers):
        h = hidden_states[layer_idx + 1]
        logits = tuned_lens.forward(h, layer_idx)
        probs = torch.softmax(logits[0], dim=-1)
        top_probs, top_indices = probs.topk(top_k, dim=-1)

        pos_preds = []
        for pos in range(probs.shape[0]):
            preds = []
            for k in range(top_k):
                tok_str = tokenizer.decode(top_indices[pos, k].item())
                preds.append((tok_str, top_probs[pos, k].item()))
            pos_preds.append(preds)
        layer_predictions.append(pos_preds)

    # Final layer: use model's own output logits (no probe needed)
    final_probs = torch.softmax(outputs.logits[0], dim=-1)
    top_probs, top_indices = final_probs.topk(top_k, dim=-1)
    pos_preds = []
    for pos in range(final_probs.shape[0]):
        preds = []
        for k in range(top_k):
            tok_str = tokenizer.decode(top_indices[pos, k].item())
            preds.append((tok_str, top_probs[pos, k].item()))
        pos_preds.append(preds)
    layer_predictions.append(pos_preds)

    input_token_strs = [tokenizer.decode(t) for t in inputs["input_ids"][0]]
    return input_token_strs, layer_predictions


# ---------------------------------------------------------------------------
# Tuned lens training
# ---------------------------------------------------------------------------

def plot_loss_curves(
    batch_losses: list[float],
    epoch_avg_losses: list[float],
    per_layer_epoch_losses: list[list[float]],
    out_path: pathlib.Path,
    model_name: str,
) -> pathlib.Path:
    """Generate a polished loss curve PNG and return its path."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Tuned Lens Training — {model_name}",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # --- Panel 1: Per-batch loss (smoothed + raw) ---
    ax = axes[0]
    steps = np.arange(1, len(batch_losses) + 1)
    raw = np.array(batch_losses)
    ax.plot(steps, raw, alpha=0.2, color="#888888", linewidth=0.5, label="Raw")
    # Moving average smoothing
    window = max(1, len(batch_losses) // 50)
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(raw, kernel, mode="valid")
        offset = (window - 1) // 2
        ax.plot(
            steps[offset : offset + len(smoothed)],
            smoothed,
            color="#2176ff",
            linewidth=2,
            label=f"Smoothed (w={window})",
        )
    ax.set_xlabel("Batch", fontsize=11)
    ax.set_ylabel("KL Divergence", fontsize=11)
    ax.set_title("Batch Loss", fontsize=12)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # --- Panel 2: Per-epoch average loss ---
    ax = axes[1]
    epoch_nums = list(range(1, len(epoch_avg_losses) + 1))
    ax.plot(
        epoch_nums, epoch_avg_losses,
        "o-", color="#ff6b35", linewidth=2.5, markersize=8,
    )
    for i, loss in enumerate(epoch_avg_losses):
        ax.annotate(
            f"{loss:.3f}",
            (epoch_nums[i], loss),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#ff6b35",
        )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Avg KL Divergence", fontsize=11)
    ax.set_title("Epoch Loss", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # --- Panel 3: Per-layer loss at each epoch ---
    ax = axes[2]
    n_layers = len(per_layer_epoch_losses[0]) if per_layer_epoch_losses else 0
    cmap = plt.cm.viridis
    for layer_idx in range(n_layers):
        color = cmap(layer_idx / max(n_layers - 1, 1))
        layer_losses = [ep[layer_idx] for ep in per_layer_epoch_losses]
        ax.plot(
            epoch_nums, layer_losses,
            "o-", color=color, linewidth=1.5, markersize=4,
            label=f"L{layer_idx}",
        )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("KL Divergence", fontsize=11)
    ax.set_title("Per-Layer Loss", fontsize=12)
    ax.legend(
        fontsize=7, ncol=2, loc="upper right",
        framealpha=0.9, borderpad=0.3, handlelength=1.5,
    )
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    plot_path = out_path / "loss_curves.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def train_tuned_lens(
    model_name: str = "gpt2",
    out_dir: str = "./tuned_lens_weights",
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-3,
    max_length: int = 512,
    num_samples: int | None = None,
) -> None:
    """
    Train tuned lens affine probes on The Pile (10k).

    For each layer, trains a learned affine transform (A_i, b_i) so that
    ``unembed(LN(A_i @ h_i + b_i))`` approximates the model's final output
    distribution, minimizing KL divergence.
    """
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- Load model and data ---
    console.print(f"[bold]Loading {model_name}...[/bold]")
    with (
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model.eval()
    for param in hf_model.parameters():
        param.requires_grad = False

    n_layers = hf_model.config.n_layer
    console.print(f"[green]Loaded {model_name} ({n_layers} layers)[/green]")

    console.print("[bold]Loading NeelNanda/pile-10k...[/bold]")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        dataset = load_dataset("NeelNanda/pile-10k", split="train")
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    console.print(f"[green]Loaded {len(dataset)} samples[/green]")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Create untrained lens (identity-initialized probes) ---
    lens = TunedLens.from_model(hf_model)

    trainable_params = list(lens.probe_parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    console.print(
        f"[bold]Training {n_trainable:,} probe parameters "
        f"({n_layers} layers)[/bold]\n"
    )

    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    # --- Tracking ---
    all_batch_losses: list[float] = []
    epoch_avg_losses: list[float] = []
    per_layer_epoch_losses: list[list[float]] = []

    # --- Training loop ---
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        layer_loss_accum = [0.0] * n_layers
        layer_loss_counts = [0] * n_layers

        shuffled = dataset.shuffle(seed=42 + epoch)
        n_total = len(shuffled)
        n_full_batches = n_total // batch_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("loss={task.fields[loss]:.4f}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Epoch {epoch + 1}/{epochs}",
                total=n_full_batches,
                loss=0.0,
            )

            for batch_start in range(0, n_total - batch_size + 1, batch_size):
                batch_texts = [
                    shuffled[batch_start + i]["text"]
                    for i in range(batch_size)
                ]
                encoded = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                with torch.no_grad():
                    outputs = hf_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    target_log_probs = torch.log_softmax(
                        outputs.logits, dim=-1
                    )

                batch_loss = torch.tensor(0.0)
                target_probs = target_log_probs.exp()

                for layer_idx in range(n_layers):
                    h = outputs.hidden_states[layer_idx + 1].detach()
                    probe_logits = lens.forward(h, layer_idx)
                    probe_log_probs = torch.log_softmax(probe_logits, dim=-1)

                    kl = (
                        target_probs
                        * (target_log_probs - probe_log_probs)
                    ).sum(dim=-1)

                    masked_kl = (kl * attention_mask).sum()
                    n_tokens = attention_mask.sum()
                    if n_tokens > 0:
                        layer_kl = masked_kl / n_tokens
                        batch_loss = batch_loss + layer_kl
                        layer_loss_accum[layer_idx] += layer_kl.item()
                        layer_loss_counts[layer_idx] += 1

                batch_loss = batch_loss / n_layers

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss_val = batch_loss.item()
                all_batch_losses.append(loss_val)
                total_loss += loss_val
                n_batches += 1
                progress.update(
                    task, advance=1, loss=total_loss / n_batches
                )

        avg_loss = total_loss / max(n_batches, 1)
        epoch_avg_losses.append(avg_loss)

        layer_avgs = [
            layer_loss_accum[i] / max(layer_loss_counts[i], 1)
            for i in range(n_layers)
        ]
        per_layer_epoch_losses.append(layer_avgs)

        console.print(
            f"  Epoch {epoch + 1}/{epochs} — avg KL loss: {avg_loss:.4f}\n"
        )

    # --- Save weights ---
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    lens.save(out_path)
    console.print(f"[bold green]Saved tuned lens to {out_path}/[/bold green]")

    # --- Plot loss curves ---
    if len(all_batch_losses) > 1:
        plot_path = plot_loss_curves(
            all_batch_losses,
            epoch_avg_losses,
            per_layer_epoch_losses,
            out_path,
            model_name,
        )
        console.print(f"[bold green]Saved loss curves to {plot_path}[/bold green]")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def prob_to_bg_fg(prob: float) -> tuple[str, str]:
    """
    Map probability [0,1] to (background_rgb, foreground) using a smooth
    red-to-green gradient. Font color is black or white for contrast.
    """
    if math.isnan(prob):
        prob = 0.0
    elif math.isinf(prob):
        prob = 1.0 if prob > 0 else 0.0
    p = max(0.0, min(1.0, prob))

    if p < 0.5:
        t = p / 0.5
        r = 200
        g = int(160 * t)
        b = 0
    else:
        t = (p - 0.5) / 0.5
        r = int(200 * (1 - t))
        g = 160 + int(40 * t)
        b = 0

    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    fg = "black" if luminance > 60 else "white"

    bg = f"rgb({r},{g},{b})"
    return bg, fg


def sanitize_token(tok: str) -> str:
    """Make a token string safe and readable for terminal display."""
    s = repr(tok)[1:-1]
    if len(s) > 12:
        s = s[:11] + "\u2026"
    return s


def display_lens(
    input_tokens: list[str],
    layer_predictions: list[list[list[tuple[str, float]]]],
    prompt: str,
    lens_type: str = "Tuned",
    has_embed_layer: bool = False,
) -> None:
    """Render the lens output as a colored table in the terminal."""
    seq_len = len(input_tokens)
    n_rows = len(layer_predictions)

    console.print(
        f'[bold underline]{lens_type} Lens: "{escape(prompt)}"[/bold underline]\n'
    )

    table = Table(
        title="Top predicted next token at each layer",
        show_lines=True,
        padding=(0, 1),
        expand=False,
    )

    table.add_column("Layer", style="bold", width=10, no_wrap=True)
    for tok in input_tokens:
        table.add_column(
            sanitize_token(tok), no_wrap=True, min_width=8, max_width=16
        )

    for row_idx in range(n_rows):
        if has_embed_layer:
            # Logit lens: Embed, L0, L1, ..., L{n-1} (out)
            if row_idx == 0:
                label = "Embed"
            elif row_idx == n_rows - 1:
                label = f"L{row_idx - 1} (out)"
            else:
                label = f"L{row_idx - 1}"
        else:
            # Tuned lens: L0, L1, ..., L{n-1}, Output
            if row_idx == n_rows - 1:
                label = "Output"
            else:
                label = f"L{row_idx}"

        row_cells: list[str | Text] = [label]
        for pos in range(seq_len):
            preds = layer_predictions[row_idx][pos]
            top_prob = preds[0][1]
            bg, fg = prob_to_bg_fg(top_prob)

            lines = []
            for tok_str, _ in preds:
                lines.append(sanitize_token(tok_str))
            cell = Text("\n".join(lines), style=f"{fg} on {bg}")
            row_cells.append(cell)
        table.add_row(*row_cells)

    console.print(table)

    # Smooth gradient legend
    console.print()
    legend = Text("Confidence: ", style="bold")
    legend.append("0%", style="bold")
    for pct in range(0, 101):
        bg, fg = prob_to_bg_fg(pct / 100.0)
        legend.append("\u2588", style=bg)
    legend.append(" 100%", style="bold")
    console.print(legend)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def visualize(
    prompt: str = typer.Argument(
        default="The Eiffel Tower is located in the city of",
        help="Text prompt to analyze.",
    ),
    model: str = typer.Option(
        "gpt2", "--model", "-m",
        help="Model name (e.g. gpt2, gpt2-medium).",
    ),
    top_k: int = typer.Option(
        1, "--top-k", "-k",
        help="Number of top predictions to show per cell.",
        min=1, max=5,
    ),
    lens_path: str = typer.Option(
        None, "--lens-path", "-l",
        help="Path to locally-trained tuned lens weights.",
    ),
) -> None:
    """Visualize how predictions evolve across transformer layers."""
    hf_model, tokenizer, tuned_lens = load_hf_model_and_tuned_lens(
        model, lens_path=lens_path
    )
    input_tokens, layer_predictions = run_tuned_lens(
        hf_model, tokenizer, tuned_lens, prompt, top_k=top_k
    )
    display_lens(
        input_tokens, layer_predictions, prompt,
        lens_type="Tuned", has_embed_layer=False,
    )


@app.command()
def train(
    model: str = typer.Option(
        "gpt2", "--model", "-m",
        help="HuggingFace model to train probes for.",
    ),
    out: str = typer.Option(
        "./tuned_lens_weights", "--out", "-o",
        help="Directory to save trained lens weights.",
    ),
    epochs: int = typer.Option(
        1, "--epochs", "-e",
        help="Number of training epochs.", min=1,
    ),
    batch_size: int = typer.Option(
        4, "--batch-size", "-b",
        help="Batch size for training.", min=1,
    ),
    lr: float = typer.Option(
        1e-3, "--lr",
        help="Learning rate for Adam optimizer.",
    ),
    max_length: int = typer.Option(
        512, "--max-length",
        help="Max token sequence length per sample.", min=32,
    ),
    num_samples: int = typer.Option(
        None, "--num-samples", "-n",
        help="Limit number of training samples (default: all 10k).",
    ),
) -> None:
    """Train tuned lens probes on The Pile (10k) dataset."""
    train_tuned_lens(
        model_name=model,
        out_dir=out,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        num_samples=num_samples,
    )


if __name__ == "__main__":
    app()
