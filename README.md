# Tuned Lens

**From-scratch PyTorch implementation**

## Overview

The tuned lens (Belrose et al., 2023) is basically a fix for the biggest problem with the [logit lens](https://github.com/aragorn-w/logit-lens). The logit lens projects each layer's residual stream through the final unembedding to get vocabulary predictions, but those intermediate layers were never trained to produce representations that play nicely with the final LayerNorm and unembedding. So early-layer predictions come out noisy and unreliable -- not because the layers lack information, but because the decoding doesn't match how those layers actually represent things.

The tuned lens fixes this by learning a small affine probe per layer that compensates for this mismatch. Each probe starts as an identity transform (so untrained tuned lens ≈ logit lens), then gets trained to minimize the KL divergence between its output distribution and the model's final output. After a few epochs on a small corpus, earlier layers become *much* more readable.

## From-Scratch Implementation

This repository contains a **from-scratch PyTorch implementation** of the tuned lens. It does **not** depend on the `tuned-lens` library. The core computation is:

```
logits_l = W_U * LN(W_l * h_l + b_l)
```

where:
- `h_l` is the residual-stream hidden state at layer `l`
- `W_l`, `b_l` are the learned per-layer affine probe parameters (initialized to identity matrix and zero vector, respectively)
- `LN` is the frozen final LayerNorm from the pretrained model
- `W_U` is the frozen unembedding matrix from the pretrained model

The probes are trained by minimizing KL divergence between the probe's output distribution and the model's final output distribution -- basically asking each probe to reproduce the model's final answer using only what's available at that layer.

> **Note:** Currently supports GPT-2 family models. Other architectures would need different attribute mappings for the LayerNorm and unembedding.

## Architecture

The `TunedLens` module (`nn.Module`) contains:

- **Per-layer affine probes**: An `nn.ModuleList` of `nn.Linear(d_model, d_model)` layers, one per transformer block. Each probe is initialized with an identity weight matrix and zero bias, so the untrained tuned lens behaves like a logit lens.
- **Frozen LayerNorm**: The model's final `LayerNorm`, with all parameters frozen (`requires_grad=False`).
- **Frozen unembedding**: The model's `lm_head` linear layer, with all parameters frozen.

For GPT-2 (12 layers, d_model=768), this amounts to approximately **7.1M trainable parameters** (768 x 768 + 768 per layer, times 12 layers), while the frozen components contribute zero trainable parameters.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/aragorn-w/tuned-lens
cd tuned-lens
uv sync
```

## Usage

### Visualization

Run the tuned lens on a prompt and display a colored terminal heatmap showing per-layer predictions:

```bash
# Using identity-initialized probes (no training)
uv run python tuned_lens.py visualize "The Eiffel Tower is located in the city of"

# Using pre-trained probes
uv run python tuned_lens.py visualize --lens-path ./tuned_lens_weights "The capital of France is"

# With top-5 predictions per cell
uv run python tuned_lens.py visualize --lens-path ./tuned_lens_weights -k 5 "Hello world"
```

### Training

Train tuned lens probes from scratch on The Pile (10k subset):

```bash
uv run python tuned_lens.py train --epochs 4 --out ./tuned_lens_weights
```

## Training Details

Probes are trained on [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k), a 10,000-sample subset of The Pile. The training procedure:

1. Load a frozen pretrained GPT-2 model.
2. Initialize identity affine probes for each of the 12 transformer layers.
3. For each batch, run a forward pass through the frozen model to collect all intermediate hidden states and the final output logits.
4. For each layer, apply the probe to the hidden state, compute `softmax` over the vocabulary, and compute the KL divergence against the model's final output distribution.
5. Sum the per-layer KL losses, backpropagate through only the probe parameters, and update with Adam.

Default hyperparameters:
- **Optimizer**: Adam, lr=1e-3
- **Batch size**: 4
- **Max sequence length**: 512 tokens
- **Epochs**: 4 (for included weights)

Training loss drops steadily across epochs. Later layers converge faster and to lower loss since their representations are already closer to the final output space.

## Pre-trained Weights

The `tuned_lens_weights/` directory contains weights trained for 4 epochs on Pile-10k with the default hyperparameters:

- `tuned_lens_probes.pt` -- Serialized probe state dict (per-layer weight matrices and bias vectors)
- `loss_curves.png` -- Training loss visualization (per-batch, per-epoch, and per-layer)

These weights can be used directly for visualization without retraining.

## Running Tests

```bash
uv run pytest test_tuned_lens.py -v
```

The test suite includes 47+ tests covering:
- `prob_to_bg_fg` color mapping (18 tests)
- `sanitize_token` display formatting (12 tests)
- `run_tuned_lens` inference (6 tests)
- `TunedLens` module internals (6 tests)
- `display_lens` rendering (8 tests)
- Model loading (1 test)
- Training pipeline (4 tests)

Note: Integration tests download GPT-2 (~500MB) and Pile-10k on first run.

## What I Found

Comparing the tuned lens side-by-side with the logit lens on the same prompts:

- **Early layers become actually readable.** The logit lens at layers 0-3 is basically garbage -- just common tokens regardless of the prompt. With trained probes, those same layers start showing meaningful predictions. For "The Eiffel Tower is located in the city of", layer 2 with the tuned lens already shows geographic-related tokens, whereas the raw logit lens at layer 2 is just predicting "the" and "of".

- **The "click" moment moves earlier.** With the logit lens, the correct answer typically snaps into place around layer 8-9. The tuned lens shows the model already has a strong signal for the right answer a few layers sooner -- it was there in the representation, the logit lens just couldn't decode it.

- **Later layers don't change much.** The trained probes for layers 9-11 are nearly identity (their loss is already very low at initialization). This confirms that the logit lens works fine for later layers -- the distributional shift problem is mainly an early/middle layer issue.

- **Per-layer loss curves are revealing.** During training, layer 0 starts with the highest KL divergence and improves the most, while layer 11 barely moves. The spread between early and late layers narrows over epochs, which is exactly what should happen if the probes are learning to compensate for the representation shift.

## Why This Is Useful

The tuned lens gives a much more honest picture of what each layer knows. The logit lens makes it look like early layers are clueless, but that's an artifact of the decoding mismatch -- the information is there, it's just not in the right format. With tuned probes, it becomes possible to actually study how representations evolve across the full depth of the network, not just the last few layers where the logit lens happens to work.

This is especially relevant for understanding *where* different types of knowledge live in a transformer. If a tuned probe at layer 4 can already predict the right answer for a factual prompt, that tells something about how deep factual recall actually needs to go vs. how deep the model actually processes it.

## Problems and Limitations

- **Requires training data.** Unlike the logit lens (which is zero-cost), the tuned lens needs a training corpus and GPU time. The probes are small, but it's still an extra step.

- **Possible overfitting.** The probes could in theory memorize patterns from the training data rather than learning a genuine "translation" of the representation space. Using a large, diverse corpus (The Pile) helps, but it's worth being aware of.

- **Probe capacity matters.** An affine probe is a strong assumption -- it says "the relationship between intermediate and final representations is approximately linear." If that assumption is wrong (maybe some layers do something highly nonlinear), the probe accuracy will be limited. A nonlinear probe could potentially extract more, but then the question becomes whether the probe itself is doing the computation, which defeats the purpose.

- **KL divergence as an objective isn't perfect.** The probes are trained to match the model's *full output distribution*, not just the top prediction. This means the probe might spend capacity matching the tail of the distribution (low-probability tokens) rather than getting the top-1 prediction right, which is usually what I care about when visualizing.

## References

- Belrose, N., Furman, Z., Smith, L., Halawi, D., Oesterling, A., McKinney-Bock, K., Levi, T., Steinhardt, J., & Hastings, P. K. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens*. arXiv:2303.08112.

- nostalgebraist (2020). *interpreting GPT: the logit lens*. LessWrong. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
