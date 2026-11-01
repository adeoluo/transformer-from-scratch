# Case study: validating a from-scratch transformer

This note records the engineering decisions behind the transformer and, more
importantly, the evidence that it is correct and that it learns. The goal is not
a large model; it is a small model whose every component is verified.

## The problem

Implement a decoder-only transformer and its training stack — reverse-mode
automatic differentiation, multi-head causal attention, pre-norm residual blocks,
a GELU feed-forward network, layer normalization, weight-tied embeddings, and an
Adam optimizer — in pure Python with no numerical libraries, and prove it is
correct.

"Prove it is correct" is the hard part. A model that trains to a low loss can
still hide a wrong gradient that happens to be close enough. The portfolio value
is in the quality of the evidence, not the size of the model.

## Decisions

- **Explicit matrix autodiff.** Every parameter is a `Tensor` that records the
  operation that produced it, so one `loss.backward()` differentiates the whole
  network. This is slower than a vectorized engine but makes each operation and
  its gradient readable and independently testable. See
  [ADR 0002](adr/0002-matrix-autodiff.md).
- **The copy task.** The headline synthetic task requires attending back to an
  earlier position, which an n-gram model provably cannot do, so the gap between
  the transformer and a bigram baseline isolates what attention buys. See
  [ADR 0003](adr/0003-copy-task.md).
- **Tanh GELU and pre-norm.** These match a common modern configuration and are
  reproduced exactly in the PyTorch reference, so parity is meaningful.

## Evidence the implementation is correct

1. **Finite-difference gradient checks** (`tests/test_autograd.py`). Each
   operation's analytic gradient matches a central finite-difference estimate,
   the standard check for a hand-written backward pass.
2. **PyTorch numerical parity** (`tests/test_torch_parity.py`). The exact forward
   pass is reconstructed in PyTorch with the same weights; logits and
   cross-entropy match to better than `1e-9`. This is an independent reference
   implementation, not a self-consistency check. PyTorch is an opt-in validation
   dependency, so the core library stays zero-dependency.

## Evidence it learns

1. **Synthetic copy task.** Held-out copy accuracy reaches 0.903 versus 0.234 for
   a bigram baseline fit on the same data, a gap that exists only because
   attention can copy.
2. **Attention sharpening.** The sharpest head's peak weight over the copy region
   rises from 0.425 to 0.981 and its entropy collapses from 1.84 to 0.00, a
   direct readout of learned, content-dependent attention.
3. **Real text.** At the character level on a public-domain prose excerpt, the
   model compresses held-out text to 3.639 bits per character, 0.663 bits better
   than a unigram baseline, so it has learned real conditional structure in
   English rather than only a synthetic pattern.

## Honest limitations

- The model is deliberately tiny and CPU-only; this is a correctness and learning
  demonstration, not a performance result. A vectorized or compiled backend and a
  GPU would be required for larger corpora and competitive throughput.
- The real-text corpus is a small bundled excerpt; the result shows the
  architecture learns language, not a state-of-the-art language model.
- No key-value cache, so generation is quadratic in sequence length.

## What this demonstrates

The ability to implement a non-trivial system from first principles and, crucially,
to prove it works: analytic gradients checked against finite differences and an
independent framework, learning demonstrated on both a controlled synthetic task
and real text, and every claim backed by a reproducible, deterministic experiment.
