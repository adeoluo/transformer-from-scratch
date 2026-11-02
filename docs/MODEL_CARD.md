# Model / System Card - Transformer From Scratch

## Overview
- **What it is:** a from-scratch reverse-mode autodiff engine and a small decoder-only
  transformer language model, trained on a synthetic copy task. Hand-written forward and
  backward passes over numpy arrays; no automatic differentiation library.
- **Base model:** none. Parameters are initialized and trained from scratch.
- **Version and date:** 0.1.0, 2026.

## Intended use
- **Primary use case:** a readable, reproducible demonstration of backpropagation through
  attention and of what attention adds over an n-gram model.
- **Primary users:** engineers and researchers learning transformer internals or teaching them.

## Out-of-scope / prohibited use
- Not a language model for real text; the vocabulary and task are synthetic.
- Not a performance reference; it runs one sequence at a time on the CPU.
- The accuracy numbers describe a copy task and must not be cited as a language benchmark.

## Performance
- Held-out copy accuracy 0.872 versus 0.234 for a bigram baseline (seed 0; 0.984 ± 0.041 over
  8 seeds in the research study).
- Gradients verified against finite differences to better than 1e-4.
- Full method and tables: [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

## Limitations and failure modes
- Tiny model and synthetic task; no claims about scale or natural language.
- No key-value cache, so generation cost is quadratic in sequence length.
- The copy mechanism is a positional lookup by one or two layer-1 heads, not an induction
  circuit; see [`RESEARCH.md`](RESEARCH.md).

## Ethical considerations
- No personal data, no network access, no external model. The task is synthetic symbols.
