# ADR-0002: A matrix-valued autodiff engine, not scalar autograd

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
The project needs reverse-mode automatic differentiation with no external library. The simplest
teaching approach is scalar autograd, where every float is a node in the graph. That is elegant
but far too slow for a transformer: a single attention matrix multiply would create thousands of
scalar nodes, and training would take hours in pure Python.

## Decision
Build the engine at the granularity of dense 2-D tensors. Each operation (matmul, biased affine
map, causal softmax, layer norm, GELU, embedding lookup, concatenation, cross-entropy) is a
single graph node with a hand-written backward closure that operates on whole matrices. A
topological sort runs the closures in reverse to accumulate gradients.

## Consequences
- The forward pass reads like standard linear algebra, and each backward closure sits next to
  its forward, so the mathematics is inspectable.
- Training a small model is seconds, not hours, which keeps every result reproducible on CPU.
- The engine is not a general tensor library: it is exactly the operations a decoder-only
  transformer needs. Adding a new architecture means adding and gradient-checking new ops.
- Every operation is validated by a finite-difference gradient check in the test suite, so the
  hand-written backward passes are held to a numerical standard rather than trusted.
