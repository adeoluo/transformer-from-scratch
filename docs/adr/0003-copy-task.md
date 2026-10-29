# ADR-0003: A copy task that an n-gram model provably cannot solve

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
To show what the transformer learns, the task must have a clear, measurable property that
depends on attention. Natural text would require a large model and vocabulary and would make the
contribution of attention hard to isolate.

## Decision
Train on a copy task: a random string, a separator, then the same string repeated. The headline
metric is next-token accuracy on the second copy, where correct prediction requires attending
back to the matching position in the first copy. A fixed-order bigram model, estimated from the
same data, is the baseline. It captures local statistics but cannot copy, so the accuracy gap is
exactly the value of attention.

## Consequences
- The comparison is honest and sharp: the bigram baseline is near chance on the copy region
  (0.234), while the transformer reaches 0.903.
- Splits are generated with seeded randomness and are disjoint, so held-out accuracy measures
  generalization, not memorization.
- The task is synthetic and small; the claims are about mechanisms, not language modeling at
  scale, which the README states plainly.
