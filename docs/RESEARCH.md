# What makes copying work: a causal study of induction in a small transformer

## Abstract

The repository's 2-layer transformer copies a random string with about 0.98 held-out accuracy.
This study asks how. I pre-registered three hypotheses (one head is causally necessary; two
layers are required; content-based attention is required), fixed numeric decision thresholds
in code, and then trained 5 architectures for 8 seeds each at the same step budget (40 runs),
ablated every attention head of the 2-layer models, and probed their attention patterns.

The headline is a negative result. **The model does not use an induction circuit.** Copying
is done by one or two *layer-1* heads that attend to a fixed relative position (exactly
`length` tokens back), with near-zero attention on look-alike positions; layer 2 is almost
dispensable (ablating all of its heads costs 0.025 accuracy). Consistent with that, **a
1-layer model copies as well as the 2-layer model** at matched steps (0.967 ± 0.090 vs
0.976 ± 0.034) and at matched parameters (0.976 ± 0.041), so the depth hypothesis is NOT
SUPPORTED. The single-head necessity hypothesis is MIXED: ablating the critical layer-1 head
collapses accuracy to near the bigram baseline in 5 of 8 seeds, but in the others the job is
shared by redundant heads, and the "sharpest" head is not always the important one. Replacing
attention with a content-independent mean-pooling mixer cuts accuracy roughly in half
(0.520 ± 0.026) but stays far above the bigram baseline (0.250 ± 0.012), so H3 is MIXED as
well.

## Research question

In this model, is the copy behaviour implemented by a small, identifiable set of attention
heads (an induction-style circuit), and what is the minimal architecture that learns it?

The repository's headline result is that a 2-layer transformer reaches about 0.90 held-out
copy accuracy against 0.23 for a bigram model. That number says the model copies; it does not
say *how*, or whether the architecture is the smallest one that would. This study tests three
specific, falsifiable claims about mechanism and architecture.

## Pre-registered hypotheses and predictions

This section, the method, and the decision thresholds were written and committed to the
branch before any full-budget run was made. The thresholds live as constants at the top of
[`src/transformer_from_scratch/research.py`](../src/transformer_from_scratch/research.py), and
the verdict functions apply them mechanically, so a verdict cannot be argued into shape after
the numbers are known. "Copy accuracy" is held-out accuracy on the copy region (the second half
of each sequence). All thresholds are absolute differences in copy accuracy.

**H1, causal necessity.** Zeroing the output of the single sharpest attention head (the head
with the highest mean peak attention weight on copy-region queries) drops held-out copy
accuracy to near the bigram baseline, while zeroing any other single head costs little.

- Prediction: with the sharpest head ablated, copy accuracy is within 0.15 of the bigram
  baseline; every other single-head ablation costs less than 0.10.
- Verdict rule: SUPPORTED if both parts hold in every informative seed; NOT SUPPORTED if the
  necessity part fails in every informative seed; MIXED otherwise. A seed is informative only
  if its intact model is itself more than 0.15 above the bigram baseline (otherwise "drops to
  near the baseline" is true by default). This informativeness condition was added after a
  5-step smoke test of the code, in which an untrained model trivially "passed", and before
  any real run.

**H2, depth requirement.** A 1-layer model cannot learn the copy task as well as a 2-layer
model at a matched training-step budget, nor at a matched parameter budget. The motivation is
the published account of induction heads, which needs two heads composed across layers (a
previous-token head feeding a head that matches on the shifted key).

- Prediction: the 2-layer model beats both the same-width 1-layer model (matched steps,
  about half the parameters) and a 1-layer model with a wider feed-forward layer (matched
  parameters) by more than 0.10 mean copy accuracy, a gap also larger than either arm's
  standard deviation across seeds.
- Verdict rule: SUPPORTED if both gaps are clear in that sense; NOT SUPPORTED if both gaps
  are at most 0.05; MIXED otherwise.
- Known risk, recorded before running: this task uses fixed-length sequences with learned
  absolute position embeddings, so the source of every copied token is always exactly
  `length` positions back. A single head that attends by position alone could solve the task
  without any composition. If that happens, H2 should fail, and that would be a statement
  about this task, not about induction in general.

**H3, attention is necessary.** Replacing softmax attention with a uniform causal
mean-pooling mixer (each position averages the value vectors of all positions up to itself,
with no query or key), at a matched budget, fails the copy task.

- Prediction: the uniform-mixer model's mean copy accuracy is within 0.10 of the bigram
  baseline.
- Verdict rule: SUPPORTED if so; NOT SUPPORTED if it comes within 0.10 of the attention model;
  MIXED if it lands in between (clearly worse than attention, clearly better than bigram).

**Attention-pattern probe (descriptive, no verdict).** For the head whose ablation costs the
most, measure the fraction of attention mass on the *source* position (the token right after
the earlier occurrence of the current token in the aligned first copy, which is the token to
predict). Because symbols repeat (4 symbols, length 5), a query often has *distractors*:
positions right after some other earlier occurrence of the same token. A purely content-based
induction head should split its mass between source and distractors; a head that also uses
position should put almost none on distractors. The probe reports both.

## Method

**Task.** The repository's copy task: 5 random symbols from an alphabet of 4, a separator,
then the same 5 symbols again (11 tokens; the model sees 10 inputs and predicts the next token
at each). The copy region is the 5 targets after the separator. Training uses 64 distinct
sequences; the held-out set is 128 sequences disjoint from training. Both splits, the model
initialization, and the minibatch order are all derived from the run seed.

**Arms.** Every arm is trained with the same budget: 300 Adam steps, batch 16, learning rate
0.02, which is the repository's default recipe. Widths are the default (`d_model` 24, 3
heads, `d_ff` 48) unless stated.

| Arm | What changes | Purpose |
|---|---|---|
| `2L-attn` | nothing (the default model) | reference; also ablated and probed |
| `1L-attn` | 1 layer | H2 at matched steps |
| `1L-attn-wide` | 1 layer, `d_ff` widened so the parameter count matches `2L-attn` | H2 at matched parameters |
| `3L-attn` | 3 layers | does more depth help further? |
| `2L-uniform` | softmax(QK) replaced by fixed causal mean pooling | H3 |

The uniform mixer keeps every other component (value and output projections, feed-forward,
norms, residuals). Its query and key matrices still exist so that the random initialization
stream is identical, but they never affect the output and receive no gradient; the tables
report both the nominal and the effective parameter count.

**Ablation.** For each trained `2L-attn` model, each head's output is set to zero before the
output projection (so only the projection bias remains of it), and held-out copy accuracy is
measured. This is repeated for every one of the 6 heads, for all 3 heads of each layer at once,
and for all 6 heads. Ablation is applied at evaluation only; the model is not retrained.

**Seeds.** 8 seeds (0 to 7) per arm, 40 training runs in total. Results are reported as mean
plus or minus sample standard deviation across seeds, with per-seed values where head
identity matters (head indices are arbitrary from one seed to the next).

**Reproduce.**

```bash
uv run python benchmarks/research_study.py --json docs/research_results.json
uv run python benchmarks/research_study.py --seeds 0 1 2 --steps 300 --workers 1   # cheaper
```

Runs are independent and individually seeded, so `--workers` changes wall time only. The raw
per-run numbers behind every table below are in
[`docs/research_results.json`](research_results.json).

## Results

All numbers are held-out copy accuracy on 128 sequences per seed (a different held-out set
for each seed), reported as mean ± sample standard deviation over 8 seeds. The tables below
are the unedited output of `benchmarks/research_study.py`; the raw per-run values are in
[`research_results.json`](research_results.json). For reference, seed 0 of `2L-attn` scores
0.897 here against 0.903 in the README, which evaluates on a smaller held-out set (64).

Seeds: 0, 1, 2, 3, 4, 5, 6, 7 (n = 8)

### Architecture comparison (matched training steps)

| Arm | Layers | Mixer | d_ff | Params (effective) | Copy acc (mean ± std) | Per seed | Overall acc | Final train loss |
|---|---:|---|---:|---:|---:|---|---:|---:|
| 2L-attn | 2 | attention | 48 | 10032 (10032) | 0.976 ± 0.034 | 0.897, 0.994, 0.978, 1.000, 0.995, 0.989, 0.989, 0.969 | 0.683 ± 0.024 | 0.523 ± 0.036 |
| 1L-attn | 1 | attention | 48 | 5232 (5232) | 0.967 ± 0.090 | 0.992, 0.745, 1.000, 1.000, 0.997, 1.000, 1.000, 1.000 | 0.677 ± 0.050 | 0.525 ± 0.067 |
| 1L-attn-wide | 1 | attention | 146 | 10034 (10034) | 0.976 ± 0.041 | 0.898, 1.000, 1.000, 1.000, 1.000, 0.988, 0.923, 1.000 | 0.678 ± 0.024 | 0.506 ± 0.025 |
| 3L-attn | 3 | attention | 48 | 14832 (14832) | 0.982 ± 0.022 | 0.975, 0.978, 1.000, 0.980, 1.000, 0.936, 0.991, 1.000 | 0.686 ± 0.007 | 0.557 ± 0.125 |
| 2L-uniform | 2 | uniform | 48 | 10032 (7728) | 0.520 ± 0.026 | 0.505, 0.503, 0.542, 0.569, 0.536, 0.487, 0.511, 0.509 | 0.456 ± 0.011 | 0.830 ± 0.065 |
| bigram baseline | - | - | - | - | 0.250 ± 0.012 | - | - | - |

### Head ablation on the 2-layer model (copy accuracy with the head zeroed)

| Ablated | Copy acc (mean ± std) | Drop vs intact (mean ± std) |
|---|---:|---:|
| nothing (intact) | 0.976 ± 0.034 | - |
| L1H0 | 0.856 ± 0.235 | 0.121 ± 0.206 |
| L1H1 | 0.639 ± 0.263 | 0.337 ± 0.274 |
| L1H2 | 0.762 ± 0.312 | 0.214 ± 0.313 |
| L2H0 | 0.971 ± 0.039 | 0.005 ± 0.016 |
| L2H1 | 0.969 ± 0.036 | 0.007 ± 0.014 |
| L2H2 | 0.972 ± 0.035 | 0.004 ± 0.013 |
| all heads in layer 1 | 0.224 ± 0.039 | 0.753 ± 0.049 |
| all heads in layer 2 | 0.951 ± 0.034 | 0.025 ± 0.027 |
| all heads | 0.217 ± 0.038 | 0.759 ± 0.051 |

Head labels are per seed and arbitrary across seeds, so the per-seed view matters:

| Seed | Intact | Bigram | Sharpest head | Acc without it | Most critical head | Its drop | Largest drop of any other head |
|---:|---:|---:|---|---:|---|---:|---:|
| 0 | 0.897 | 0.256 | L1H0 | 0.330 | L1H0 | 0.567 | 0.111 |
| 1 | 0.994 | 0.242 | L1H2 | 0.252 | L1H2 | 0.742 | 0.100 |
| 2 | 0.978 | 0.230 | L1H2 | 0.284 | L1H2 | 0.694 | 0.027 |
| 3 | 1.000 | 0.263 | L1H2 | 0.927 | L1H1 | 0.514 | 0.073 |
| 4 | 0.995 | 0.244 | L1H1 | 0.728 | L1H1 | 0.267 | 0.034 |
| 5 | 0.989 | 0.248 | L1H1 | 0.220 | L1H1 | 0.769 | 0.039 |
| 6 | 0.989 | 0.245 | L1H1 | 0.619 | L1H1 | 0.370 | 0.286 |
| 7 | 0.969 | 0.270 | L1H1 | 0.370 | L1H1 | 0.598 | 0.086 |

### Attention signature of the most critical head

| Seed | Head | Mass on source | Argmax = source | Mass on previous token | Ambiguous queries | Source mass (ambiguous) | Distractor mass (ambiguous) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | L1H0 | 0.749 | 0.750 | 0.000 | 90/128 | 0.777 | 0.122 |
| 1 | L1H2 | 0.995 | 1.000 | 0.000 | 93/128 | 0.994 | 0.001 |
| 2 | L1H2 | 0.970 | 0.984 | 0.000 | 83/128 | 0.960 | 0.023 |
| 3 | L1H1 | 0.999 | 1.000 | 0.000 | 91/128 | 0.999 | 0.001 |
| 4 | L1H1 | 0.995 | 1.000 | 0.000 | 85/128 | 0.993 | 0.004 |
| 5 | L1H1 | 0.983 | 0.992 | 0.000 | 85/128 | 0.978 | 0.004 |
| 6 | L1H1 | 0.750 | 0.750 | 0.000 | 84/128 | 0.774 | 0.095 |
| 7 | L1H1 | 0.969 | 0.977 | 0.000 | 90/128 | 0.967 | 0.000 |

Largest previous-token mass among layer-1 heads (a previous-token head is the first
half of the classic two-layer induction circuit):

| Seed | Head | Mass on previous token |
|---:|---|---:|
| 0 | L1H2 | 0.248 |
| 1 | L1H1 | 0.013 |
| 2 | L1H1 | 0.157 |
| 3 | L1H2 | 0.249 |
| 4 | L1H0 | 0.249 |
| 5 | L1H2 | 0.330 |
| 6 | L1H2 | 0.224 |
| 7 | L1H0 | 0.218 |

Budget for every run: 300 steps, batch 16, learning rate 0.02; 128 held-out sequences per
seed. The full study (40 runs plus ablations and probes) took 693 s of wall time with 8 worker
processes on an 8-core laptop; run sequentially it is roughly 35 to 40 minutes, and about 14
minutes with `--seeds 0 1 2`.

### Reading the results

**Copying lives in layer 1, not layer 2.** In every seed the head whose ablation costs the
most is a layer-1 head. Zeroing all three layer-1 heads drops accuracy to 0.224 ± 0.039, at or
below the bigram baseline, while zeroing all three layer-2 heads costs only 0.025 ± 0.027, and
no single layer-2 head costs more than 0.04 in any seed. That is the opposite of the textbook
induction circuit, where the copying head sits in the second layer and reads a previous-token
head from the first.

**The critical head is positional, not content-matching.** Across seeds it puts 0.75 to 0.999
of its attention on the source position. The decisive measurement is on *ambiguous* queries
(about two thirds of them, because 4 symbols repeat within 5 positions), where the current
token also occurs elsewhere earlier. A content-based induction head would split its weight
across all positions that follow an earlier copy of the current token; this head puts
0.000 to 0.12 on those distractors and keeps 0.77 to 0.999 on the true source. Its
previous-token mass is 0.000 in every seed. Because every sequence has the same length, "the
token `length` positions back" is always the right answer, and learned absolute position
embeddings make that a one-head, one-layer lookup. In 6 of 8 seeds some layer-1 head does put
0.22 to 0.33 of its mass on the previous token, but no layer-2 head depends on it enough to
matter.

**Sharpness is a poor proxy for importance.** In 7 of 8 seeds the sharpest head (highest
mean peak attention weight) is also the most critical one. In seed 3 it is not: the sharpest
head is confidently attending somewhere other than the source (0.00 source mass in the
post-hoc table below) and removing it costs 0.073, while the head that actually copies is a
different one whose removal costs 0.514.

### Post-hoc analysis (exploratory, not pre-registered)

These diagnostics were added after the main run to explain two things it did not: why H1
fails in some seeds, and how a mixer with no queries or keys copies at all. They come from
`benchmarks/research_posthoc.py`, which retrains the same arms with the same seeds and budget.
Treat them as hypothesis-generating.

### Copy accuracy by target position (mean ± std over seeds)

| Target | 2L-attn | 2L-uniform | Multiset oracle |
|---:|---:|---:|---:|
| 1 | 1.000 ± 0.000 | 0.786 ± 0.111 | 0.502 ± 0.034 |
| 2 | 0.979 ± 0.034 | 0.615 ± 0.085 | 0.535 ± 0.034 |
| 3 | 0.993 ± 0.019 | 0.493 ± 0.085 | 0.548 ± 0.042 |
| 4 | 0.921 ± 0.178 | 0.400 ± 0.122 | 0.634 ± 0.019 |
| 5 | 0.989 ± 0.016 | 0.307 ± 0.050 | 1.000 ± 0.000 |

### Layer-1 attention mass on the source position, per copy target

| Seed | Head | t1 | t2 | t3 | t4 | t5 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | L1H0 | 1.00 | 0.99 | 1.00 | 0.00 | 1.00 |
| 0 | L1H1 | 0.33 | 0.05 | 0.00 | 0.21 | 0.00 |
| 0 | L1H2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | L1H0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | L1H1 | 0.99 | 0.87 | 0.01 | 0.00 | 0.00 |
| 1 | L1H2 | 0.70 | 1.00 | 1.00 | 0.97 | 1.00 |
| 2 | L1H0 | 0.00 | 0.00 | 0.03 | 0.11 | 0.36 |
| 2 | L1H1 | 0.00 | 0.00 | 0.74 | 0.00 | 0.00 |
| 2 | L1H2 | 1.00 | 1.00 | 0.87 | 1.00 | 1.00 |
| 3 | L1H0 | 0.00 | 0.89 | 0.99 | 1.00 | 0.65 |
| 3 | L1H1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 3 | L1H2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | L1H0 | 0.13 | 0.26 | 0.14 | 0.05 | 0.00 |
| 4 | L1H1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 |
| 4 | L1H2 | 1.00 | 1.00 | 0.96 | 0.65 | 1.00 |
| 5 | L1H0 | 0.42 | 0.00 | 0.03 | 0.03 | 0.00 |
| 5 | L1H1 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 |
| 5 | L1H2 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 |
| 6 | L1H0 | 1.00 | 0.75 | 1.00 | 0.08 | 0.00 |
| 6 | L1H1 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 6 | L1H2 | 0.00 | 0.00 | 0.00 | 0.10 | 0.27 |
| 7 | L1H0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | L1H1 | 0.97 | 1.00 | 1.00 | 0.92 | 1.00 |
| 7 | L1H2 | 0.20 | 0.96 | 0.00 | 0.88 | 0.99 |

**Why H1 fails in some seeds: the work is split or duplicated across layer-1 heads.** The
per-target table shows that copying is partitioned by position. In seed 0 one head covers
targets 1, 2, 3 and 5 but not target 4, which is why that seed's intact accuracy is lowest.
In seed 6 two heads cover overlapping subsets of targets (L1H0 misses targets 4 and 5, L1H1
misses target 3), so removing either one costs only part of the accuracy (0.370 and 0.286). In
seed 4 two heads largely duplicate each other: removing L1H2 costs 0.034 and removing L1H1
costs 0.267, but removing all of layer 1 drops accuracy to 0.172. A single-head ablation cannot see this kind of
backup; a joint ablation can.

**The uniform mixer does not use an order-free counting strategy.** My first explanation for
H3's 0.52 was that mean pooling gives the model the *counts* of each symbol, so it could
predict "whatever symbol is still left to copy". That strategy gets steadily better along the
copy (the last target is fully determined by counts, accuracy 1.0). The uniform model does the
opposite: it is best on the first target (0.786) and worst on the last (0.307). So the
counting explanation is refuted. A more plausible account, not tested here, is that the value
vectors are a nonlinear (layer-normed) function of token and position, so their average still
carries some "symbol s was at position i" information in superposition; as the pooled
window grows with each step of the copy, each item's share shrinks and decoding degrades.

## Verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1, a single head is causally necessary | **MIXED** | The sharpest head's ablation reaches within 0.15 of the bigram baseline in 5 of 8 seeds; both parts of H1 (that, plus every other head costing < 0.10) hold in only 4 of 8. In seed 3 the sharpest head is not the important one; in seeds 4 and 6 copying is shared across two heads. |
| H2, two layers are required | **NOT SUPPORTED** | 2L minus 1L at matched steps = +0.010; 2L minus 1L at matched parameters = +0.000. Both gaps are far below the pre-registered 0.10 and inside the seed-to-seed spread. 3 layers (0.982 ± 0.022) do not help either. |
| H3, content-based attention is required | **MIXED** | Mean pooling scores 0.520 ± 0.026: 0.456 below attention, but 0.27 above the bigram baseline, so it neither fails the task (within 0.10 of bigram) nor matches attention. |

**H1 reasoning.** The broad claim behind H1, that copying depends on a small, identifiable set
of heads, holds: in every seed it is one or two layer-1 heads, and removing layer 1 entirely
takes the model to the baseline. The specific claim, that one *sharpest* head is necessary and
the others are irrelevant, holds only in about half of the seeds, because training sometimes
spreads the lookup across two heads and because peak attention weight sometimes identifies
the wrong head. The pre-registered rule gives MIXED and that is the honest reading.

**H2 reasoning.** The prediction relied on the copy task needing an induction circuit. It
does not: the source position is always a fixed offset away, so one layer of position-based
attention suffices. This was the risk written down before the run, and it materialized. The
1-layer matched-steps arm has one weak seed (0.745) that inflates its spread, but its median
seed is 1.000. H2 being false here says nothing against the induction-head literature; it says
this task does not test it.

**H3 reasoning.** Attention clearly matters (removing the ability to choose where to look
costs almost half the accuracy, and every seed of the uniform model is below every seed of the
attention model). But the prediction that a uniform mixer would collapse to the baseline was
wrong: a content-independent average, followed by the feed-forward layers, still copies about
half of the symbols, mostly the early ones.

## Threats to validity

- **Tiny model, synthetic task.** About 10k parameters, 4 symbols, strings of length 5. The
  mechanism found here is specific to fixed-length sequences with absolute position embeddings;
  it is exactly the shortcut a real induction head does not have available, so none of these
  results should be read as statements about induction in language models.
- **The task admits a positional shortcut.** This is the main reason H2 failed, and it also
  limits what H1 can show: a study that wants to see induction needs variable-length or
  randomly offset repeats so that "attend `length` back" stops working.
- **Few seeds and small evaluation sets.** 8 seeds and 128 held-out sequences per seed (640
  copy targets). Accuracies are quantized in steps of about 0.0016 per seed, and a single bad
  seed (for example 1L-attn seed 1 at 0.745) moves a mean and a standard deviation noticeably.
  I report per-seed values next to every mean for that reason; no significance tests were run.
- **One training budget.** Every arm gets the default recipe (300 steps, learning rate 0.02).
  That recipe is the repository default, chosen for the 2-layer model before this study
  existed, which if anything favours the 2-layer arm; it still did not win. A different budget could change the 1-layer
  and uniform-mixer numbers.
- **Zero ablation.** Zeroing a head is the simplest intervention but pushes the downstream
  layers off-distribution; mean ablation or activation patching would give a cleaner estimate
  of each head's contribution, especially for the small layer-2 effects.
- **Head selection by sharpness.** H1 named "the sharpest head" as the candidate. That choice
  was made before running, and the results show it is an imperfect selector; the per-seed table
  reports the ablation-selected critical head alongside it so both readings are visible.
- **Pure-Python budget.** The engine is slow (about a minute per 2-layer run), which is what
  limited the study to 8 seeds, a single task size, and no hyperparameter sweep.
- **Post-hoc analysis is exploratory.** The per-target tables and the refuted counting
  explanation were produced after seeing the main results and are not covered by the
  pre-registration.

## What I would do next

1. **Make the task require induction.** Insert the repeated string at a random offset, or
   vary the string length per sequence, so a fixed positional lookup no longer works. Then
   rerun H1 and H2 unchanged. The prediction from the literature is that 1-layer models fail
   and a layer-2 head becomes critical; that would be a much sharper test than this one.
2. **Remove the shortcut directly.** Train without absolute position embeddings (or with
   relative ones) on the same task and check whether the model falls back to an induction
   circuit.
3. **Better interventions.** Replace zero ablation with mean ablation and with activation
   patching between pairs of sequences that differ in one symbol, to measure each head's
   contribution per target position rather than in aggregate.
4. **Test the superposition account of the uniform mixer.** Probe the pooled vector with a
   linear classifier for "symbol at position i" and check whether decodability falls with the
   width of the pooling window, as the per-target accuracy suggests.
5. **More seeds and a learning-rate sweep per arm**, so each architecture is compared at its
   own best budget rather than at the 2-layer model's.
