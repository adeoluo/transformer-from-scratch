"""A causal study of how the copy task is solved, and what architecture it needs.

This module holds the experiment code behind ``docs/RESEARCH.md``. It answers
three pre-registered questions about the small transformer in this package:

* **H1, causal necessity.** Zeroing the output of the single sharpest attention
  head should drop held-out copy accuracy to near the bigram baseline, while
  zeroing any other single head should cost little.
* **H2, depth.** A 1-layer model should copy worse than a 2-layer model at a
  matched training-step budget and at a matched parameter budget.
* **H3, attention.** Replacing softmax attention with a fixed causal
  mean-pooling mixer should fail the task at a matched budget.

The decision thresholds for each verdict are module constants fixed before any
result was produced, so the verdict functions below cannot be tuned after the
fact without the change showing up in version control. Everything is seeded,
offline, and uses only the standard library.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from transformer_from_scratch.data import (
    CopyTask,
    test_sequences,
    to_inputs_targets,
    train_sequences,
)
from transformer_from_scratch.evaluation import BigramBaseline, evaluate, evaluate_bigram
from transformer_from_scratch.inspect import attention_pattern, sharpest_head
from transformer_from_scratch.model import ModelConfig, Transformer
from transformer_from_scratch.training import TrainConfig, TrainResult, train

# Pre-registered decision thresholds (absolute copy-accuracy differences).
NEAR_BASELINE_MARGIN = 0.15  # H1: "near the bigram baseline" means within this of it
SMALL_COST = 0.10  # H1: "costs little" means a drop smaller than this
DEPTH_GAP = 0.10  # H2: a "clear" depth effect is a mean gap larger than this
NO_EFFECT = 0.05  # H2: a gap no larger than this counts as no effect
MIXER_MARGIN = 0.10  # H3: thresholds for "fails" and "as good as attention"

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT SUPPORTED"
MIXED = "MIXED"

HeadId = tuple[int, int]


@dataclass(frozen=True)
class StudyConfig:
    """Task, base architecture, and budget shared by every arm of the study."""

    n_symbols: int = 4
    length: int = 5
    d_model: int = 24
    n_head: int = 3
    d_ff: int = 48
    steps: int = 300
    batch_size: int = 16
    learning_rate: float = 0.02
    n_test: int = 128
    n_probe: int = 32
    include_three_layer: bool = True

    @property
    def task(self) -> CopyTask:
        return CopyTask(n_symbols=self.n_symbols, length=self.length)

    def base_model(self) -> ModelConfig:
        task = self.task
        return ModelConfig(
            vocab_size=task.vocab_size,
            context=task.seq_len,
            d_model=self.d_model,
            n_head=self.n_head,
            n_layer=2,
            d_ff=self.d_ff,
        )


@dataclass(frozen=True)
class Variant:
    """One architecture arm of the study."""

    name: str
    description: str
    model: ModelConfig


def parameter_count(cfg: ModelConfig, *, effective: bool = False) -> int:
    """Return the number of scalar parameters a model with ``cfg`` has.

    With ``effective=True`` the query and key matrices of uniform-mixer heads are
    excluded, because they never influence the output and receive no gradient.
    """

    d = cfg.d_model
    embeddings = cfg.vocab_size * d + cfg.context * d
    final_norm = 2 * d
    qk_per_head = 2 * d * cfg.d_head
    v_per_head = d * cfg.d_head
    per_head = v_per_head if effective and cfg.mixer == "uniform" else v_per_head + qk_per_head
    attention = cfg.n_head * per_head + d * d + d
    feed_forward = d * cfg.d_ff + cfg.d_ff + cfg.d_ff * d + d
    norms = 4 * d
    return embeddings + final_norm + cfg.n_layer * (attention + feed_forward + norms)


def matched_d_ff(cfg: ModelConfig, *, n_layer: int = 1) -> int:
    """Return the ``d_ff`` that gives an ``n_layer`` model the parameter count of ``cfg``."""

    target = parameter_count(cfg)
    without_ff = parameter_count(replace(cfg, n_layer=n_layer, d_ff=0))
    per_unit = n_layer * (2 * cfg.d_model + 1)
    return max(1, round((target - without_ff) / per_unit))


def variants(cfg: StudyConfig) -> list[Variant]:
    """Return the architecture arms, all trained with the same step budget."""

    base = cfg.base_model()
    arms = [
        Variant("2L-attn", "2 layers, softmax attention (reference)", base),
        Variant("1L-attn", "1 layer, same widths (matched steps)", replace(base, n_layer=1)),
        Variant(
            "1L-attn-wide",
            "1 layer, wider FFN (matched parameters)",
            replace(base, n_layer=1, d_ff=matched_d_ff(base)),
        ),
    ]
    if cfg.include_three_layer:
        arms.append(Variant("3L-attn", "3 layers, same widths", replace(base, n_layer=3)))
    arms.append(
        Variant("2L-uniform", "2 layers, causal mean-pooling mixer", replace(base, mixer="uniform"))
    )
    return arms


@contextmanager
def ablate_heads(model: Transformer, heads: Iterable[HeadId]) -> Iterator[Transformer]:
    """Temporarily zero the output of the given ``(layer, head)`` pairs.

    The head's contribution to the residual stream is removed before the output
    projection, so the projection bias is all that remains of the head. Flags are
    restored on exit even if evaluation raises.
    """

    targets = [model.blocks[layer].attn.heads[head] for layer, head in heads]
    previous = [h.ablated for h in targets]
    for h in targets:
        h.ablated = True
    try:
        yield model
    finally:
        for h, prev in zip(targets, previous, strict=True):
            h.ablated = prev


def copy_accuracy(model: Transformer, task: CopyTask, sequences: list[list[int]]) -> float:
    return evaluate(model, task, sequences).copy_accuracy


@dataclass
class HeadAblation:
    """Copy accuracy of a model with nothing, one head, or a whole layer zeroed."""

    intact: float
    per_head: list[list[float]]
    per_layer: list[float]
    all_heads: float

    def drop(self, layer: int, head: int) -> float:
        return self.intact - self.per_head[layer][head]

    def critical_head(self) -> HeadId:
        """The head whose removal costs the most copy accuracy (first on ties)."""

        best: HeadId = (0, 0)
        for layer, row in enumerate(self.per_head):
            for head in range(len(row)):
                if self.drop(layer, head) > self.drop(*best):
                    best = (layer, head)
        return best

    def max_other_drop(self, exclude: HeadId) -> float:
        drops = [
            self.drop(layer, head)
            for layer, row in enumerate(self.per_head)
            for head in range(len(row))
            if (layer, head) != exclude
        ]
        return max(drops) if drops else 0.0


def head_ablation(model: Transformer, task: CopyTask, sequences: list[list[int]]) -> HeadAblation:
    """Measure copy accuracy with every single head, and every layer, ablated."""

    n_layer, n_head = model.cfg.n_layer, model.cfg.n_head
    per_head: list[list[float]] = []
    for layer in range(n_layer):
        row = []
        for head in range(n_head):
            with ablate_heads(model, [(layer, head)]):
                row.append(copy_accuracy(model, task, sequences))
        per_head.append(row)
    per_layer = []
    for layer in range(n_layer):
        with ablate_heads(model, [(layer, h) for h in range(n_head)]):
            per_layer.append(copy_accuracy(model, task, sequences))
    every = [(layer, h) for layer in range(n_layer) for h in range(n_head)]
    with ablate_heads(model, every):
        all_heads = copy_accuracy(model, task, sequences)
    return HeadAblation(
        intact=copy_accuracy(model, task, sequences),
        per_head=per_head,
        per_layer=per_layer,
        all_heads=all_heads,
    )


@dataclass(frozen=True)
class InductionScore:
    """Where one head puts its attention on copy-region queries.

    For a query at position ``p`` in the second copy, the *source* is the token
    right after the earlier occurrence of the current token in the aligned first
    copy, which is exactly the token to predict. *Distractors* are positions that
    follow some other earlier occurrence of the same token: a purely content-based
    induction head cannot tell them apart from the source, while a head that also
    uses position can. ``previous_token_mass`` is the weight on ``p - 1``.
    """

    layer: int
    head: int
    source_mass: float
    source_argmax_rate: float
    previous_token_mass: float
    ambiguous_source_mass: float
    ambiguous_distractor_mass: float
    n_queries: int
    n_ambiguous: int


def induction_score(
    model: Transformer,
    task: CopyTask,
    sequences: list[list[int]],
    *,
    layer: int,
    head: int,
) -> InductionScore:
    """Measure the induction signature of one head over copy-region queries."""

    length = task.length
    source = argmax_hits = previous = amb_source = amb_distractor = 0.0
    n_queries = n_ambiguous = 0
    for seq in sequences:
        inputs = seq[:-1]
        pattern = attention_pattern(model, inputs, layer=layer, head=head)
        for p in range(length + 1, len(inputs)):
            row = pattern[p][: p + 1]
            src = p - length
            n_queries += 1
            source += row[src]
            previous += row[p - 1]
            if max(range(len(row)), key=lambda j: row[j]) == src:
                argmax_hits += 1
            distractors = {j + 1 for j in range(p) if inputs[j] == inputs[p] and j + 1 != src}
            if distractors:
                n_ambiguous += 1
                amb_source += row[src]
                amb_distractor += sum(row[j] for j in distractors)
    return InductionScore(
        layer=layer,
        head=head,
        source_mass=source / n_queries if n_queries else 0.0,
        source_argmax_rate=argmax_hits / n_queries if n_queries else 0.0,
        previous_token_mass=previous / n_queries if n_queries else 0.0,
        ambiguous_source_mass=amb_source / n_ambiguous if n_ambiguous else 0.0,
        ambiguous_distractor_mass=amb_distractor / n_ambiguous if n_ambiguous else 0.0,
        n_queries=n_queries,
        n_ambiguous=n_ambiguous,
    )


@dataclass
class RunResult:
    """Everything measured for one (variant, seed) training run."""

    variant: str
    seed: int
    n_layer: int
    mixer: str
    d_ff: int
    parameters: int
    effective_parameters: int
    copy_accuracy: float
    overall_accuracy: float
    perplexity: float
    bigram_copy_accuracy: float
    final_loss: float
    seconds: float
    ablation: HeadAblation | None = None
    sharpest: HeadId | None = None
    probes: list[InductionScore] = field(default_factory=list)


def train_arm(cfg: StudyConfig, variant: Variant, seed: int) -> TrainResult:
    """Train one arm with one seed using the study's shared budget."""

    train_cfg = TrainConfig(
        steps=cfg.steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        seed=seed,
    )
    return train(cfg.task, variant.model, train_cfg, log_every=max(cfg.steps, 1))


def run_variant(cfg: StudyConfig, variant: Variant, seed: int, *, analyze: bool) -> RunResult:
    """Train one arm with one seed and evaluate it on that seed's held-out split.

    With ``analyze`` the trained model is also ablated head by head and every
    head's induction signature is measured.
    """

    task = cfg.task
    start = time.perf_counter()
    result = train_arm(cfg, variant, seed)
    seconds = time.perf_counter() - start
    model = result.model
    held_out = test_sequences(task, count=cfg.n_test, seed=seed)
    report = evaluate(model, task, held_out)
    bigram = BigramBaseline.fit(task.vocab_size, train_sequences(task, seed=seed))
    run = RunResult(
        variant=variant.name,
        seed=seed,
        n_layer=variant.model.n_layer,
        mixer=variant.model.mixer,
        d_ff=variant.model.d_ff,
        parameters=parameter_count(variant.model),
        effective_parameters=parameter_count(variant.model, effective=True),
        copy_accuracy=report.copy_accuracy,
        overall_accuracy=report.overall_accuracy,
        perplexity=report.perplexity,
        bigram_copy_accuracy=evaluate_bigram(bigram, task, held_out).copy_accuracy,
        final_loss=result.losses[-1][1],
        seconds=seconds,
    )
    if analyze:
        probe = [seq[:-1] for seq in held_out[: cfg.n_probe]]
        run.ablation = head_ablation(model, task, held_out)
        sharp = sharpest_head(model, probe, start=task.length)
        run.sharpest = (sharp.layer, sharp.head)
        probe_full = held_out[: cfg.n_probe]
        run.probes = [
            induction_score(model, task, probe_full, layer=layer, head=head)
            for layer in range(variant.model.n_layer)
            for head in range(variant.model.n_head)
        ]
    return run


def _job(args: tuple[StudyConfig, Variant, int, bool]) -> RunResult:
    cfg, variant, seed, analyze = args
    return run_variant(cfg, variant, seed, analyze=analyze)


def run_study(cfg: StudyConfig, seeds: Sequence[int], *, workers: int = 1) -> list[RunResult]:
    """Train every arm for every seed; the 2-layer attention runs are also analyzed.

    Runs are independent and individually seeded, so the results are identical
    whether they execute sequentially or in a process pool.
    """

    jobs = [(cfg, v, seed, v.name == "2L-attn") for v in variants(cfg) for seed in seeds]
    if workers <= 1:
        return [_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_job, jobs))


# ---------------------------------------------------------------------------
# Post-hoc (exploratory) diagnostics, added after the pre-registered runs
# ---------------------------------------------------------------------------


def copy_accuracy_by_position(
    model: Transformer, task: CopyTask, sequences: list[list[int]]
) -> list[float]:
    """Return held-out accuracy at each copy-region target, first to last."""

    region = list(task.copy_region())
    correct = [0] * len(region)
    for seq in sequences:
        inputs, targets = to_inputs_targets(seq)
        logits = model.logits(inputs).data
        for k, pos in enumerate(region):
            row = logits[pos]
            if max(range(len(row)), key=lambda j: row[j]) == targets[pos]:
                correct[k] += 1
    return [c / len(sequences) if sequences else 0.0 for c in correct]


def multiset_oracle_by_position(task: CopyTask, sequences: list[list[int]]) -> list[float]:
    """Expected accuracy of a predictor that knows only which symbols remain to be copied.

    At the ``k``-th copy target the remaining multiset is the first copy minus the
    ``k`` symbols already re-emitted. Given only that multiset every ordering is
    equally likely, so the best guess is a most frequent remaining symbol, with
    ties broken uniformly at random. This is what an order-blind, count-based
    strategy can reach; it is a reference point, not an upper bound on any model.
    """

    out = [0.0] * task.length
    for seq in sequences:
        body = seq[: task.length]
        for k in range(task.length):
            remaining = body[k:]
            counts = {s: remaining.count(s) for s in set(remaining)}
            top = max(counts.values())
            winners = [s for s, c in counts.items() if c == top]
            if body[k] in winners:
                out[k] += 1.0 / len(winners)
    return [v / len(sequences) if sequences else 0.0 for v in out]


def source_mass_by_position(
    model: Transformer,
    task: CopyTask,
    sequences: list[list[int]],
    *,
    layer: int,
    head: int,
) -> list[float]:
    """Mean attention weight on the source position for each copy-region query."""

    length = task.length
    region = list(task.copy_region())
    totals = [0.0] * len(region)
    for seq in sequences:
        inputs = seq[:-1]
        pattern = attention_pattern(model, inputs, layer=layer, head=head)
        for k, p in enumerate(region):
            totals[k] += pattern[p][p - length]
    return [t / len(sequences) if sequences else 0.0 for t in totals]


# ---------------------------------------------------------------------------
# Aggregation and pre-registered verdicts
# ---------------------------------------------------------------------------


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Return the mean and sample standard deviation (0 for a single value)."""

    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _by_variant(runs: Sequence[RunResult]) -> dict[str, list[RunResult]]:
    grouped: dict[str, list[RunResult]] = {}
    for run in runs:
        grouped.setdefault(run.variant, []).append(run)
    return grouped


def _analyzed(runs: Sequence[RunResult]) -> list[RunResult]:
    return [r for r in runs if r.ablation is not None and r.sharpest is not None]


def verdict_h1(runs: Sequence[RunResult]) -> tuple[str, str]:
    """H1: the sharpest head is necessary and no other single head matters much.

    A seed only counts if its intact model is itself clearly above the bigram
    baseline; otherwise "drops to near the baseline" is true by default.
    """

    analyzed = [
        r
        for r in _analyzed(runs)
        if r.ablation is not None
        and r.ablation.intact > r.bigram_copy_accuracy + NEAR_BASELINE_MARGIN
    ]
    if not analyzed:
        return MIXED, "no seed learned the task well enough to test necessity"
    necessary = 0
    both = 0
    for run in analyzed:
        assert run.ablation is not None and run.sharpest is not None
        ablated = run.ablation.per_head[run.sharpest[0]][run.sharpest[1]]
        is_necessary = ablated <= run.bigram_copy_accuracy + NEAR_BASELINE_MARGIN
        others_small = run.ablation.max_other_drop(run.sharpest) < SMALL_COST
        necessary += is_necessary
        both += is_necessary and others_small
    n = len(analyzed)
    reason = (
        f"sharpest-head ablation within {NEAR_BASELINE_MARGIN} of bigram in {necessary}/{n} "
        f"informative seeds; that and every other head costing < {SMALL_COST} in {both}/{n}"
    )
    if both == n:
        return SUPPORTED, reason
    if necessary == 0:
        return NOT_SUPPORTED, reason
    return MIXED, reason


def _gap(reference: list[float], other: list[float]) -> tuple[float, bool]:
    ref_mean, ref_std = mean_std(reference)
    other_mean, other_std = mean_std(other)
    gap = ref_mean - other_mean
    return gap, gap > DEPTH_GAP and gap > max(ref_std, other_std)


def verdict_h2(runs: Sequence[RunResult]) -> tuple[str, str]:
    """H2: 2 layers beat 1 layer at matched steps and at matched parameters."""

    grouped = _by_variant(runs)
    ref = [r.copy_accuracy for r in grouped.get("2L-attn", [])]
    narrow = [r.copy_accuracy for r in grouped.get("1L-attn", [])]
    wide = [r.copy_accuracy for r in grouped.get("1L-attn-wide", [])]
    if not (ref and narrow and wide):
        return MIXED, "missing arms"
    gap_steps, clear_steps = _gap(ref, narrow)
    gap_params, clear_params = _gap(ref, wide)
    reason = (
        f"2L minus 1L (matched steps) = {gap_steps:+.3f}; "
        f"2L minus 1L-wide (matched parameters) = {gap_params:+.3f}"
    )
    if clear_steps and clear_params:
        return SUPPORTED, reason
    if gap_steps <= NO_EFFECT and gap_params <= NO_EFFECT:
        return NOT_SUPPORTED, reason
    return MIXED, reason


def verdict_h3(runs: Sequence[RunResult]) -> tuple[str, str]:
    """H3: a uniform causal mean-pooling mixer fails the copy task."""

    grouped = _by_variant(runs)
    ref = [r.copy_accuracy for r in grouped.get("2L-attn", [])]
    uni = [r.copy_accuracy for r in grouped.get("2L-uniform", [])]
    if not (ref and uni):
        return MIXED, "missing arms"
    bigram = mean_std([r.bigram_copy_accuracy for r in grouped["2L-uniform"]])[0]
    uni_mean = mean_std(uni)[0]
    gap = mean_std(ref)[0] - uni_mean
    reason = f"uniform mixer {uni_mean:.3f} vs attention gap {gap:+.3f}; bigram {bigram:.3f}"
    if uni_mean <= bigram + MIXER_MARGIN:
        return SUPPORTED, reason
    if gap <= MIXER_MARGIN:
        return NOT_SUPPORTED, reason
    return MIXED, reason


def _fmt(values: Sequence[float]) -> str:
    mean, std = mean_std(values)
    return f"{mean:.3f} ± {std:.3f}"


def format_report(runs: Sequence[RunResult]) -> str:
    """Render all result tables and verdicts as Markdown."""

    grouped = _by_variant(runs)
    seeds = sorted({r.seed for r in runs})
    lines = [f"Seeds: {', '.join(str(s) for s in seeds)} (n = {len(seeds)})", ""]

    lines += [
        "### Architecture comparison (matched training steps)",
        "",
        "| Arm | Layers | Mixer | d_ff | Params (effective) | Copy acc (mean ± std) "
        "| Per seed | Overall acc | Final train loss |",
        "|---|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for name, group in grouped.items():
        first = group[0]
        copy = [r.copy_accuracy for r in group]
        lines.append(
            f"| {name} | {first.n_layer} | {first.mixer} | {first.d_ff} | "
            f"{first.parameters} ({first.effective_parameters}) | {_fmt(copy)} | "
            f"{', '.join(f'{c:.3f}' for c in copy)} | "
            f"{_fmt([r.overall_accuracy for r in group])} | "
            f"{_fmt([r.final_loss for r in group])} |"
        )
    lines.append(
        f"| bigram baseline | - | - | - | - | {_fmt([r.bigram_copy_accuracy for r in runs])} "
        "| - | - | - |"
    )

    analyzed = _analyzed(runs)
    if analyzed:
        abl = [r.ablation for r in analyzed if r.ablation is not None]
        n_layer = len(abl[0].per_head)
        n_head = len(abl[0].per_head[0])
        lines += [
            "",
            "### Head ablation on the 2-layer model (copy accuracy with the head zeroed)",
            "",
            "| Ablated | Copy acc (mean ± std) | Drop vs intact (mean ± std) |",
            "|---|---:|---:|",
            f"| nothing (intact) | {_fmt([a.intact for a in abl])} | - |",
        ]
        for layer in range(n_layer):
            for head in range(n_head):
                lines.append(
                    f"| L{layer + 1}H{head} | {_fmt([a.per_head[layer][head] for a in abl])} | "
                    f"{_fmt([a.drop(layer, head) for a in abl])} |"
                )
        for layer in range(n_layer):
            lines.append(
                f"| all heads in layer {layer + 1} | {_fmt([a.per_layer[layer] for a in abl])} | "
                f"{_fmt([a.intact - a.per_layer[layer] for a in abl])} |"
            )
        lines.append(
            f"| all heads | {_fmt([a.all_heads for a in abl])} | "
            f"{_fmt([a.intact - a.all_heads for a in abl])} |"
        )
        lines += [
            "",
            "Head labels are per seed and arbitrary across seeds, so the per-seed view matters:",
            "",
            "| Seed | Intact | Bigram | Sharpest head | Acc without it | Most critical head "
            "| Its drop | Largest drop of any other head |",
            "|---:|---:|---:|---|---:|---|---:|---:|",
        ]
        for run in analyzed:
            assert run.ablation is not None and run.sharpest is not None
            a, (sl, sh) = run.ablation, run.sharpest
            cl, ch = a.critical_head()
            lines.append(
                f"| {run.seed} | {a.intact:.3f} | {run.bigram_copy_accuracy:.3f} | "
                f"L{sl + 1}H{sh} | {a.per_head[sl][sh]:.3f} | L{cl + 1}H{ch} | "
                f"{a.drop(cl, ch):.3f} | {a.max_other_drop((cl, ch)):.3f} |"
            )
        lines += [
            "",
            "### Attention signature of the most critical head",
            "",
            "| Seed | Head | Mass on source | Argmax = source | Mass on previous token "
            "| Ambiguous queries | Source mass (ambiguous) | Distractor mass (ambiguous) |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for run in analyzed:
            assert run.ablation is not None
            cl, ch = run.ablation.critical_head()
            score = next(p for p in run.probes if (p.layer, p.head) == (cl, ch))
            lines.append(
                f"| {run.seed} | L{cl + 1}H{ch} | {score.source_mass:.3f} | "
                f"{score.source_argmax_rate:.3f} | {score.previous_token_mass:.3f} | "
                f"{score.n_ambiguous}/{score.n_queries} | {score.ambiguous_source_mass:.3f} | "
                f"{score.ambiguous_distractor_mass:.3f} |"
            )
        lines += [
            "",
            "Largest previous-token mass among layer-1 heads (a previous-token head is the first",
            "half of the classic two-layer induction circuit):",
            "",
            "| Seed | Head | Mass on previous token |",
            "|---:|---|---:|",
        ]
        for run in analyzed:
            first_layer = [p for p in run.probes if p.layer == 0]
            best = max(first_layer, key=lambda p: p.previous_token_mass)
            lines.append(f"| {run.seed} | L1H{best.head} | {best.previous_token_mass:.3f} |")

    lines += ["", "### Verdicts (thresholds fixed in code before the runs)", ""]
    for label, (verdict, reason) in (
        ("H1", verdict_h1(runs)),
        ("H2", verdict_h2(runs)),
        ("H3", verdict_h3(runs)),
    ):
        lines.append(f"- **{label}: {verdict}.** {reason}.")
    return "\n".join(lines) + "\n"


def to_json(runs: Sequence[RunResult]) -> list[dict[str, Any]]:
    """Return the raw per-run results as JSON-serializable dicts."""

    return [asdict(run) for run in runs]
