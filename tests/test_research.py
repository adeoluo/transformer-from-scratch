"""Tests for the research study: ablation mechanics, the uniform mixer, verdicts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from transformer_from_scratch import tensor as T
from transformer_from_scratch.data import CopyTask, to_inputs_targets
from transformer_from_scratch.data import test_sequences as held_out_sequences
from transformer_from_scratch.evaluation import evaluate
from transformer_from_scratch.inspect import attention_pattern
from transformer_from_scratch.model import ModelConfig, Transformer
from transformer_from_scratch.research import (
    MIXED,
    NOT_SUPPORTED,
    SUPPORTED,
    HeadAblation,
    RunResult,
    StudyConfig,
    ablate_heads,
    copy_accuracy_by_position,
    format_report,
    head_ablation,
    induction_score,
    matched_d_ff,
    mean_std,
    multiset_oracle_by_position,
    parameter_count,
    run_study,
    source_mass_by_position,
    to_json,
    variants,
    verdict_h1,
    verdict_h2,
    verdict_h3,
)
from transformer_from_scratch.tensor import Tensor
from transformer_from_scratch.training import TrainResult

TINY = StudyConfig(
    n_symbols=3,
    length=3,
    d_model=8,
    n_head=2,
    d_ff=8,
    steps=2,
    batch_size=2,
    n_test=6,
    n_probe=2,
    include_three_layer=False,
)


def _row_input(cfg: ModelConfig, rows: int) -> Tensor:
    return Tensor([[0.1 * (i + 1) * (j - 2) for j in range(cfg.d_model)] for i in range(rows)])


def test_ablated_head_outputs_zeros_and_flag_is_restored(small_cfg: ModelConfig) -> None:
    model = Transformer(small_cfg, seed=0)
    head = model.blocks[0].attn.heads[1]
    x = _row_input(small_cfg, 4)
    with ablate_heads(model, [(0, 1)]):
        out = head.forward(x)
        assert out.shape == (4, small_cfg.d_head)
        assert all(v == 0.0 for row in out.data for v in row)
    assert head.ablated is False
    assert any(v != 0.0 for row in head.forward(x).data for v in row)


def test_ablation_changes_logits_and_restores_on_error(small_cfg: ModelConfig) -> None:
    model = Transformer(small_cfg, seed=0)
    ids = [1, 2, 3, 0, 1]
    intact = model.logits(ids).data.tolist()
    with ablate_heads(model, [(1, 0)]):
        assert model.logits(ids).data.tolist() != intact
    with pytest.raises(RuntimeError), ablate_heads(model, [(1, 0)]):
        raise RuntimeError("boom")
    assert model.logits(ids).data.tolist() == intact


def test_ablating_every_head_removes_cross_position_information(
    small_cfg: ModelConfig,
) -> None:
    model = Transformer(small_cfg, seed=0)
    a, b = [1, 2, 3], [4, 2, 3]
    assert model.logits(a).data[2].tolist() != model.logits(b).data[2].tolist()
    every = [(layer, h) for layer in range(small_cfg.n_layer) for h in range(small_cfg.n_head)]
    with ablate_heads(model, every):
        assert model.logits(a).data[2].tolist() == model.logits(b).data[2].tolist()


def test_uniform_head_is_causal_mean_of_values(small_cfg: ModelConfig) -> None:
    from dataclasses import replace

    cfg = replace(small_cfg, mixer="uniform")
    model = Transformer(cfg, seed=0)
    head = model.blocks[0].attn.heads[0]
    x = _row_input(cfg, 3)
    v = T.matmul(x, head.w_v).data
    out = head.forward(x).data
    for i in range(3):
        for d in range(cfg.d_head):
            expected = sum(v[j][d] for j in range(i + 1)) / (i + 1)
            assert out[i][d] == pytest.approx(expected)
    pattern = attention_pattern(model, [1, 2, 3], layer=0, head=0)
    assert pattern[2] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert pattern[0][1] == 0.0


def test_uniform_mixer_queries_and_keys_get_no_gradient(small_cfg: ModelConfig) -> None:
    from dataclasses import replace

    model = Transformer(replace(small_cfg, mixer="uniform"), seed=0)
    inputs, targets = to_inputs_targets([1, 2, 3, 0, 1, 2, 3])
    model.loss(inputs, targets).backward()
    head = model.blocks[0].attn.heads[0]
    assert all(g == 0.0 for row in head.w_q.grad + head.w_k.grad for g in row)
    assert any(g != 0.0 for row in head.w_v.grad for g in row)


def test_default_model_is_unchanged_by_the_mixer_option(small_cfg: ModelConfig) -> None:
    from dataclasses import replace

    attn = Transformer(small_cfg, seed=3)
    uniform = Transformer(replace(small_cfg, mixer="uniform"), seed=3)
    for p, q in zip(attn.parameters(), uniform.parameters(), strict=True):
        assert p.data.tolist() == q.data.tolist()


def test_unknown_mixer_is_rejected() -> None:
    with pytest.raises(ValueError, match="mixer"):
        ModelConfig(vocab_size=5, context=8, mixer="bogus")


def test_parameter_count_matches_the_built_model() -> None:
    cfg = StudyConfig(d_model=8, n_head=2, d_ff=8, include_three_layer=True)
    for variant in variants(cfg):
        model = Transformer(variant.model, seed=0)
        actual = sum(p.rows * p.cols for p in model.parameters())
        assert parameter_count(variant.model) == actual
        effective = parameter_count(variant.model, effective=True)
        if variant.model.mixer == "uniform":
            assert effective < actual
        else:
            assert effective == actual
    names = [v.name for v in variants(cfg)]
    assert names == ["2L-attn", "1L-attn", "1L-attn-wide", "3L-attn", "2L-uniform"]


def test_matched_d_ff_matches_parameter_budget() -> None:
    base = StudyConfig().base_model()
    from dataclasses import replace

    wide = replace(base, n_layer=1, d_ff=matched_d_ff(base))
    assert abs(parameter_count(wide) - parameter_count(base)) <= 2 * base.d_model + 1
    assert wide.d_ff == 146


def test_head_ablation_and_induction_probe_on_trained_model(
    small_task: CopyTask, trained: TrainResult
) -> None:
    model = trained.model
    held_out = held_out_sequences(small_task, count=12, seed=0)
    result = head_ablation(model, small_task, held_out)
    assert len(result.per_head) == model.cfg.n_layer
    assert len(result.per_head[0]) == model.cfg.n_head
    assert len(result.per_layer) == model.cfg.n_layer
    assert all(0.0 <= v <= 1.0 for row in result.per_head for v in row)
    layer, head = result.critical_head()
    assert result.drop(layer, head) >= result.max_other_drop((layer, head))
    assert all(h.ablated is False for b in model.blocks for h in b.attn.heads)

    score = induction_score(model, small_task, held_out, layer=layer, head=head)
    assert score.n_queries == len(held_out) * (small_task.length - 1)
    for value in (
        score.source_mass,
        score.source_argmax_rate,
        score.previous_token_mass,
        score.ambiguous_source_mass,
    ):
        assert 0.0 <= value <= 1.0
    assert score.ambiguous_source_mass + score.ambiguous_distractor_mass <= 1.0 + 1e-9


def test_multiset_oracle_by_position() -> None:
    task = CopyTask(n_symbols=3, length=3)
    assert multiset_oracle_by_position(task, [[1, 1, 2, 0, 1, 1, 2]]) == [1.0, 0.5, 1.0]
    assert multiset_oracle_by_position(task, []) == [0.0, 0.0, 0.0]


def test_per_position_diagnostics_on_trained_model(
    small_task: CopyTask, trained: TrainResult
) -> None:
    model = trained.model
    held_out = held_out_sequences(small_task, count=10, seed=0)
    per_pos = copy_accuracy_by_position(model, small_task, held_out)
    assert len(per_pos) == small_task.length
    overall = evaluate(model, small_task, held_out).copy_accuracy
    assert sum(per_pos) / len(per_pos) == pytest.approx(overall)
    masses = source_mass_by_position(model, small_task, held_out, layer=0, head=0)
    assert len(masses) == small_task.length
    assert all(0.0 <= m <= 1.0 for m in masses)
    assert copy_accuracy_by_position(model, small_task, []) == [0.0] * small_task.length
    assert (
        source_mass_by_position(model, small_task, [], layer=0, head=0) == [0.0] * small_task.length
    )


def _comparable(runs: list[RunResult]) -> list[dict[str, object]]:
    rows = to_json(runs)
    for row in rows:
        row.pop("seconds")
    return rows


def test_run_study_is_deterministic_and_worker_independent() -> None:
    first = run_study(TINY, [0, 1], workers=1)
    second = run_study(TINY, [0, 1], workers=1)
    pooled = run_study(TINY, [0, 1], workers=2)
    assert _comparable(first) == _comparable(second) == _comparable(pooled)
    analyzed = [r for r in first if r.ablation is not None]
    assert {r.variant for r in analyzed} == {"2L-attn"}
    assert all(r.sharpest is not None and len(r.probes) == 4 for r in analyzed)
    json.dumps(to_json(first))

    report = format_report(first)
    for heading in ("Architecture comparison", "Head ablation", "Verdicts", "**H1", "**H3"):
        assert heading in report
    assert "\u2014" not in report and "\u2013" not in report


def test_mean_std() -> None:
    assert mean_std([]) == (0.0, 0.0)
    assert mean_std([0.5]) == (0.5, 0.0)
    mean, std = mean_std([0.0, 1.0])
    assert mean == 0.5
    assert std == pytest.approx(0.7071, abs=1e-4)


def _run(
    variant: str,
    seed: int,
    acc: float,
    *,
    bigram: float = 0.25,
    ablation: HeadAblation | None = None,
) -> RunResult:
    return RunResult(
        variant=variant,
        seed=seed,
        n_layer=2,
        mixer="attention",
        d_ff=48,
        parameters=1,
        effective_parameters=1,
        copy_accuracy=acc,
        overall_accuracy=acc,
        perplexity=1.0,
        bigram_copy_accuracy=bigram,
        final_loss=1.0,
        seconds=0.0,
        ablation=ablation,
        sharpest=(1, 0) if ablation is not None else None,
    )


def _ablation(sharp_acc: float, other_acc: float, intact: float = 0.9) -> HeadAblation:
    return HeadAblation(
        intact=intact,
        per_head=[[other_acc, other_acc], [sharp_acc, other_acc]],
        per_layer=[other_acc, sharp_acc],
        all_heads=0.2,
    )


def test_verdict_h1() -> None:
    supported = [_run("2L-attn", s, 0.9, ablation=_ablation(0.3, 0.88)) for s in range(3)]
    assert verdict_h1(supported)[0] == SUPPORTED
    not_necessary = [_run("2L-attn", s, 0.9, ablation=_ablation(0.85, 0.88)) for s in range(3)]
    assert verdict_h1(not_necessary)[0] == NOT_SUPPORTED
    others_matter = [_run("2L-attn", s, 0.9, ablation=_ablation(0.3, 0.5)) for s in range(3)]
    assert verdict_h1(others_matter)[0] == MIXED
    untrained = [_run("2L-attn", 0, 0.3, ablation=_ablation(0.3, 0.3, intact=0.3))]
    assert verdict_h1(untrained)[0] == MIXED
    assert verdict_h1([])[0] == MIXED


def test_verdict_h2() -> None:
    def arms(two: list[float], one: list[float], wide: list[float]) -> list[RunResult]:
        return (
            [_run("2L-attn", i, a) for i, a in enumerate(two)]
            + [_run("1L-attn", i, a) for i, a in enumerate(one)]
            + [_run("1L-attn-wide", i, a) for i, a in enumerate(wide)]
        )

    assert verdict_h2(arms([0.9, 0.92], [0.5, 0.52], [0.6, 0.62]))[0] == SUPPORTED
    assert verdict_h2(arms([0.9, 0.92], [0.9, 0.95], [0.91, 0.93]))[0] == NOT_SUPPORTED
    assert verdict_h2(arms([0.9, 0.92], [0.5, 0.52], [0.9, 0.92]))[0] == MIXED
    assert verdict_h2(arms([0.9, 0.92], [], []))[0] == MIXED


def test_verdict_h3() -> None:
    ref = [_run("2L-attn", i, 0.9) for i in range(2)]
    assert verdict_h3([*ref, *(_run("2L-uniform", i, 0.3) for i in range(2))])[0] == SUPPORTED
    assert verdict_h3([*ref, *(_run("2L-uniform", i, 0.85) for i in range(2))])[0] == (
        NOT_SUPPORTED
    )
    assert verdict_h3([*ref, *(_run("2L-uniform", i, 0.6) for i in range(2))])[0] == MIXED
    assert verdict_h3(ref)[0] == MIXED


@pytest.mark.slow
def test_research_runner_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "research_study.py"
    spec = importlib.util.spec_from_file_location("research_study", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(["--seeds", "0", "--steps", "30", "--n-test", "16", "--no-three-layer"])
    out = capsys.readouterr().out
    assert "2L-uniform" in out and "Verdicts" in out
