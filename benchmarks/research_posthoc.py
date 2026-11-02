"""Post-hoc (exploratory) diagnostics for the research study in docs/RESEARCH.md.

These analyses were added after the pre-registered runs to explain two results:
why the uniform mean-pooling mixer lands between the bigram baseline and
attention, and how copying is split across layer-1 heads. They retrain the
2-layer attention and uniform-mixer arms with the same seeds and budget, so the
models are identical to the ones in the main study.

    uv run python benchmarks/research_posthoc.py
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

from transformer_from_scratch.data import test_sequences
from transformer_from_scratch.research import (
    StudyConfig,
    copy_accuracy_by_position,
    mean_std,
    multiset_oracle_by_position,
    source_mass_by_position,
    train_arm,
    variants,
)


def _analyze(args: tuple[StudyConfig, int]) -> dict[str, object]:
    cfg, seed = args
    arms = {v.name: v for v in variants(cfg)}
    task = cfg.task
    held_out = test_sequences(task, count=cfg.n_test, seed=seed)
    attn = train_arm(cfg, arms["2L-attn"], seed).model
    uniform = train_arm(cfg, arms["2L-uniform"], seed).model
    return {
        "seed": seed,
        "attn": copy_accuracy_by_position(attn, task, held_out),
        "uniform": copy_accuracy_by_position(uniform, task, held_out),
        "oracle": multiset_oracle_by_position(task, held_out),
        "layer1_source": [
            source_mass_by_position(attn, task, held_out, layer=0, head=h)
            for h in range(attn.cfg.n_head)
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    cfg = StudyConfig(steps=args.steps)
    jobs = [(cfg, s) for s in args.seeds]
    if args.workers <= 1:
        rows = [_analyze(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_analyze, jobs))

    n = cfg.length
    print("### Copy accuracy by target position (mean ± std over seeds)\n")
    print("| Target | 2L-attn | 2L-uniform | Multiset oracle |")
    print("|---:|---:|---:|---:|")
    for k in range(n):
        cells = []
        for key in ("attn", "uniform", "oracle"):
            vals = [float(r[key][k]) for r in rows]  # type: ignore[index]
            m, sd = mean_std(vals)
            cells.append(f"{m:.3f} ± {sd:.3f}")
        print(f"| {k + 1} | " + " | ".join(cells) + " |")

    print("\n### Layer-1 attention mass on the source position, per copy target\n")
    print("| Seed | Head | " + " | ".join(f"t{k + 1}" for k in range(n)) + " |")
    print("|---:|---|" + "---:|" * n)
    for r in rows:
        for h, masses in enumerate(r["layer1_source"]):  # type: ignore[arg-type]
            print(f"| {r['seed']} | L1H{h} | " + " | ".join(f"{m:.2f}" for m in masses) + " |")


if __name__ == "__main__":
    main()
