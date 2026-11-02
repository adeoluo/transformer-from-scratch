"""Run the pre-registered research study described in docs/RESEARCH.md.

Trains every architecture arm for every seed at the same step budget, ablates
each attention head of the 2-layer models, probes their attention patterns, and
prints Markdown tables plus the verdict for each hypothesis. Deterministic and
offline; runs are independent, so ``--workers`` changes wall time only.

    uv run python benchmarks/research_study.py                 # full study
    uv run python benchmarks/research_study.py --seeds 0 --steps 50 --no-three-layer
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from transformer_from_scratch.research import StudyConfig, format_report, run_study, to_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--no-three-layer", action="store_true")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--json", type=Path, help="write raw per-run results here")
    parser.add_argument("--markdown", type=Path, help="write the report here")
    args = parser.parse_args(argv)

    cfg = StudyConfig(
        steps=args.steps, n_test=args.n_test, include_three_layer=not args.no_three_layer
    )
    start = time.perf_counter()
    runs = run_study(cfg, args.seeds, workers=args.workers)
    elapsed = time.perf_counter() - start

    report = format_report(runs)
    cpu_seconds = sum(r.seconds for r in runs)
    footer = (
        f"\nTraining budget: {cfg.steps} steps, batch {cfg.batch_size}, lr {cfg.learning_rate}; "
        f"held-out sequences per seed: {cfg.n_test}. Total training CPU time "
        f"{cpu_seconds:.0f} s, wall time {elapsed:.0f} s with {args.workers} worker(s).\n"
    )
    print(report + footer)
    if args.markdown:
        args.markdown.write_text(report + footer, encoding="utf-8")
    if args.json:
        payload = {"config": cfg.__dict__, "seeds": args.seeds, "runs": to_json(runs)}
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
