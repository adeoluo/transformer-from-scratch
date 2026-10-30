"""A deterministic scale benchmark for transformer inference throughput.

Trains one small transformer on the copy task, then runs the forward pass
(``model.logits``) over progressively larger batches of held-out sequences.
For each workload it reports wall-clock seconds, inference throughput in
sequences per second, peak memory in MiB measured with the standard-library
``tracemalloc`` module, and held-out copy accuracy as the quality metric.

The data is synthetic and the model is intentionally tiny. This pure-Python
from-scratch implementation is not competitive on wall-clock time with compiled
tensor libraries; the point is to show how the hand-written forward pass scales,
not to beat PyTorch. Deterministic and offline; run with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

from transformer_from_scratch.data import CopyTask, test_sequences
from transformer_from_scratch.evaluation import evaluate
from transformer_from_scratch.model import ModelConfig, Transformer
from transformer_from_scratch.training import TrainConfig, train

WORKLOAD_SIZES: tuple[int, ...] = (40, 80, 160, 320)
_BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class ScaleRow:
    """One measured workload: size, timing, throughput, memory, and quality."""

    size: int
    seconds: float
    throughput: float
    peak_mib: float
    copy_accuracy: float


def _train_model(steps: int) -> tuple[Transformer, CopyTask]:
    task = CopyTask()
    cfg = ModelConfig(vocab_size=task.vocab_size, context=task.seq_len)
    result = train(task, cfg, TrainConfig(steps=steps))
    return result.model, task


def run_scale(sizes: tuple[int, ...] = WORKLOAD_SIZES, *, steps: int = 80) -> list[ScaleRow]:
    """Train one small model, then measure forward-pass scaling across sizes."""

    model, task = _train_model(steps)
    rows: list[ScaleRow] = []
    tracemalloc.start()
    for size in sizes:
        sequences = test_sequences(task, count=size, seed=0)
        tracemalloc.reset_peak()
        start = time.perf_counter()
        report = evaluate(model, task, sequences)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        rows.append(
            ScaleRow(
                size=len(sequences),
                seconds=elapsed,
                throughput=len(sequences) / elapsed if elapsed else 0.0,
                peak_mib=peak / _BYTES_PER_MIB,
                copy_accuracy=report.copy_accuracy,
            )
        )
    tracemalloc.stop()
    return rows


def main() -> None:
    print("Synthetic copy-task data; tiny from-scratch transformer.")
    print("Pure-Python inference; not benchmarked against compiled tensor libraries.")
    print(f"{'sequences':>10}  {'seconds':>9}  {'seq/sec':>9}  {'peak MiB':>9}  {'copy acc':>9}")
    for row in run_scale():
        print(
            f"{row.size:>10d}  {row.seconds:>9.3f}  {row.throughput:>9.1f}  "
            f"{row.peak_mib:>9.2f}  {row.copy_accuracy:>9.3f}"
        )


if __name__ == "__main__":
    main()
