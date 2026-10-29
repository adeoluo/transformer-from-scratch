"""A deterministic micro-benchmark for training and the copy task.

Trains the default transformer on the copy task and reports training time,
held-out copy accuracy against the bigram baseline, perplexity, and the sharpest
attention head. Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from transformer_from_scratch.data import CopyTask, test_sequences, train_sequences
from transformer_from_scratch.evaluation import BigramBaseline, evaluate, evaluate_bigram
from transformer_from_scratch.inspect import sharpest_head
from transformer_from_scratch.model import ModelConfig, Transformer
from transformer_from_scratch.training import TrainConfig, train


def main() -> None:
    task = CopyTask()
    cfg = ModelConfig(vocab_size=task.vocab_size, context=task.seq_len)
    test_set = test_sequences(task, seed=0)

    untrained_peak = sharpest_head(
        Transformer(cfg, seed=0), test_set[:20], start=task.length
    ).mean_peak_weight

    start = time.perf_counter()
    result = train(task, cfg, TrainConfig(), log_every=100)
    elapsed = time.perf_counter() - start

    model_report = evaluate(result.model, task, test_set)
    baseline = BigramBaseline.fit(task.vocab_size, train_sequences(task, seed=0))
    bigram_report = evaluate_bigram(baseline, task, test_set)
    trained_peak = sharpest_head(result.model, test_set[:20], start=task.length).mean_peak_weight

    print(f"Task: copy {task.length} symbols from {task.n_symbols}, context {task.seq_len}")
    print(f"Model: d_model {cfg.d_model}, {cfg.n_head} heads, {cfg.n_layer} layers")
    print(f"Training (300 steps): {elapsed:.0f} s")
    print(
        f"Held-out copy accuracy: {model_report.copy_accuracy:.3f} "
        f"(bigram {bigram_report.copy_accuracy:.3f})"
    )
    print(
        f"Overall accuracy: {model_report.overall_accuracy:.3f}  perplexity: "
        f"{model_report.perplexity:.3f}"
    )
    print(f"Sharpest attention head peak: {untrained_peak:.3f} -> {trained_peak:.3f}")


if __name__ == "__main__":
    main()
