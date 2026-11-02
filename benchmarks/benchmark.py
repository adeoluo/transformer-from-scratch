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
from transformer_from_scratch.inspect import attention_sharpness, sharpest_head
from transformer_from_scratch.model import ModelConfig, Transformer
from transformer_from_scratch.training import TrainConfig, train


def main() -> None:
    task = CopyTask()
    cfg = ModelConfig(vocab_size=task.vocab_size, context=task.seq_len)
    test_set = test_sequences(task, seed=0)

    def head_stats(model: Transformer) -> tuple[float, float]:
        """Mean peak weight and mean entropy of the sharpest head over 20 sequences."""
        head = sharpest_head(model, test_set[:20], start=task.length)
        entropies = [
            attention_sharpness(
                model, seq, layer=head.layer, head=head.head, start=task.length
            ).mean_entropy
            for seq in test_set[:20]
        ]
        return head.mean_peak_weight, sum(entropies) / len(entropies)

    untrained_peak, untrained_entropy = head_stats(Transformer(cfg, seed=0))

    start = time.perf_counter()
    result = train(task, cfg, TrainConfig(), log_every=100)
    elapsed = time.perf_counter() - start

    model_report = evaluate(result.model, task, test_set)
    baseline = BigramBaseline.fit(task.vocab_size, train_sequences(task, seed=0))
    bigram_report = evaluate_bigram(baseline, task, test_set)
    trained_peak, trained_entropy = head_stats(result.model)

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
    print(f"Sharpest attention head entropy: {untrained_entropy:.2f} -> {trained_entropy:.2f}")


if __name__ == "__main__":
    main()
