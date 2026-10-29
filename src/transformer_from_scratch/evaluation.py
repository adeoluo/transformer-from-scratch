"""Evaluation: next-token accuracy, copy-region accuracy, perplexity, baseline.

The headline metric is accuracy on the copy region of held-out sequences, where
correct prediction requires attending back to the first copy. A fixed bigram
model, estimated from the same training data, is the baseline: it captures local
statistics but cannot copy, so the gap isolates what attention buys.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from transformer_from_scratch.data import CopyTask, to_inputs_targets
from transformer_from_scratch.model import Transformer
from transformer_from_scratch.tensor import softmax_row


@dataclass(frozen=True)
class EvalReport:
    """Accuracy and perplexity on a held-out set."""

    copy_accuracy: float
    overall_accuracy: float
    perplexity: float

    def format(self) -> str:
        return (
            f"copy_acc={self.copy_accuracy:.4f}, overall_acc={self.overall_accuracy:.4f}, "
            f"ppl={self.perplexity:.3f}"
        )


def evaluate(model: Transformer, task: CopyTask, sequences: list[list[int]]) -> EvalReport:
    """Return copy-region accuracy, overall accuracy, and perplexity."""

    copy_positions = set(task.copy_region())
    copy_correct = 0
    copy_total = 0
    overall_correct = 0
    overall_total = 0
    nll = 0.0
    nll_count = 0
    for seq in sequences:
        inputs, targets = to_inputs_targets(seq)
        logits = model.logits(inputs)
        for pos in range(len(targets)):
            row = logits.data[pos]
            pred = max(range(len(row)), key=lambda j: row[j])
            probs = softmax_row(row)
            nll += -math.log(max(probs[targets[pos]], 1e-12))
            nll_count += 1
            overall_total += 1
            if pred == targets[pos]:
                overall_correct += 1
            if pos in copy_positions:
                copy_total += 1
                if pred == targets[pos]:
                    copy_correct += 1
    return EvalReport(
        copy_accuracy=copy_correct / copy_total if copy_total else 0.0,
        overall_accuracy=overall_correct / overall_total if overall_total else 0.0,
        perplexity=math.exp(nll / nll_count) if nll_count else float("inf"),
    )


class BigramBaseline:
    """A fixed-order bigram next-token model estimated from training sequences."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self._counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    @classmethod
    def fit(cls, vocab_size: int, sequences: list[list[int]]) -> BigramBaseline:
        model = cls(vocab_size)
        from itertools import pairwise

        for seq in sequences:
            for a, b in pairwise(seq):
                model._counts[a][b] += 1
        return model

    def predict(self, prev: int) -> int:
        row = self._counts.get(prev)
        if not row:
            return 0
        return max(row, key=lambda k: row[k])


def evaluate_bigram(
    baseline: BigramBaseline, task: CopyTask, sequences: list[list[int]]
) -> EvalReport:
    """Evaluate the bigram baseline with the same metrics as the model."""

    copy_positions = set(task.copy_region())
    copy_correct = copy_total = overall_correct = overall_total = 0
    for seq in sequences:
        inputs, targets = to_inputs_targets(seq)
        for pos in range(len(targets)):
            pred = baseline.predict(inputs[pos])
            overall_total += 1
            if pred == targets[pos]:
                overall_correct += 1
            if pos in copy_positions:
                copy_total += 1
                if pred == targets[pos]:
                    copy_correct += 1
    return EvalReport(
        copy_accuracy=copy_correct / copy_total if copy_total else 0.0,
        overall_accuracy=overall_correct / overall_total if overall_total else 0.0,
        perplexity=float("nan"),
    )
