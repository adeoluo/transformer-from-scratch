"""The induction (copy) task: data the model can only solve by attending back.

A sequence is a random string of symbols, a separator, then the same string
again. The language model is trained to predict the next token everywhere, but
the interesting positions are in the second copy: to predict them the model must
look back to the matching position in the first copy, which is exactly what an
attention induction head does. A fixed-order n-gram model cannot, because the
symbol to copy depends on a token many positions earlier, not on the local
context.

Everything is generated with a seeded random generator, and the train and test
splits are disjoint strings, so held-out accuracy measures generalization.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SEPARATOR = 0  # token id reserved as the copy separator


@dataclass(frozen=True)
class CopyTask:
    """Describes the vocabulary and shape of the copy task."""

    n_symbols: int = 4
    length: int = 5

    @property
    def vocab_size(self) -> int:
        return self.n_symbols + 1  # symbols are 1..n_symbols; 0 is the separator

    @property
    def seq_len(self) -> int:
        return 2 * self.length + 1

    def make_sequence(self, rng: random.Random) -> list[int]:
        body = [rng.randint(1, self.n_symbols) for _ in range(self.length)]
        return [*body, SEPARATOR, *body]

    def copy_region(self) -> range:
        """Target positions that require copying (the second half of the sequence)."""

        return range(self.length, self.seq_len - 1)


def _split_capacity(task: CopyTask) -> int:
    """Return the largest per-split count that keeps train and test disjoint.

    There are ``n_symbols ** length`` distinct sequences; half of them can go to
    each split. A task with fewer than two distinct sequences cannot be split.
    """

    total = int(task.n_symbols**task.length)
    if total < 2:
        raise ValueError(
            "copy task has fewer than two distinct sequences; increase --symbols or --length"
        )
    return total // 2


def _unique_sequences(
    task: CopyTask, count: int, rng: random.Random, exclude: set[tuple[int, ...]]
) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set(exclude)
    out: list[list[int]] = []
    attempts = 0
    while len(out) < count and attempts < count * 200:
        attempts += 1
        seq = task.make_sequence(rng)
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        out.append(seq)
    if len(out) < count:
        raise RuntimeError(f"could only generate {len(out)} of {count} sequences")
    return out


def train_sequences(task: CopyTask, count: int = 64, *, seed: int = 0) -> list[list[int]]:
    """Return the training split of copy sequences.

    The number of distinct sequences is ``n_symbols ** length``. The requested
    count is capped at half of that space so the training and held-out splits can
    always be disjoint; for the default task the cap never binds, so behavior is
    unchanged, while small demo configurations no longer fail to generate.
    """

    count = min(count, _split_capacity(task))
    return _unique_sequences(task, count, random.Random(seed), set())


def test_sequences(task: CopyTask, count: int = 64, *, seed: int = 0) -> list[list[int]]:
    """Return a held-out split disjoint from :func:`train_sequences`."""

    count = min(count, _split_capacity(task))
    train = {tuple(s) for s in train_sequences(task, seed=seed)}
    return _unique_sequences(task, count, random.Random(seed + 7919), train)


def to_inputs_targets(sequence: list[int]) -> tuple[list[int], list[int]]:
    """Split a sequence into model inputs and next-token targets."""

    return sequence[:-1], sequence[1:]
