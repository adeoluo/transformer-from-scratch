"""Character-level language modeling on real English text.

The copy task isolates what attention buys, but it is synthetic. This module
trains the same transformer on a bundled excerpt of real public-domain prose
(the opening of Jane Austen's *Pride and Prejudice*, published 1813) at the
character level, and measures held-out compression in **bits per character**
against a unigram baseline. Lower bits per character means the model has learned
real structure in natural language, not just a toy pattern.

Everything is deterministic and offline: the corpus is bundled, training is
seeded, and no network or external data is used.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from transformer_from_scratch.model import ModelConfig, Transformer
from transformer_from_scratch.tensor import softmax_row
from transformer_from_scratch.training import Adam, TrainConfig

# A public-domain excerpt (Jane Austen, *Pride and Prejudice*, 1813).
CORPUS = (
    "It is a truth universally acknowledged, that a single man in possession of a "
    "good fortune, must be in want of a wife. However little known the feelings or "
    "views of such a man may be on his first entering a neighbourhood, this truth is "
    "so well fixed in the minds of the surrounding families, that he is considered "
    "the rightful property of some one or other of their daughters. My dear Mr. "
    "Bennet, said his lady to him one day, have you heard that Netherfield Park is "
    "let at last? Mr. Bennet replied that he had not. But it is, returned she; for "
    "Mrs. Long has just been here, and she told me all about it. Mr. Bennet made no "
    "answer. Do you not want to know who has taken it? cried his wife impatiently. "
    "You want to tell me, and I have no objection to hearing it. This was invitation "
    "enough. Why, my dear, you must know, Mrs. Long says that Netherfield is taken by "
    "a young man of large fortune from the north of England; that he came down on "
    "Monday in a chaise and four to see the place, and was so much delighted with it "
    "that he agreed with Mr. Morris immediately; that he is to take possession before "
    "Michaelmas, and some of his servants are to be in the house by the end of next "
    "week. What is his name? Bingley. Is he married or single? Oh! single, my dear, "
    "to be sure! A single man of large fortune; four or five thousand a year. What a "
    "fine thing for our girls! How so? How can it affect them? My dear Mr. Bennet, "
    "replied his wife, how can you be so tiresome! You must know that I am thinking of "
    "his marrying one of them. Is that his design in settling here? Design! Nonsense, "
    "how can you talk so! But it is very likely that he may fall in love with one of "
    "them, and therefore you must visit him as soon as he comes."
)


@dataclass(frozen=True)
class CharVocab:
    """A bijective map between characters and integer ids."""

    chars: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> CharVocab:
        return cls(tuple(sorted(set(text))))

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        index = {ch: i for i, ch in enumerate(self.chars)}
        return [index[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


@dataclass(frozen=True)
class TextTrainResult:
    """The trained model, its vocabulary, and the training loss history."""

    model: Transformer
    vocab: CharVocab
    losses: list[float]
    bits_per_char: float
    baseline_bits_per_char: float


def _windows(ids: list[int], context: int) -> list[tuple[list[int], list[int]]]:
    """Return (input, target) windows where target is the input shifted by one."""

    pairs: list[tuple[list[int], list[int]]] = []
    for start in range(0, len(ids) - context):
        chunk = ids[start : start + context + 1]
        pairs.append((chunk[:-1], chunk[1:]))
    return pairs


def unigram_bits_per_char(train_text: str, eval_text: str) -> float:
    """Return the cross-entropy (in bits) of a unigram model fit on ``train_text``."""

    counts: dict[str, int] = {}
    for ch in train_text:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(train_text)
    vocab = len(counts)
    # Laplace smoothing so unseen eval characters do not give infinite loss.
    total_bits = 0.0
    for ch in eval_text:
        probability = (counts.get(ch, 0) + 1) / (total + vocab)
        total_bits += -math.log2(probability)
    return total_bits / len(eval_text) if eval_text else 0.0


def bits_per_char(model: Transformer, vocab: CharVocab, text: str) -> float:
    """Return the model's held-out cross-entropy in bits per character.

    Each character is predicted from the preceding context (up to the model's
    window), so this is the compression rate a arithmetic coder would achieve.
    """

    ids = vocab.encode(text)
    context = model.cfg.context
    total_bits = 0.0
    counted = 0
    for position in range(1, len(ids)):
        window = ids[max(0, position - context) : position]
        logits = model.logits(window)
        probabilities = softmax_row(logits.data[-1])
        probability = max(probabilities[ids[position]], 1e-12)
        total_bits += -math.log2(probability)
        counted += 1
    return total_bits / counted if counted else 0.0


def train_char_lm(
    *,
    text: str = CORPUS,
    context: int = 32,
    d_model: int = 48,
    n_head: int = 4,
    n_layer: int = 2,
    d_ff: int = 96,
    steps: int = 400,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    holdout_fraction: float = 0.15,
    seed: int = 0,
) -> TextTrainResult:
    """Train a character-level transformer on ``text`` and measure compression.

    The text is split into a training prefix and a held-out suffix. Bits per
    character on the held-out suffix is compared against a unigram baseline fit on
    the training prefix, so the gap measures the structure the model learned.
    """

    vocab = CharVocab.from_text(text)
    split = int(len(text) * (1.0 - holdout_fraction))
    train_text, held_out = text[:split], text[split:]
    train_ids = vocab.encode(train_text)

    cfg = ModelConfig(
        vocab_size=vocab.size,
        context=context,
        d_model=d_model,
        n_head=n_head,
        n_layer=n_layer,
        d_ff=d_ff,
    )
    model = Transformer(cfg, seed=seed)
    windows = _windows(train_ids, context)
    if not windows:
        raise ValueError("corpus is shorter than the context window")

    optimizer = Adam(model.parameters(), TrainConfig(learning_rate=learning_rate, seed=seed))
    rng = random.Random(seed)
    losses: list[float] = []
    for _ in range(steps):
        for param in model.parameters():
            param.zero_grad()
        batch = [rng.choice(windows) for _ in range(batch_size)]
        step_loss = 0.0
        for inputs, targets in batch:
            loss = model.loss(inputs, targets)
            loss.backward()
            step_loss += loss.data[0][0]
        for param in model.parameters():
            for i in range(param.rows):
                row = param.grad[i]
                for j in range(param.cols):
                    row[j] /= batch_size
        optimizer.step()
        losses.append(step_loss / batch_size)

    model_bits = bits_per_char(model, vocab, held_out)
    baseline_bits = unigram_bits_per_char(train_text, held_out)
    return TextTrainResult(
        model=model,
        vocab=vocab,
        losses=losses,
        bits_per_char=model_bits,
        baseline_bits_per_char=baseline_bits,
    )


def generate_text(
    model: Transformer,
    vocab: CharVocab,
    prompt: str,
    n_chars: int,
    *,
    temperature: float = 0.8,
    seed: int = 0,
) -> str:
    """Sample ``n_chars`` continuation characters from ``prompt``."""

    rng = random.Random(seed)
    ids = vocab.encode(prompt)
    for _ in range(n_chars):
        window = ids[-model.cfg.context :]
        logits = model.logits(window)
        scaled = [value / max(temperature, 1e-6) for value in logits.data[-1]]
        probabilities = softmax_row(scaled)
        ids.append(_sample(probabilities, rng))
    return vocab.decode(ids)


def _sample(probabilities: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1
