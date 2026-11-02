"""Tests for real-text character-level training and checkpointing."""

from __future__ import annotations

import tempfile
from pathlib import Path

from transformer_from_scratch.model import ModelConfig, Transformer
from transformer_from_scratch.text import (
    CORPUS,
    CharVocab,
    bits_per_char,
    generate_text,
    train_char_lm,
    unigram_bits_per_char,
)

_TINY = "It is a truth universally acknowledged that a single man wants a wife. " * 3


def test_char_vocab_round_trips() -> None:
    vocab = CharVocab.from_text("hello world")
    assert vocab.decode(vocab.encode("hello world")) == "hello world"
    # the vocabulary is sorted and deduplicated
    assert vocab.chars == tuple(sorted(set("hello world")))


def test_bundled_corpus_is_real_prose() -> None:
    assert len(CORPUS) > 1000
    assert "Bennet" in CORPUS  # a character from the public-domain source


def test_unigram_baseline_is_positive() -> None:
    bits = unigram_bits_per_char(_TINY, _TINY[:20])
    assert bits > 0.0


def test_char_training_reduces_loss() -> None:
    result = train_char_lm(
        text=_TINY,
        context=12,
        d_model=16,
        n_head=2,
        n_layer=1,
        d_ff=32,
        steps=40,
        batch_size=8,
        seed=0,
    )
    assert result.losses[-1] < result.losses[0]
    assert result.bits_per_char > 0.0
    assert result.vocab.size == len(set(_TINY))


def test_char_training_is_deterministic() -> None:
    kwargs = {
        "text": _TINY,
        "context": 12,
        "d_model": 16,
        "n_head": 2,
        "n_layer": 1,
        "d_ff": 32,
        "steps": 20,
        "batch_size": 8,
        "seed": 3,
    }
    a = train_char_lm(**kwargs)  # type: ignore[arg-type]
    b = train_char_lm(**kwargs)  # type: ignore[arg-type]
    assert a.losses == b.losses
    assert a.bits_per_char == b.bits_per_char


def test_generate_text_stays_in_vocabulary() -> None:
    result = train_char_lm(
        text=_TINY,
        context=12,
        d_model=16,
        n_head=2,
        n_layer=1,
        d_ff=32,
        steps=10,
        batch_size=8,
        seed=0,
    )
    sample = generate_text(result.model, result.vocab, "It is ", 20, seed=1)
    assert len(sample) == len("It is ") + 20
    assert all(ch in result.vocab.chars for ch in sample)


def test_bits_per_char_matches_helper() -> None:
    result = train_char_lm(
        text=_TINY,
        context=12,
        d_model=16,
        n_head=2,
        n_layer=1,
        d_ff=32,
        steps=10,
        batch_size=8,
        seed=0,
    )
    recomputed = bits_per_char(result.model, result.vocab, _TINY[-20:])
    assert recomputed > 0.0


def test_checkpoint_round_trips_exactly() -> None:
    cfg = ModelConfig(vocab_size=11, context=8, d_model=16, n_head=4, n_layer=2, d_ff=32)
    model = Transformer(cfg, seed=0)
    token_ids = [3, 1, 4, 1, 5, 9, 2]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.json"
        model.save(path)
        restored = Transformer.load(path)
    assert restored.logits(token_ids).data.tolist() == model.logits(token_ids).data.tolist()
    assert [p.data.tolist() for p in restored.parameters()] == [
        p.data.tolist() for p in model.parameters()
    ]


def test_checkpoint_rejects_mismatched_shape() -> None:
    import pytest

    cfg = ModelConfig(vocab_size=11, context=8, d_model=16, n_head=4, n_layer=2, d_ff=32)
    model = Transformer(cfg, seed=0)
    state = model.to_state()
    state["parameters"][0] = [[0.0]]  # wrong shape
    with pytest.raises(ValueError, match="shape"):
        Transformer.from_state(state)
