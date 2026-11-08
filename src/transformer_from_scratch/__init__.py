"""Transformer From Scratch - an autodiff engine and a transformer language model."""

from __future__ import annotations

from transformer_from_scratch.data import CopyTask, test_sequences, train_sequences
from transformer_from_scratch.evaluation import (
    BigramBaseline,
    EvalReport,
    evaluate,
    evaluate_bigram,
)
from transformer_from_scratch.inspect import attention_pattern, attention_sharpness, sharpest_head
from transformer_from_scratch.model import ModelConfig, Transformer
from transformer_from_scratch.tensor import Tensor
from transformer_from_scratch.text import (
    CORPUS,
    CharVocab,
    TextTrainResult,
    bits_per_char,
    generate_text,
    train_char_lm,
    unigram_bits_per_char,
)
from transformer_from_scratch.training import Adam, TrainConfig, TrainResult, train

__version__ = "0.2.0"

__all__ = [
    "CORPUS",
    "Adam",
    "BigramBaseline",
    "CharVocab",
    "CopyTask",
    "EvalReport",
    "ModelConfig",
    "Tensor",
    "TextTrainResult",
    "TrainConfig",
    "TrainResult",
    "Transformer",
    "__version__",
    "attention_pattern",
    "attention_sharpness",
    "bits_per_char",
    "evaluate",
    "evaluate_bigram",
    "generate_text",
    "sharpest_head",
    "test_sequences",
    "train",
    "train_char_lm",
    "train_sequences",
    "unigram_bits_per_char",
]
