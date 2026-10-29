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
from transformer_from_scratch.training import Adam, TrainConfig, TrainResult, train

__version__ = "0.1.0"

__all__ = [
    "Adam",
    "BigramBaseline",
    "CopyTask",
    "EvalReport",
    "ModelConfig",
    "Tensor",
    "TrainConfig",
    "TrainResult",
    "Transformer",
    "__version__",
    "attention_pattern",
    "attention_sharpness",
    "evaluate",
    "evaluate_bigram",
    "sharpest_head",
    "test_sequences",
    "train",
    "train_sequences",
]
