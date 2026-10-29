"""Shared, session-scoped fixtures so the model is trained once for all tests."""

from __future__ import annotations

import pytest

from transformer_from_scratch.data import CopyTask, test_sequences
from transformer_from_scratch.model import ModelConfig
from transformer_from_scratch.service import TransformerService
from transformer_from_scratch.training import TrainConfig, TrainResult, train


@pytest.fixture(scope="session")
def small_task() -> CopyTask:
    return CopyTask(n_symbols=4, length=4)


@pytest.fixture(scope="session")
def small_cfg(small_task: CopyTask) -> ModelConfig:
    return ModelConfig(
        vocab_size=small_task.vocab_size,
        context=small_task.seq_len,
        d_model=16,
        n_head=2,
        n_layer=2,
        d_ff=32,
    )


@pytest.fixture(scope="session")
def trained(small_task: CopyTask, small_cfg: ModelConfig) -> TrainResult:
    return train(small_task, small_cfg, TrainConfig(steps=80, batch_size=8, seed=0), log_every=80)


@pytest.fixture(scope="session")
def service(small_task: CopyTask, trained: TrainResult) -> TransformerService:
    return TransformerService(
        model=trained.model,
        task=small_task,
        test_set=test_sequences(small_task, count=20, seed=0),
    )
