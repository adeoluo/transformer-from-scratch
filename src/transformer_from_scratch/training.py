"""Training the transformer with an Adam optimizer written from scratch.

The loop processes a minibatch of sequences per step, accumulating the summed
loss into a single autodiff graph, running one backward pass, and applying Adam.
Gradients are zeroed between steps. Everything is deterministic given a seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from transformer_from_scratch.data import CopyTask, to_inputs_targets, train_sequences
from transformer_from_scratch.model import ModelConfig, Transformer
from transformer_from_scratch.tensor import Tensor


@dataclass(frozen=True)
class TrainConfig:
    """Optimization hyperparameters."""

    steps: int = 300
    batch_size: int = 16
    learning_rate: float = 0.02
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    seed: int = 0


class Adam:
    """The Adam optimizer over a fixed list of parameter tensors."""

    def __init__(self, params: list[Tensor], cfg: TrainConfig) -> None:
        self.params = params
        self.cfg = cfg
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self) -> None:
        cfg = self.cfg
        self.t += 1
        bias1 = 1.0 - cfg.beta1**self.t
        bias2 = 1.0 - cfg.beta2**self.t
        for p, m, v in zip(self.params, self.m, self.v, strict=True):
            g = p.grad + cfg.weight_decay * p.data if cfg.weight_decay else p.grad
            m *= cfg.beta1
            m += (1.0 - cfg.beta1) * g
            v *= cfg.beta2
            v += (1.0 - cfg.beta2) * g * g
            p.data -= cfg.learning_rate * (m / bias1) / (np.sqrt(v / bias2) + cfg.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()


@dataclass
class TrainResult:
    """The trained model and its per-checkpoint loss history."""

    model: Transformer
    losses: list[tuple[int, float]] = field(default_factory=list)


def train(
    task: CopyTask,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    *,
    log_every: int = 50,
) -> TrainResult:
    """Train a transformer on the copy task and return it with a loss history."""

    train_cfg = train_cfg or TrainConfig()
    model_cfg = model_cfg or ModelConfig(vocab_size=task.vocab_size, context=task.seq_len)
    model = Transformer(model_cfg, seed=train_cfg.seed)
    optimizer = Adam(model.parameters(), train_cfg)
    rng = random.Random(train_cfg.seed + 1)
    sequences = train_sequences(task, seed=train_cfg.seed)
    history: list[tuple[int, float]] = []

    for step in range(1, train_cfg.steps + 1):
        batch = [rng.choice(sequences) for _ in range(train_cfg.batch_size)]
        optimizer.zero_grad()
        losses = []
        for seq in batch:
            inputs, targets = to_inputs_targets(seq)
            losses.append(model.loss(inputs, targets))
        total = _sum_losses(losses)
        total.backward()
        _average_grads(model.parameters(), train_cfg.batch_size)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == train_cfg.steps:
            history.append((step, total.data[0][0] / train_cfg.batch_size))

    return TrainResult(model=model, losses=history)


def _sum_losses(losses: list[Tensor]) -> Tensor:
    from transformer_from_scratch import tensor as tops

    acc = losses[0]
    for nxt in losses[1:]:
        acc = tops.add(acc, nxt)
    return acc


def _average_grads(params: list[Tensor], batch_size: int) -> None:
    inv = 1.0 / batch_size
    for p in params:
        p.grad *= inv
