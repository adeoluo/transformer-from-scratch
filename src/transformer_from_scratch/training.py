"""Training the transformer with an Adam optimizer written from scratch.

The loop processes a minibatch of sequences per step, accumulating the summed
loss into a single autodiff graph, running one backward pass, and applying Adam.
Gradients are zeroed between steps. Everything is deterministic given a seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

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
        self.m = [[[0.0] * p.cols for _ in range(p.rows)] for p in params]
        self.v = [[[0.0] * p.cols for _ in range(p.rows)] for p in params]

    def step(self) -> None:
        cfg = self.cfg
        self.t += 1
        bias1 = 1.0 - cfg.beta1**self.t
        bias2 = 1.0 - cfg.beta2**self.t
        for idx, p in enumerate(self.params):
            m = self.m[idx]
            v = self.v[idx]
            for i in range(p.rows):
                g_row = p.grad[i]
                p_row = p.data[i]
                m_row = m[i]
                v_row = v[i]
                for j in range(p.cols):
                    g = g_row[j]
                    if cfg.weight_decay:
                        g += cfg.weight_decay * p_row[j]
                    m_row[j] = cfg.beta1 * m_row[j] + (1.0 - cfg.beta1) * g
                    v_row[j] = cfg.beta2 * v_row[j] + (1.0 - cfg.beta2) * g * g
                    m_hat = m_row[j] / bias1
                    v_hat = v_row[j] / bias2
                    p_row[j] -= cfg.learning_rate * m_hat / (math.sqrt(v_hat) + cfg.eps)

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
        for row in p.grad:
            for j in range(len(row)):
                row[j] *= inv
