"""A decoder-only transformer language model built on the autodiff engine.

The model is intentionally small and explicit: learned token and positional
embeddings, a stack of pre-norm transformer blocks (multi-head causal
self-attention followed by a GELU feed-forward network), a final layer norm, and
a weight-tied output projection. Every parameter is a :class:`Tensor`, and the
forward pass is composed entirely from the operations in :mod:`tensor`, so a
single call to ``loss.backward()`` differentiates the whole network.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformer_from_scratch import tensor as T
from transformer_from_scratch.tensor import Tensor

MIXERS = ("attention", "uniform")


@dataclass(frozen=True)
class ModelConfig:
    """Shape and size of the transformer."""

    vocab_size: int
    context: int
    d_model: int = 24
    n_head: int = 3
    n_layer: int = 2
    d_ff: int = 48
    mixer: str = "attention"

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.mixer not in MIXERS:
            raise ValueError(f"mixer must be one of {MIXERS}, got {self.mixer!r}")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head


def _rand_matrix(rows: int, cols: int, scale: float, rng: random.Random) -> list[list[float]]:
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


def uniform_causal_weights(t: int) -> Tensor:
    """Return the fixed ``(t, t)`` causal mean-pooling matrix (row ``i`` is ``1/(i+1)``)."""

    data = [[1.0 / (i + 1) if j <= i else 0.0 for j in range(t)] for i in range(t)]
    return Tensor(data, requires_grad=False)


class AttentionHead:
    """One causal self-attention head with its own query, key, and value maps.

    With ``uniform=True`` the learned softmax(QK) weights are replaced by a fixed
    causal mean over all previous positions, a content-independent mixer used as
    a matched-budget control in the research study. The query and key matrices
    are still created (so the random stream and parameter layout match the
    attention model) but receive no gradient. Setting ``ablated`` makes the head
    output zeros, which is how the study measures a head's causal contribution.
    """

    def __init__(
        self, d_model: int, d_head: int, rng: random.Random, *, uniform: bool = False
    ) -> None:
        s = 1.0 / math.sqrt(d_model)
        self.w_q = Tensor(_rand_matrix(d_model, d_head, s, rng))
        self.w_k = Tensor(_rand_matrix(d_model, d_head, s, rng))
        self.w_v = Tensor(_rand_matrix(d_model, d_head, s, rng))
        self.scale = 1.0 / math.sqrt(d_head)
        self.d_head = d_head
        self.uniform = uniform
        self.ablated = False

    def parameters(self) -> list[Tensor]:
        return [self.w_q, self.w_k, self.w_v]

    def forward(self, x: Tensor) -> Tensor:
        if self.ablated:
            return Tensor([[0.0] * self.d_head for _ in range(x.rows)], requires_grad=False)
        v = T.matmul(x, self.w_v)
        if self.uniform:
            return T.matmul(uniform_causal_weights(x.rows), v)
        q = T.matmul(x, self.w_q)
        k = T.matmul(x, self.w_k)
        scores = T.scale(T.matmul(q, T.transpose(k)), self.scale)
        weights = T.causal_softmax(scores)
        return T.matmul(weights, v)


class MultiHeadAttention:
    """Several attention heads concatenated and projected back to ``d_model``."""

    def __init__(self, cfg: ModelConfig, rng: random.Random) -> None:
        uniform = cfg.mixer == "uniform"
        self.heads = [
            AttentionHead(cfg.d_model, cfg.d_head, rng, uniform=uniform) for _ in range(cfg.n_head)
        ]
        s = 1.0 / math.sqrt(cfg.d_model)
        self.w_o = Tensor(_rand_matrix(cfg.d_model, cfg.d_model, s, rng))
        self.b_o = Tensor([[0.0] * cfg.d_model])

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = [self.w_o, self.b_o]
        for head in self.heads:
            params.extend(head.parameters())
        return params

    def forward(self, x: Tensor) -> Tensor:
        head_outputs = [head.forward(x) for head in self.heads]
        combined = T.concat_cols(head_outputs)
        return T.add_bias(T.matmul(combined, self.w_o), self.b_o)


class FeedForward:
    """A two-layer GELU feed-forward network."""

    def __init__(self, cfg: ModelConfig, rng: random.Random) -> None:
        s1 = 1.0 / math.sqrt(cfg.d_model)
        s2 = 1.0 / math.sqrt(cfg.d_ff)
        self.w1 = Tensor(_rand_matrix(cfg.d_model, cfg.d_ff, s1, rng))
        self.b1 = Tensor([[0.0] * cfg.d_ff])
        self.w2 = Tensor(_rand_matrix(cfg.d_ff, cfg.d_model, s2, rng))
        self.b2 = Tensor([[0.0] * cfg.d_model])

    def parameters(self) -> list[Tensor]:
        return [self.w1, self.b1, self.w2, self.b2]

    def forward(self, x: Tensor) -> Tensor:
        hidden = T.gelu(T.add_bias(T.matmul(x, self.w1), self.b1))
        return T.add_bias(T.matmul(hidden, self.w2), self.b2)


class _LayerNormParams:
    def __init__(self, d_model: int) -> None:
        self.gamma = Tensor([[1.0] * d_model])
        self.beta = Tensor([[0.0] * d_model])

    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]

    def forward(self, x: Tensor) -> Tensor:
        return T.layer_norm(x, self.gamma, self.beta)


class Block:
    """A pre-norm transformer block: attention and feed-forward, each residual."""

    def __init__(self, cfg: ModelConfig, rng: random.Random) -> None:
        self.ln1 = _LayerNormParams(cfg.d_model)
        self.attn = MultiHeadAttention(cfg, rng)
        self.ln2 = _LayerNormParams(cfg.d_model)
        self.ff = FeedForward(cfg, rng)

    def parameters(self) -> list[Tensor]:
        return [
            *self.ln1.parameters(),
            *self.attn.parameters(),
            *self.ln2.parameters(),
            *self.ff.parameters(),
        ]

    def forward(self, x: Tensor) -> Tensor:
        x = T.add(x, self.attn.forward(self.ln1.forward(x)))
        return T.add(x, self.ff.forward(self.ln2.forward(x)))


class Transformer:
    """A decoder-only language model with weight-tied input and output embeddings."""

    def __init__(self, cfg: ModelConfig, *, seed: int = 0) -> None:
        self.cfg = cfg
        rng = random.Random(seed)
        s = 1.0 / math.sqrt(cfg.d_model)
        self.token_embed = Tensor(_rand_matrix(cfg.vocab_size, cfg.d_model, s, rng))
        self.pos_embed = Tensor(_rand_matrix(cfg.context, cfg.d_model, s, rng))
        self.blocks = [Block(cfg, rng) for _ in range(cfg.n_layer)]
        self.ln_f = _LayerNormParams(cfg.d_model)

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = [self.token_embed, self.pos_embed, *self.ln_f.parameters()]
        for block in self.blocks:
            params.extend(block.parameters())
        return params

    def logits(self, token_ids: list[int]) -> Tensor:
        """Return ``(T, vocab)`` next-token logits for a sequence of token ids."""

        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        if len(token_ids) > self.cfg.context:
            raise ValueError("sequence longer than the model context")
        tok = T.embedding(self.token_embed, token_ids)
        pos = T.embedding(self.pos_embed, list(range(len(token_ids))))
        x = T.add(tok, pos)
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_f.forward(x)
        return T.matmul(x, T.transpose(self.token_embed))

    def loss(self, token_ids: list[int], targets: list[int]) -> Tensor:
        return T.cross_entropy(self.logits(token_ids), targets)

    def to_state(self) -> dict[str, Any]:
        """Serialize the config and every parameter matrix to a plain dict.

        Parameters are stored in the fixed order returned by :meth:`parameters`,
        so loading reconstructs a model with the same cfg and then copies the
        values back in that order. This is the checkpoint format used by
        :meth:`save` and :meth:`load`.
        """

        return {
            "format": 1,
            "config": asdict(self.cfg),
            "parameters": [p.data for p in self.parameters()],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Transformer:
        """Rebuild a model from :meth:`to_state` output."""

        cfg = ModelConfig(**state["config"])
        model = cls(cfg)
        saved = state["parameters"]
        params = model.parameters()
        if len(saved) != len(params):
            raise ValueError("checkpoint parameter count does not match the model")
        for param, values in zip(params, saved, strict=True):
            if (len(values), len(values[0]) if values else 0) != param.shape:
                raise ValueError("checkpoint parameter shape does not match the model")
            for i in range(param.rows):
                param.data[i] = list(values[i])
        return model

    def save(self, path: str | Path) -> None:
        """Write a JSON checkpoint of the config and parameters to ``path``."""

        Path(path).write_text(json.dumps(self.to_state()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Transformer:
        """Load a model from a JSON checkpoint written by :meth:`save`."""

        return cls.from_state(json.loads(Path(path).read_text(encoding="utf-8")))
