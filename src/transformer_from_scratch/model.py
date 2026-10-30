"""A decoder-only transformer language model built on the autodiff engine.

The model is intentionally small and explicit: learned token and positional
embeddings, a stack of pre-norm transformer blocks (multi-head causal
self-attention followed by a GELU feed-forward network), a final layer norm, and
a weight-tied output projection. Every parameter is a :class:`Tensor`, and the
forward pass is composed entirely from the operations in :mod:`tensor`, so a
single call to ``loss.backward()`` differentiates the whole network.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from transformer_from_scratch import tensor as T
from transformer_from_scratch.tensor import Tensor


@dataclass(frozen=True)
class ModelConfig:
    """Shape and size of the transformer."""

    vocab_size: int
    context: int
    d_model: int = 24
    n_head: int = 3
    n_layer: int = 2
    d_ff: int = 48

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head


def _rand_matrix(rows: int, cols: int, scale: float, rng: random.Random) -> list[list[float]]:
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


class AttentionHead:
    """One causal self-attention head with its own query, key, and value maps."""

    def __init__(self, d_model: int, d_head: int, rng: random.Random) -> None:
        s = 1.0 / math.sqrt(d_model)
        self.w_q = Tensor(_rand_matrix(d_model, d_head, s, rng))
        self.w_k = Tensor(_rand_matrix(d_model, d_head, s, rng))
        self.w_v = Tensor(_rand_matrix(d_model, d_head, s, rng))
        self.scale = 1.0 / math.sqrt(d_head)

    def parameters(self) -> list[Tensor]:
        return [self.w_q, self.w_k, self.w_v]

    def forward(self, x: Tensor) -> Tensor:
        q = T.matmul(x, self.w_q)
        k = T.matmul(x, self.w_k)
        v = T.matmul(x, self.w_v)
        scores = T.scale(T.matmul(q, T.transpose(k)), self.scale)
        weights = T.causal_softmax(scores)
        return T.matmul(weights, v)


class MultiHeadAttention:
    """Several attention heads concatenated and projected back to ``d_model``."""

    def __init__(self, cfg: ModelConfig, rng: random.Random) -> None:
        self.heads = [AttentionHead(cfg.d_model, cfg.d_head, rng) for _ in range(cfg.n_head)]
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
