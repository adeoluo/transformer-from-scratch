"""Inspecting what the trained model attends to.

The copy task is solved by an induction head: when predicting a token in the
second copy, the head attends to the matching position in the first copy. This
module recomputes a single head's causal attention pattern for a sequence so the
behavior can be read off directly, and summarizes how much weight lands on the
induction offset. The same hooks are reused by the interpretability project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from transformer_from_scratch.model import Transformer
from transformer_from_scratch.tensor import softmax_row


def _forward_to_layer(model: Transformer, token_ids: list[int], layer: int) -> list[list[float]]:
    """Return the residual-stream input to ``layer`` as plain floats."""

    from transformer_from_scratch import tensor as T

    tok = T.embedding(model.token_embed, token_ids)
    pos = T.embedding(model.pos_embed, list(range(len(token_ids))))
    x = T.add(tok, pos)
    for block in model.blocks[:layer]:
        x = block.forward(x)
    return [row[:] for row in model.blocks[layer].ln1.forward(x).data]


def attention_pattern(
    model: Transformer, token_ids: list[int], *, layer: int, head: int
) -> list[list[float]]:
    """Return the ``(T, T)`` causal attention weights of one layer and head."""

    x = _forward_to_layer(model, token_ids, layer)
    attn_head = model.blocks[layer].attn.heads[head]
    w_q = attn_head.w_q.data
    w_k = attn_head.w_k.data
    d_head = attn_head.w_q.cols
    scale = 1.0 / math.sqrt(d_head)

    def project(weights: list[list[float]]) -> list[list[float]]:
        out = []
        for row in x:
            out.append(
                [sum(row[p] * weights[p][j] for p in range(len(row))) for j in range(d_head)]
            )
        return out

    q = project(w_q)
    k = project(w_k)
    pattern: list[list[float]] = []
    for i in range(len(token_ids)):
        scores = [sum(q[i][d] * k[j][d] for d in range(d_head)) * scale for j in range(i + 1)]
        weights = softmax_row(scores)
        pattern.append(weights + [0.0] * (len(token_ids) - i - 1))
    return pattern


@dataclass(frozen=True)
class SharpnessSummary:
    """How peaked a head's attention is over the copy region of a sequence."""

    layer: int
    head: int
    mean_peak_weight: float
    mean_entropy: float


def attention_sharpness(
    model: Transformer, token_ids: list[int], *, layer: int, head: int, start: int
) -> SharpnessSummary:
    """Summarize how confident a head's attention is from position ``start`` on.

    An untrained head spreads its weight almost uniformly over the allowed
    positions (high entropy, low peak). A head that has learned to copy places
    most of its weight on a few source positions (low entropy, high peak). The
    contrast is a direct, honest readout that structured attention was learned.
    """

    pattern = attention_pattern(model, token_ids, layer=layer, head=head)
    peaks: list[float] = []
    entropies: list[float] = []
    for i in range(start, len(token_ids)):
        row = pattern[i][: i + 1]
        peaks.append(max(row))
        entropies.append(-sum(w * math.log(w) for w in row if w > 0.0))
    n = len(peaks)
    return SharpnessSummary(
        layer=layer,
        head=head,
        mean_peak_weight=sum(peaks) / n if n else 0.0,
        mean_entropy=sum(entropies) / n if n else 0.0,
    )


def sharpest_head(
    model: Transformer, sequences: list[list[int]], *, start: int
) -> SharpnessSummary:
    """Return the layer and head with the most peaked attention across sequences."""

    best: SharpnessSummary | None = None
    for layer in range(model.cfg.n_layer):
        for head in range(model.cfg.n_head):
            peaks = [
                attention_sharpness(
                    model, seq, layer=layer, head=head, start=start
                ).mean_peak_weight
                for seq in sequences
            ]
            mean_peak = sum(peaks) / len(peaks) if peaks else 0.0
            if best is None or mean_peak > best.mean_peak_weight:
                best = SharpnessSummary(
                    layer=layer, head=head, mean_peak_weight=mean_peak, mean_entropy=0.0
                )
    assert best is not None
    return best
