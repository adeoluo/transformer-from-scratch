"""A small reverse-mode automatic differentiation engine over dense matrices.

Everything the transformer needs is built from a single :class:`Tensor` type whose
operations each record a local backward closure. A topological sort then runs the
closures in reverse to accumulate gradients, which is reverse-mode autodiff. There
is no NumPy and no other numerical library; matrices are plain lists of lists of
floats and every gradient is written out by hand.

The operations are deliberately the minimal set a transformer needs: matrix
multiply, biased affine maps, elementwise addition, scaling, GELU, row-wise
causal softmax, row-wise layer normalization, embedding lookup, column
concatenation, and a softmax cross-entropy loss. Each is covered by a finite
difference gradient check in the test suite.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

Matrix = list[list[float]]


class Tensor:
    """A 2-D array of floats with a recorded gradient and backward closure."""

    __slots__ = ("_backward", "_prev", "cols", "data", "grad", "requires_grad")

    def __init__(
        self,
        data: Matrix,
        _children: tuple[Tensor, ...] = (),
        *,
        requires_grad: bool = True,
    ) -> None:
        self.data = data
        self.cols = len(data[0]) if data else 0
        self.grad: Matrix = [[0.0] * self.cols for _ in range(len(data))]
        self.requires_grad = requires_grad
        self._backward: Callable[[], None] = lambda: None
        self._prev: tuple[Tensor, ...] = _children

    @property
    def rows(self) -> int:
        return len(self.data)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.data), self.cols)

    def zero_grad(self) -> None:
        for row in self.grad:
            for j in range(len(row)):
                row[j] = 0.0

    def backward(self) -> None:
        """Run reverse-mode differentiation from this (scalar) tensor."""

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = [[1.0] * self.cols for _ in range(self.rows)]
        for node in reversed(topo):
            node._backward()


def _zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix product ``a @ b`` with shapes ``(m, k) @ (k, n) -> (m, n)``."""

    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"shape mismatch for matmul: {a.shape} @ {b.shape}")
    out_data = _zeros(m, n)
    for i in range(m):
        a_row = a.data[i]
        out_row = out_data[i]
        for p in range(k):
            aip = a_row[p]
            if aip == 0.0:
                continue
            b_row = b.data[p]
            for j in range(n):
                out_row[j] += aip * b_row[j]
    out = Tensor(out_data, (a, b))

    def _backward() -> None:
        for i in range(m):
            g_row = out.grad[i]
            a_row = a.data[i]
            ag_row = a.grad[i]
            for p in range(k):
                b_row = b.data[p]
                s = 0.0
                for j in range(n):
                    s += g_row[j] * b_row[j]
                ag_row[p] += s
                aip = a_row[p]
                bg_row = b.grad[p]
                for j in range(n):
                    bg_row[j] += aip * g_row[j]

    out._backward = _backward
    return out


def add_bias(x: Tensor, bias: Tensor) -> Tensor:
    """Add a row-vector ``bias`` of shape ``(1, n)`` to every row of ``x``."""

    m, n = x.shape
    if bias.shape != (1, n):
        raise ValueError(f"bias must be (1, {n}), got {bias.shape}")
    b_row = bias.data[0]
    out = Tensor([[x.data[i][j] + b_row[j] for j in range(n)] for i in range(m)], (x, bias))

    def _backward() -> None:
        for i in range(m):
            g_row = out.grad[i]
            xg_row = x.grad[i]
            for j in range(n):
                xg_row[j] += g_row[j]
                bias.grad[0][j] += g_row[j]

    out._backward = _backward
    return out


def add(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise sum of two equally shaped tensors (used for residuals)."""

    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for add: {a.shape} vs {b.shape}")
    m, n = a.shape
    out = Tensor([[a.data[i][j] + b.data[i][j] for j in range(n)] for i in range(m)], (a, b))

    def _backward() -> None:
        for i in range(m):
            g_row = out.grad[i]
            ag_row = a.grad[i]
            bg_row = b.grad[i]
            for j in range(n):
                ag_row[j] += g_row[j]
                bg_row[j] += g_row[j]

    out._backward = _backward
    return out


def scale(x: Tensor, factor: float) -> Tensor:
    """Multiply every element by a constant ``factor``."""

    m, n = x.shape
    out = Tensor([[x.data[i][j] * factor for j in range(n)] for i in range(m)], (x,))

    def _backward() -> None:
        for i in range(m):
            g_row = out.grad[i]
            xg_row = x.grad[i]
            for j in range(n):
                xg_row[j] += g_row[j] * factor

    out._backward = _backward
    return out


_GELU_C = math.sqrt(2.0 / math.pi)


def gelu(x: Tensor) -> Tensor:
    """The tanh approximation of the Gaussian error linear unit, elementwise."""

    m, n = x.shape
    out_data = _zeros(m, n)
    for i in range(m):
        for j in range(n):
            v = x.data[i][j]
            inner = _GELU_C * (v + 0.044715 * v * v * v)
            out_data[i][j] = 0.5 * v * (1.0 + math.tanh(inner))
    out = Tensor(out_data, (x,))

    def _backward() -> None:
        for i in range(m):
            g_row = out.grad[i]
            xg_row = x.grad[i]
            for j in range(n):
                v = x.data[i][j]
                inner = _GELU_C * (v + 0.044715 * v * v * v)
                tanh_inner = math.tanh(inner)
                sech2 = 1.0 - tanh_inner * tanh_inner
                dinner = _GELU_C * (1.0 + 3.0 * 0.044715 * v * v)
                deriv = 0.5 * (1.0 + tanh_inner) + 0.5 * v * sech2 * dinner
                xg_row[j] += g_row[j] * deriv

    out._backward = _backward
    return out


def causal_softmax(scores: Tensor) -> Tensor:
    """Row-wise softmax where row ``i`` attends only to columns ``j <= i``.

    ``scores`` is square ``(T, T)``. Masked positions receive zero weight and no
    gradient, implementing autoregressive attention without materializing an
    additive ``-inf`` mask.
    """

    t, t2 = scores.shape
    if t != t2:
        raise ValueError(f"causal_softmax expects a square matrix, got {scores.shape}")
    out_data = _zeros(t, t)
    for i in range(t):
        row = scores.data[i]
        top = max(row[j] for j in range(i + 1))
        exps = [math.exp(row[j] - top) for j in range(i + 1)]
        total = sum(exps)
        for j in range(i + 1):
            out_data[i][j] = exps[j] / total
    out = Tensor(out_data, (scores,))

    def _backward() -> None:
        for i in range(t):
            g_row = out.grad[i]
            p_row = out_data[i]
            dot = 0.0
            for j in range(i + 1):
                dot += g_row[j] * p_row[j]
            sg_row = scores.grad[i]
            for j in range(i + 1):
                sg_row[j] += p_row[j] * (g_row[j] - dot)

    out._backward = _backward
    return out


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, *, eps: float = 1e-5) -> Tensor:
    """Row-wise layer normalization with learned scale ``gamma`` and shift ``beta``."""

    m, n = x.shape
    if gamma.shape != (1, n) or beta.shape != (1, n):
        raise ValueError("gamma and beta must have shape (1, n)")
    out_data = _zeros(m, n)
    norm = _zeros(m, n)
    inv_std = [0.0] * m
    for i in range(m):
        row = x.data[i]
        mean = sum(row) / n
        var = sum((v - mean) ** 2 for v in row) / n
        istd = 1.0 / math.sqrt(var + eps)
        inv_std[i] = istd
        for j in range(n):
            norm[i][j] = (row[j] - mean) * istd
            out_data[i][j] = norm[i][j] * gamma.data[0][j] + beta.data[0][j]
    out = Tensor(out_data, (x, gamma, beta))

    def _backward() -> None:
        for i in range(m):
            g_row = out.grad[i]
            istd = inv_std[i]
            dnorm = [g_row[j] * gamma.data[0][j] for j in range(n)]
            mean_dnorm = sum(dnorm) / n
            mean_dnorm_norm = sum(dnorm[j] * norm[i][j] for j in range(n)) / n
            xg_row = x.grad[i]
            for j in range(n):
                xg_row[j] += istd * (dnorm[j] - mean_dnorm - norm[i][j] * mean_dnorm_norm)
                gamma.grad[0][j] += g_row[j] * norm[i][j]
                beta.grad[0][j] += g_row[j]

    out._backward = _backward
    return out


def transpose(x: Tensor) -> Tensor:
    """Matrix transpose ``(m, n) -> (n, m)``."""

    m, n = x.shape
    out = Tensor([[x.data[i][j] for i in range(m)] for j in range(n)], (x,))

    def _backward() -> None:
        for j in range(n):
            g_row = out.grad[j]
            for i in range(m):
                x.grad[i][j] += g_row[i]

    out._backward = _backward
    return out


def concat_cols(parts: Sequence[Tensor]) -> Tensor:
    """Concatenate tensors with equal row counts along their columns."""

    rows = parts[0].rows
    if any(p.rows != rows for p in parts):
        raise ValueError("concat_cols requires equal row counts")
    offsets: list[int] = []
    running = 0
    for p in parts:
        offsets.append(running)
        running += p.cols
    out_data = [[0.0] * running for _ in range(rows)]
    for part, off in zip(parts, offsets, strict=True):
        for i in range(rows):
            for j in range(part.cols):
                out_data[i][off + j] = part.data[i][j]
    out = Tensor(out_data, tuple(parts))

    def _backward() -> None:
        for part, off in zip(parts, offsets, strict=True):
            for i in range(rows):
                g_row = out.grad[i]
                pg_row = part.grad[i]
                for j in range(part.cols):
                    pg_row[j] += g_row[off + j]

    out._backward = _backward
    return out


def embedding(table: Tensor, ids: Sequence[int]) -> Tensor:
    """Gather rows ``ids`` from an embedding ``table`` of shape ``(V, D)``."""

    d = table.cols
    out = Tensor([list(table.data[i]) for i in ids], (table,))

    def _backward() -> None:
        for pos, token in enumerate(ids):
            g_row = out.grad[pos]
            tg_row = table.grad[token]
            for j in range(d):
                tg_row[j] += g_row[j]

    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, targets: Sequence[int]) -> Tensor:
    """Mean softmax cross-entropy over rows; returns a ``(1, 1)`` loss tensor."""

    t, v = logits.shape
    if len(targets) != t:
        raise ValueError("targets length must match the number of rows")
    probs = _zeros(t, v)
    total_loss = 0.0
    for i in range(t):
        row = logits.data[i]
        top = max(row)
        exps = [math.exp(x - top) for x in row]
        denom = sum(exps)
        for j in range(v):
            probs[i][j] = exps[j] / denom
        total_loss += -math.log(max(probs[i][targets[i]], 1e-12))
    out = Tensor([[total_loss / t]], (logits,))

    def _backward() -> None:
        scale_g = out.grad[0][0] / t
        for i in range(t):
            lg_row = logits.grad[i]
            tgt = targets[i]
            for j in range(v):
                lg_row[j] += scale_g * (probs[i][j] - (1.0 if j == tgt else 0.0))

    out._backward = _backward
    return out


def softmax_row(values: Sequence[float]) -> list[float]:
    """A plain (non-differentiable) softmax used by sampling at inference time."""

    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]
