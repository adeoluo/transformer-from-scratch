"""A small reverse-mode automatic differentiation engine over dense matrices.

Everything the transformer needs is built from a single :class:`Tensor` type whose
operations each record a local backward closure. A topological sort then runs the
closures in reverse to accumulate gradients, which is reverse-mode autodiff. Every
gradient is written out by hand; the arithmetic runs on 2-D numpy arrays, so a matrix
multiply is one BLAS call instead of three nested Python loops.

The operations are deliberately the minimal set a transformer needs: matrix
multiply, biased affine maps, elementwise addition, scaling, GELU, row-wise
causal softmax, row-wise layer normalization, embedding lookup, column
concatenation, and a softmax cross-entropy loss. Each is covered by a finite
difference gradient check in the test suite.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
Matrix = list[list[float]]


class Tensor:
    """A 2-D array of floats with a recorded gradient and backward closure."""

    __slots__ = ("_backward", "_prev", "cols", "data", "grad", "requires_grad")

    def __init__(
        self,
        data: Matrix | Array,
        _children: tuple[Tensor, ...] = (),
        *,
        requires_grad: bool = True,
    ) -> None:
        array = np.asarray(data, dtype=np.float64)
        if array.ndim != 2:
            array = array.reshape(len(array), -1) if array.size else np.zeros((len(array), 0))
        self.data: Array = array
        self.cols = int(array.shape[1])
        self.grad: Array = np.zeros_like(array)
        self.requires_grad = requires_grad
        self._backward: Callable[[], None] = lambda: None
        self._prev: tuple[Tensor, ...] = _children

    @property
    def rows(self) -> int:
        return int(self.data.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)

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
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix product ``a @ b`` with shapes ``(m, k) @ (k, n) -> (m, n)``."""

    if a.cols != b.rows:
        raise ValueError(f"shape mismatch for matmul: {a.shape} @ {b.shape}")
    out = Tensor(a.data @ b.data, (a, b))

    def _backward() -> None:
        a.grad += out.grad @ b.data.T
        b.grad += a.data.T @ out.grad

    out._backward = _backward
    return out


def add_bias(x: Tensor, bias: Tensor) -> Tensor:
    """Add a row-vector ``bias`` of shape ``(1, n)`` to every row of ``x``."""

    n = x.cols
    if bias.shape != (1, n):
        raise ValueError(f"bias must be (1, {n}), got {bias.shape}")
    out = Tensor(x.data + bias.data, (x, bias))

    def _backward() -> None:
        x.grad += out.grad
        bias.grad += out.grad.sum(axis=0, keepdims=True)

    out._backward = _backward
    return out


def add(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise sum of two equally shaped tensors (used for residuals)."""

    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for add: {a.shape} vs {b.shape}")
    out = Tensor(a.data + b.data, (a, b))

    def _backward() -> None:
        a.grad += out.grad
        b.grad += out.grad

    out._backward = _backward
    return out


def scale(x: Tensor, factor: float) -> Tensor:
    """Multiply every element by a constant ``factor``."""

    out = Tensor(x.data * factor, (x,))

    def _backward() -> None:
        x.grad += out.grad * factor

    out._backward = _backward
    return out


_GELU_C = math.sqrt(2.0 / math.pi)


def gelu(x: Tensor) -> Tensor:
    """The tanh approximation of the Gaussian error linear unit, elementwise."""

    v = x.data
    tanh_inner = np.tanh(_GELU_C * (v + 0.044715 * v * v * v))
    out = Tensor(0.5 * v * (1.0 + tanh_inner), (x,))

    def _backward() -> None:
        sech2 = 1.0 - tanh_inner * tanh_inner
        dinner = _GELU_C * (1.0 + 3.0 * 0.044715 * v * v)
        deriv = 0.5 * (1.0 + tanh_inner) + 0.5 * v * sech2 * dinner
        x.grad += out.grad * deriv

    out._backward = _backward
    return out


def causal_softmax(scores: Tensor) -> Tensor:
    """Row-wise softmax where row ``i`` attends only to columns ``j <= i``.

    ``scores`` is square ``(T, T)``. Masked positions receive zero weight and no
    gradient, implementing autoregressive attention.
    """

    t, t2 = scores.shape
    if t != t2:
        raise ValueError(f"causal_softmax expects a square matrix, got {scores.shape}")
    allowed = np.tril(np.ones((t, t), dtype=bool))
    masked = np.where(allowed, scores.data, -np.inf)
    exps = np.exp(masked - masked.max(axis=1, keepdims=True))
    probs = exps / exps.sum(axis=1, keepdims=True)
    out = Tensor(probs, (scores,))

    def _backward() -> None:
        # Masked probabilities are zero, so they add nothing to the dot product and
        # receive no gradient.
        dot = (out.grad * probs).sum(axis=1, keepdims=True)
        scores.grad += probs * (out.grad - dot)

    out._backward = _backward
    return out


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, *, eps: float = 1e-5) -> Tensor:
    """Row-wise layer normalization with learned scale ``gamma`` and shift ``beta``."""

    n = x.cols
    if gamma.shape != (1, n) or beta.shape != (1, n):
        raise ValueError("gamma and beta must have shape (1, n)")
    mean = x.data.mean(axis=1, keepdims=True)
    var = ((x.data - mean) ** 2).mean(axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    norm = (x.data - mean) * inv_std
    out = Tensor(norm * gamma.data + beta.data, (x, gamma, beta))

    def _backward() -> None:
        g = out.grad
        dnorm = g * gamma.data
        mean_dnorm = dnorm.mean(axis=1, keepdims=True)
        mean_dnorm_norm = (dnorm * norm).mean(axis=1, keepdims=True)
        x.grad += inv_std * (dnorm - mean_dnorm - norm * mean_dnorm_norm)
        gamma.grad += (g * norm).sum(axis=0, keepdims=True)
        beta.grad += g.sum(axis=0, keepdims=True)

    out._backward = _backward
    return out


def transpose(x: Tensor) -> Tensor:
    """Matrix transpose ``(m, n) -> (n, m)``."""

    out = Tensor(x.data.T.copy(), (x,))

    def _backward() -> None:
        x.grad += out.grad.T

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
    out = Tensor(np.concatenate([p.data for p in parts], axis=1), tuple(parts))

    def _backward() -> None:
        for part, off in zip(parts, offsets, strict=True):
            part.grad += out.grad[:, off : off + part.cols]

    out._backward = _backward
    return out


def embedding(table: Tensor, ids: Sequence[int]) -> Tensor:
    """Gather rows ``ids`` from an embedding ``table`` of shape ``(V, D)``."""

    index = np.asarray(ids, dtype=np.intp)
    out = Tensor(table.data[index], (table,))

    def _backward() -> None:
        np.add.at(table.grad, index, out.grad)  # repeated ids accumulate

    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, targets: Sequence[int]) -> Tensor:
    """Mean softmax cross-entropy over rows; returns a ``(1, 1)`` loss tensor."""

    t, _v = logits.shape
    if len(targets) != t:
        raise ValueError("targets length must match the number of rows")
    target_index = np.asarray(targets, dtype=np.intp)
    exps = np.exp(logits.data - logits.data.max(axis=1, keepdims=True))
    probs = exps / exps.sum(axis=1, keepdims=True)
    picked = probs[np.arange(t), target_index]
    total_loss = float(-np.log(np.maximum(picked, 1e-12)).sum())
    out = Tensor([[total_loss / t]], (logits,))

    def _backward() -> None:
        grad = probs.copy()
        grad[np.arange(t), target_index] -= 1.0
        logits.grad += (out.grad[0, 0] / t) * grad

    out._backward = _backward
    return out


def softmax_row(values: Sequence[float]) -> list[float]:
    """A plain (non-differentiable) softmax used by sampling at inference time."""

    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]
