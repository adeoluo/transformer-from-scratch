"""Independent numerical-parity check against PyTorch.

This reconstructs the from-scratch model's exact forward and backward pass in
PyTorch, loads the same weights, and asserts both logits and all parameter
gradients match to a tight tolerance. It is the independent reference
comparison for the hand-written autodiff: the from-scratch engine is validated
against finite differences in ``test_autograd.py`` and against PyTorch's
``autograd`` engine here.

PyTorch is installed via the ``validation`` dependency group (run in CI on
Python 3.12 and 3.13, and locally via ``uv sync --group validation``), keeping
the installed runtime package lightweight (``numpy`` only).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from transformer_from_scratch.model import ModelConfig, Transformer
from transformer_from_scratch.tensor import Tensor

torch: Any = pytest.importorskip("torch")


def _t(matrix: Tensor, *, requires_grad: bool = False) -> Any:
    return torch.tensor(matrix.data.tolist(), dtype=torch.float64, requires_grad=requires_grad)


def _torch_forward(
    model: Transformer,
    token_ids: list[int],
    *,
    param_map: dict[int, Any] | None = None,
) -> Any:
    def p(param: Tensor) -> Any:
        if param_map is not None:
            return param_map[id(param)]
        return _t(param)

    positions = list(range(len(token_ids)))
    token_embed = p(model.token_embed)
    pos_embed = p(model.pos_embed)
    x = token_embed[token_ids] + pos_embed[positions]

    def layer_norm(h: Any, gamma: Tensor, beta: Tensor) -> Any:
        mean = h.mean(dim=-1, keepdim=True)
        var = ((h - mean) ** 2).mean(dim=-1, keepdim=True)
        normed = (h - mean) / torch.sqrt(var + 1e-5)
        return normed * p(gamma) + p(beta)

    seq = len(token_ids)
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.float64) * float("-inf"), diagonal=1)
    for block in model.blocks:
        normed = layer_norm(x, block.ln1.gamma, block.ln1.beta)
        head_outputs = []
        for head in block.attn.heads:
            q = normed @ p(head.w_q)
            k = normed @ p(head.w_k)
            v = normed @ p(head.w_v)
            scores = (q @ k.transpose(0, 1)) * head.scale + mask
            weights = torch.softmax(scores, dim=-1)
            head_outputs.append(weights @ v)
        combined = torch.cat(head_outputs, dim=-1)
        attn = combined @ p(block.attn.w_o) + p(block.attn.b_o)
        x = x + attn

        normed2 = layer_norm(x, block.ln2.gamma, block.ln2.beta)
        hidden = torch.nn.functional.gelu(
            normed2 @ p(block.ff.w1) + p(block.ff.b1), approximate="tanh"
        )
        ff = hidden @ p(block.ff.w2) + p(block.ff.b2)
        x = x + ff

    x = layer_norm(x, model.ln_f.gamma, model.ln_f.beta)
    return x @ token_embed.transpose(0, 1)


def test_logits_match_pytorch_reference() -> None:
    cfg = ModelConfig(vocab_size=11, context=8, d_model=16, n_head=4, n_layer=2, d_ff=32)
    model = Transformer(cfg, seed=0)
    token_ids = [3, 1, 4, 1, 5, 9, 2]

    ours = model.logits(token_ids).data.tolist()
    reference = _torch_forward(model, token_ids).tolist()

    max_abs_diff = max(
        abs(ours[i][j] - reference[i][j]) for i in range(len(ours)) for j in range(len(ours[0]))
    )
    assert max_abs_diff < 1e-9, f"max abs diff {max_abs_diff}"


def test_cross_entropy_matches_pytorch_reference() -> None:
    cfg = ModelConfig(vocab_size=11, context=8, d_model=16, n_head=4, n_layer=2, d_ff=32)
    model = Transformer(cfg, seed=1)
    token_ids = [2, 7, 1, 8, 2, 8]
    targets = [7, 1, 8, 2, 8, 1]

    ours = float(model.loss(token_ids, targets).data[0, 0])
    logits = _torch_forward(model, token_ids)
    reference = torch.nn.functional.cross_entropy(
        logits, torch.tensor(targets), reduction="mean"
    ).item()
    assert math.isclose(ours, reference, rel_tol=1e-9, abs_tol=1e-9)


def test_backward_gradients_match_pytorch_reference() -> None:
    cfg = ModelConfig(vocab_size=13, context=8, d_model=16, n_head=4, n_layer=2, d_ff=32)
    model = Transformer(cfg, seed=2)
    for idx, param in enumerate(model.parameters()):
        if param.rows == 1:
            param.data = param.data + 0.05 * (idx + 1)

    token_ids = [3, 1, 4, 1, 5, 9, 2]
    targets = [1, 4, 1, 5, 9, 2, 6]

    for p in model.parameters():
        p.zero_grad()
    loss = model.loss(token_ids, targets)
    loss.backward()

    param_map = {id(p): _t(p, requires_grad=True) for p in model.parameters()}
    ref_logits = _torch_forward(model, token_ids, param_map=param_map)
    ref_loss = torch.nn.functional.cross_entropy(
        ref_logits, torch.tensor(targets), reduction="mean"
    )
    ref_loss.backward()

    assert len(model.parameters()) == 48
    for p in model.parameters():
        ref_grad = param_map[id(p)].grad
        assert ref_grad is not None
        ours_g = p.grad.tolist()
        ref_g = ref_grad.tolist()
        max_diff = max(
            abs(ours_g[r][c] - ref_g[r][c])
            for r in range(len(ours_g))
            for c in range(len(ours_g[0]))
        )
        assert max_diff < 1e-9, f"gradient mismatch {max_diff} for shape {p.shape}"
