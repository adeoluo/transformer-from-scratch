"""Independent numerical-parity check against PyTorch.

This reconstructs the from-scratch model's exact forward pass in PyTorch, loads
the same weights, and asserts the logits match to a tight tolerance. It is the
independent reference comparison for the hand-written autodiff: the from-scratch
engine is validated against finite differences in ``test_autograd.py`` and
against a mature framework here.

PyTorch is an optional ``validation`` extra, not a runtime dependency, so this
test is skipped when torch is not installed and the core library stays zero
dependency.
"""

from __future__ import annotations

import math

import pytest

from transformer_from_scratch.model import ModelConfig, Transformer

torch = pytest.importorskip("torch")


def _t(matrix: list[list[float]]) -> torch.Tensor:
    return torch.tensor(matrix, dtype=torch.float64)


def _torch_forward(model: Transformer, token_ids: list[int]) -> torch.Tensor:
    positions = list(range(len(token_ids)))
    token_embed = _t(model.token_embed.data)
    pos_embed = _t(model.pos_embed.data)
    x = token_embed[token_ids] + pos_embed[positions]

    def layer_norm(
        h: torch.Tensor, gamma: list[list[float]], beta: list[list[float]]
    ) -> torch.Tensor:
        mean = h.mean(dim=-1, keepdim=True)
        var = ((h - mean) ** 2).mean(dim=-1, keepdim=True)
        normed = (h - mean) / torch.sqrt(var + 1e-5)
        return normed * _t(gamma) + _t(beta)

    seq = len(token_ids)
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.float64) * float("-inf"), diagonal=1)
    for block in model.blocks:
        normed = layer_norm(x, block.ln1.gamma.data, block.ln1.beta.data)
        head_outputs = []
        for head in block.attn.heads:
            q = normed @ _t(head.w_q.data)
            k = normed @ _t(head.w_k.data)
            v = normed @ _t(head.w_v.data)
            scores = (q @ k.transpose(0, 1)) * head.scale + mask
            weights = torch.softmax(scores, dim=-1)
            head_outputs.append(weights @ v)
        combined = torch.cat(head_outputs, dim=-1)
        attn = combined @ _t(block.attn.w_o.data) + _t(block.attn.b_o.data)
        x = x + attn

        normed2 = layer_norm(x, block.ln2.gamma.data, block.ln2.beta.data)
        hidden = torch.nn.functional.gelu(
            normed2 @ _t(block.ff.w1.data) + _t(block.ff.b1.data), approximate="tanh"
        )
        ff = hidden @ _t(block.ff.w2.data) + _t(block.ff.b2.data)
        x = x + ff

    x = layer_norm(x, model.ln_f.gamma.data, model.ln_f.beta.data)
    return x @ token_embed.transpose(0, 1)


def test_logits_match_pytorch_reference() -> None:
    cfg = ModelConfig(vocab_size=11, context=8, d_model=16, n_head=4, n_layer=2, d_ff=32)
    model = Transformer(cfg, seed=0)
    token_ids = [3, 1, 4, 1, 5, 9, 2]

    ours = model.logits(token_ids).data
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

    ours = model.loss(token_ids, targets).data[0][0]
    logits = _torch_forward(model, token_ids)
    reference = torch.nn.functional.cross_entropy(
        logits, torch.tensor(targets), reduction="mean"
    ).item()
    assert math.isclose(ours, reference, rel_tol=1e-9, abs_tol=1e-9)
