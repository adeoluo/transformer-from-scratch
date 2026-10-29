"""Finite-difference gradient checks for the autodiff engine."""

from __future__ import annotations

from transformer_from_scratch.data import CopyTask
from transformer_from_scratch.model import ModelConfig, Transformer
from transformer_from_scratch.tensor import Tensor


def _numeric_grad(loss_fn, tensor: Tensor, eps: float = 1e-5) -> list[list[float]]:  # type: ignore[no-untyped-def]
    grad = [[0.0] * tensor.cols for _ in range(tensor.rows)]
    for i in range(tensor.rows):
        for j in range(tensor.cols):
            original = tensor.data[i][j]
            tensor.data[i][j] = original + eps
            plus = loss_fn().data[0][0]
            tensor.data[i][j] = original - eps
            minus = loss_fn().data[0][0]
            tensor.data[i][j] = original
            grad[i][j] = (plus - minus) / (2 * eps)
    return grad


def _max_diff(a: list[list[float]], b: list[list[float]]) -> float:
    return max(abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0])))


def test_gradients_match_finite_differences_across_all_parameter_types() -> None:
    cfg = ModelConfig(vocab_size=5, context=6, d_model=12, n_head=2, n_layer=2, d_ff=16)
    model = Transformer(cfg, seed=1)
    ids = [1, 2, 0, 3, 4, 2]
    targets = [2, 0, 3, 4, 2, 1]

    def loss_fn() -> Tensor:
        for p in model.parameters():
            p.zero_grad()
        return model.loss(ids, targets)

    loss = loss_fn()
    loss.backward()

    checked = {
        "token_embed": model.token_embed,
        "pos_embed": model.pos_embed,
        "w_q": model.blocks[0].attn.heads[0].w_q,
        "w_o": model.blocks[0].attn.w_o,
        "b_o": model.blocks[0].attn.b_o,
        "ff_w1": model.blocks[0].ff.w1,
        "ff_b1": model.blocks[0].ff.b1,
        "ff_w2": model.blocks[1].ff.w2,
        "ln1_gamma": model.blocks[0].ln1.gamma,
        "ln2_beta": model.blocks[1].ln2.beta,
        "ln_f_gamma": model.ln_f.gamma,
    }
    for _name, param in checked.items():
        analytic = [row[:] for row in param.grad]
        numeric = _numeric_grad(lambda: model.loss(ids, targets), param)
        assert _max_diff(analytic, numeric) < 1e-4


def test_backward_accumulates_into_shared_weight_tied_embedding() -> None:
    cfg = ModelConfig(vocab_size=4, context=4, d_model=8, n_head=2, n_layer=1, d_ff=8)
    model = Transformer(cfg, seed=2)
    for p in model.parameters():
        p.zero_grad()
    model.loss([1, 2, 3], [2, 3, 1]).backward()
    # The token embedding is used both as input and output projection; its grad is nonzero.
    assert any(any(abs(v) > 0 for v in row) for row in model.token_embed.grad)


def test_random_restart_changes_initial_loss() -> None:
    task = CopyTask()
    cfg = ModelConfig(vocab_size=task.vocab_size, context=task.seq_len)
    a = Transformer(cfg, seed=0).loss([1, 2, 0, 1, 2], [2, 0, 1, 2, 1]).data[0][0]
    b = Transformer(cfg, seed=1).loss([1, 2, 0, 1, 2], [2, 0, 1, 2, 1]).data[0][0]
    assert a != b
