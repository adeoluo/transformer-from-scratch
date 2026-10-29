"""Tests for tensor ops, the model shapes, data, training, and evaluation."""

from __future__ import annotations

from transformer_from_scratch import tensor as T
from transformer_from_scratch.data import CopyTask, to_inputs_targets, train_sequences
from transformer_from_scratch.data import test_sequences as held_out_sequences
from transformer_from_scratch.evaluation import BigramBaseline, evaluate, evaluate_bigram
from transformer_from_scratch.inspect import attention_pattern, attention_sharpness, sharpest_head
from transformer_from_scratch.model import ModelConfig, Transformer
from transformer_from_scratch.tensor import Tensor
from transformer_from_scratch.training import TrainConfig, train


def test_matmul_values() -> None:
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    out = T.matmul(a, b)
    assert out.data == [[19.0, 22.0], [43.0, 50.0]]


def test_causal_softmax_is_lower_triangular_and_normalized() -> None:
    scores = Tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 1.0, 2.0]])
    out = T.causal_softmax(scores)
    assert out.data[0][1] == 0.0 and out.data[0][2] == 0.0
    assert out.data[1][2] == 0.0
    for i in range(3):
        assert abs(sum(out.data[i][: i + 1]) - 1.0) < 1e-9


def test_layer_norm_zero_mean_unit_scale() -> None:
    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    gamma = Tensor([[1.0, 1.0, 1.0, 1.0]])
    beta = Tensor([[0.0, 0.0, 0.0, 0.0]])
    out = T.layer_norm(x, gamma, beta)
    mean = sum(out.data[0]) / 4
    assert abs(mean) < 1e-6


def test_concat_cols_shapes() -> None:
    a = Tensor([[1.0], [2.0]])
    b = Tensor([[3.0, 4.0], [5.0, 6.0]])
    out = T.concat_cols([a, b])
    assert out.shape == (2, 3)
    assert out.data == [[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]]


def test_model_logits_shape() -> None:
    task = CopyTask()
    model = Transformer(ModelConfig(vocab_size=task.vocab_size, context=task.seq_len), seed=0)
    logits = model.logits([1, 2, 0, 1, 2])
    assert logits.shape == (5, task.vocab_size)


def test_config_rejects_bad_head_division() -> None:
    import pytest

    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=5, context=5, d_model=10, n_head=3)


def test_data_splits_disjoint_and_shaped() -> None:
    task = CopyTask(n_symbols=4, length=5)
    train_set = train_sequences(task, count=40, seed=0)
    test_set = held_out_sequences(task, count=40, seed=0)
    assert all(len(s) == task.seq_len for s in train_set)
    train_keys = {tuple(s) for s in train_set}
    test_keys = {tuple(s) for s in test_set}
    assert train_keys.isdisjoint(test_keys)
    # second half repeats the first half
    for s in train_set:
        assert s[: task.length] == s[task.length + 1 :]


def test_inputs_targets_shift() -> None:
    inp, tgt = to_inputs_targets([1, 2, 3, 4])
    assert inp == [1, 2, 3]
    assert tgt == [2, 3, 4]


def test_training_learns_to_copy_above_bigram(trained, small_task) -> None:  # type: ignore[no-untyped-def]
    test_set = held_out_sequences(small_task, count=20, seed=0)
    model_report = evaluate(trained.model, small_task, test_set)
    baseline = BigramBaseline.fit(small_task.vocab_size, train_sequences(small_task, seed=0))
    bigram_report = evaluate_bigram(baseline, small_task, test_set)
    assert model_report.copy_accuracy > 0.7
    assert model_report.copy_accuracy > bigram_report.copy_accuracy + 0.3
    assert trained.losses[0][1] > trained.losses[-1][1]


def test_training_is_deterministic() -> None:
    task = CopyTask(n_symbols=4, length=4)
    cfg = ModelConfig(vocab_size=task.vocab_size, context=task.seq_len, d_model=12, n_head=2)
    a = train(task, cfg, TrainConfig(steps=8, batch_size=6, seed=0))
    b = train(task, cfg, TrainConfig(steps=8, batch_size=6, seed=0))
    assert a.model.token_embed.data == b.model.token_embed.data


def test_attention_sharpness_increases_with_training(trained, small_task, small_cfg) -> None:  # type: ignore[no-untyped-def]
    seqs = held_out_sequences(small_task, count=10, seed=0)
    untrained = Transformer(small_cfg, seed=0)
    u = sharpest_head(untrained, seqs, start=small_task.length).mean_peak_weight
    t = sharpest_head(trained.model, seqs, start=small_task.length).mean_peak_weight
    assert t > u + 0.2


def test_attention_pattern_rows_are_causal_distributions() -> None:
    task = CopyTask(n_symbols=4, length=4)
    model = Transformer(ModelConfig(vocab_size=task.vocab_size, context=task.seq_len), seed=0)
    tokens = held_out_sequences(task, count=10, seed=0)[0][:-1]
    pattern = attention_pattern(model, tokens, layer=0, head=0)
    for i, row in enumerate(pattern):
        assert abs(sum(row[: i + 1]) - 1.0) < 1e-9
        assert all(row[j] == 0.0 for j in range(i + 1, len(tokens)))
    summary = attention_sharpness(model, tokens, layer=0, head=0, start=task.length)
    assert 0.0 <= summary.mean_peak_weight <= 1.0
