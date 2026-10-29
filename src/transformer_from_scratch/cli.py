"""Command line interface for the transformer."""

from __future__ import annotations

import argparse

from transformer_from_scratch.data import (
    CopyTask,
    test_sequences,
    to_inputs_targets,
    train_sequences,
)
from transformer_from_scratch.evaluation import BigramBaseline, evaluate, evaluate_bigram
from transformer_from_scratch.inspect import attention_pattern, sharpest_head
from transformer_from_scratch.model import ModelConfig
from transformer_from_scratch.service import generate
from transformer_from_scratch.training import TrainConfig, train


def _task(args: argparse.Namespace) -> CopyTask:
    return CopyTask(n_symbols=args.symbols, length=args.length)


def _train(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    task = _task(args)
    model_cfg = ModelConfig(vocab_size=task.vocab_size, context=task.seq_len)
    result = train(
        task, model_cfg, TrainConfig(steps=args.steps, seed=args.seed), log_every=args.steps
    )
    return task, result


def _cmd_train(args: argparse.Namespace) -> None:
    task, result = _train(args)
    test = test_sequences(task, seed=args.seed)
    train_set = train_sequences(task, seed=args.seed)
    baseline = BigramBaseline.fit(task.vocab_size, train_set)
    model_report = evaluate(result.model, task, test)
    bigram_report = evaluate_bigram(baseline, task, test)
    print(
        f"Trained {args.steps} steps on the copy task "
        f"(symbols={args.symbols}, length={args.length})."
    )
    print(f"  transformer: {model_report.format()}")
    print(f"  bigram:      copy_acc={bigram_report.copy_accuracy:.4f}")
    sh = sharpest_head(result.model, test[:20], start=task.length)
    print(
        f"  sharpest attention head: layer {sh.layer} head {sh.head}, "
        f"peak {sh.mean_peak_weight:.3f}"
    )


def _cmd_evaluate(args: argparse.Namespace) -> None:
    task, result = _train(args)
    report = evaluate(result.model, task, test_sequences(task, seed=args.seed))
    print(report.format())


def _cmd_generate(args: argparse.Namespace) -> None:
    task, result = _train(args)
    prompt = test_sequences(task, seed=args.seed)[0][: task.length + 1]
    out = generate(result.model, prompt, task.length)
    print(f"prompt:    {prompt}")
    print(f"generated: {out}")
    print(f"expected:  {test_sequences(task, seed=args.seed)[0]}")


def _cmd_attention(args: argparse.Namespace) -> None:
    task, result = _train(args)
    seq = test_sequences(task, seed=args.seed)[0]
    inputs, _ = to_inputs_targets(seq)
    sh = sharpest_head(result.model, test_sequences(task, seed=args.seed)[:20], start=task.length)
    pattern = attention_pattern(result.model, inputs, layer=sh.layer, head=sh.head)
    print(f"Attention of layer {sh.layer} head {sh.head} on {inputs}:")
    for i, row in enumerate(pattern):
        cells = " ".join(f"{w:.2f}" for w in row[: i + 1])
        print(f"  pos {i:>2} (tok {inputs[i]}): {cells}")


def _cmd_serve(args: argparse.Namespace) -> None:
    from transformer_from_scratch.service import serve

    serve(host=args.host, port=args.port)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbols", type=int, default=4)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="transformer-from-scratch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("train", "train and report accuracy against a bigram baseline"),
        ("evaluate", "train, then report held-out accuracy and perplexity"),
        ("generate", "train, then greedily complete a held-out prompt"),
        ("attention", "train, then print the sharpest head's attention pattern"),
    ):
        p = sub.add_parser(name, help=help_text)
        _add_common(p)

    serve = sub.add_parser("serve", help="start the HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    handlers = {
        "train": _cmd_train,
        "evaluate": _cmd_evaluate,
        "generate": _cmd_generate,
        "attention": _cmd_attention,
        "serve": _cmd_serve,
    }
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
