"""Tests for greedy token generation and the CLI subcommands."""

from __future__ import annotations

from transformer_from_scratch.cli import main
from transformer_from_scratch.training import TrainResult


def test_model_generate(trained: TrainResult) -> None:
    generated = trained.model.generate([1, 2, 3, 4, 0], 3)
    assert generated[:5] == [1, 2, 3, 4, 0]
    assert len(generated) == 8


def test_cli_commands(capsys) -> None:  # type: ignore[no-untyped-def]
    common = ["--symbols", "4", "--length", "4", "--steps", "6"]
    main(["train", *common])
    main(["evaluate", *common])
    main(["generate", *common])
    main(["attention", *common])
    out = capsys.readouterr().out
    assert "transformer:" in out
    assert "generated:" in out
    assert "Attention of layer" in out
