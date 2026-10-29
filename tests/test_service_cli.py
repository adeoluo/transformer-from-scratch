"""Tests for the HTTP service routing, generation, and the CLI."""

from __future__ import annotations

from transformer_from_scratch.cli import main
from transformer_from_scratch.service import generate, handle


def test_health_and_ready(service) -> None:  # type: ignore[no-untyped-def]
    status, payload = handle("GET", "/health", {}, service)
    assert status == 200
    assert payload["vocab_size"] == service.task.vocab_size
    status, payload = handle("GET", "/ready", {}, service)
    assert status == 200 and payload["status"] == "ready"


def test_generate_route_and_helper(service) -> None:  # type: ignore[no-untyped-def]
    status, payload = handle(
        "POST", "/generate", {"prompt": [1, 2, 3, 4, 0], "max_new_tokens": 3}, service
    )
    assert status == 200
    assert payload["generated"][:5] == [1, 2, 3, 4, 0]
    assert len(payload["new_tokens"]) == 3
    assert generate(service.model, [1, 2, 3, 4, 0], 3) == payload["generated"]


def test_generate_validation(service) -> None:  # type: ignore[no-untyped-def]
    for body in (
        {"prompt": []},
        {"prompt": [99]},
        {"prompt": [1, 2], "max_new_tokens": 0},
        {"prompt": [1, 2], "max_new_tokens": 999},
        {"prompt": [True, 2]},
    ):
        status, payload = handle("POST", "/generate", body, service)
        assert status == 400 and "error" in payload


def test_evaluate_route(service) -> None:  # type: ignore[no-untyped-def]
    status, payload = handle("GET", "/evaluate", {}, service)
    assert status == 200
    assert 0.0 <= payload["copy_accuracy"] <= 1.0


def test_attention_route_and_validation(service) -> None:  # type: ignore[no-untyped-def]
    status, payload = handle(
        "POST", "/attention", {"tokens": [1, 2, 3, 4, 0], "layer": 0, "head": 0}, service
    )
    assert status == 200
    assert len(payload["attention"]) == 5
    for body in ({"tokens": [1], "layer": 9}, {"tokens": [1], "head": 9}, {"tokens": [99]}):
        status, _ = handle("POST", "/attention", body, service)
        assert status == 400


def test_unknown_route_and_metrics(service) -> None:  # type: ignore[no-untyped-def]
    status, _ = handle("GET", "/nope", {}, service)
    assert status == 404
    handle("GET", "/evaluate", {}, service)
    assert "service_requests_total" in service.metrics.render_prometheus()


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
