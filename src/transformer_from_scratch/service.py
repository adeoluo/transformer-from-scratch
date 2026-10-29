"""A dependency-free HTTP service exposing the trained transformer.

The service trains a small transformer once at startup, then answers generation,
evaluation, and attention-inspection requests. The routing logic is a pure
function tested without a socket.

Endpoints
---------
``GET  /health``     liveness and model shape.
``GET  /ready``      readiness once the model is trained.
``GET  /metrics``    per-route request, error, and latency metrics (Prometheus text).
``POST /generate``   body ``{"prompt": [int], "max_new_tokens": int?}``; greedy continuation.
``GET  /evaluate``   copy-region accuracy, overall accuracy, and perplexity on held-out data.
``POST /attention``  body ``{"tokens": [int], "layer": int?, "head": int?}``; attention pattern.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from transformer_from_scratch.data import CopyTask, test_sequences
from transformer_from_scratch.evaluation import EvalReport, evaluate
from transformer_from_scratch.inspect import attention_pattern
from transformer_from_scratch.model import Transformer
from transformer_from_scratch.observability import ServiceMetrics
from transformer_from_scratch.training import TrainConfig, train

JsonObject = dict[str, Any]
_MAX_NEW_TOKENS = 32


def generate(model: Transformer, prompt: list[int], max_new_tokens: int) -> list[int]:
    """Greedily extend ``prompt`` by up to ``max_new_tokens`` tokens."""

    tokens = list(prompt)
    for _ in range(max_new_tokens):
        if len(tokens) >= model.cfg.context:
            break
        logits = model.logits(tokens)
        last = logits.data[-1]
        tokens.append(max(range(len(last)), key=lambda j: last[j]))
    return tokens


@dataclass(frozen=True)
class TransformerService:
    """Holds the trained model and answers generation and evaluation requests."""

    model: Transformer
    task: CopyTask
    test_set: list[list[int]]
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def trained(
        cls, *, task: CopyTask | None = None, train_cfg: TrainConfig | None = None
    ) -> TransformerService:
        task = task or CopyTask()
        result = train(task, train_cfg=train_cfg or TrainConfig())
        return cls(model=result.model, task=task, test_set=test_sequences(task, seed=0))

    def evaluate(self) -> EvalReport:
        return evaluate(self.model, self.task, self.test_set)

    def generate(self, prompt: list[int], max_new_tokens: int) -> JsonObject:
        tokens = generate(self.model, prompt, max_new_tokens)
        return {"prompt": prompt, "generated": tokens, "new_tokens": tokens[len(prompt) :]}

    def attention(self, tokens: list[int], layer: int, head: int) -> JsonObject:
        pattern = attention_pattern(self.model, tokens, layer=layer, head=head)
        return {"tokens": tokens, "layer": layer, "head": head, "attention": pattern}


def _valid_tokens(value: Any, vocab: int, limit: int) -> list[int] | str:
    if not isinstance(value, list) or not value:
        return "must be a non-empty list of integers"
    if len(value) > limit:
        return f"exceeds {limit} tokens"
    out: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0 or item >= vocab:
            return f"token ids must be integers in 0..{vocab - 1}"
        out.append(item)
    return out


def _route(
    method: str, path: str, body: Mapping[str, Any], service: TransformerService
) -> tuple[int, JsonObject]:
    cfg = service.model.cfg
    if method == "GET" and path == "/health":
        return 200, {
            "status": "ok",
            "vocab_size": cfg.vocab_size,
            "context": cfg.context,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
        }
    if method == "GET" and path == "/evaluate":
        report = service.evaluate()
        return 200, {
            "copy_accuracy": round(report.copy_accuracy, 6),
            "overall_accuracy": round(report.overall_accuracy, 6),
            "perplexity": round(report.perplexity, 6),
        }
    if method == "POST" and path == "/generate":
        prompt = _valid_tokens(body.get("prompt"), cfg.vocab_size, cfg.context)
        if isinstance(prompt, str):
            return 400, {"error": f"'prompt' {prompt}"}
        max_new = body.get("max_new_tokens", 8)
        if (
            not isinstance(max_new, int)
            or isinstance(max_new, bool)
            or not (1 <= max_new <= _MAX_NEW_TOKENS)
        ):
            return 400, {"error": f"'max_new_tokens' must be an integer in 1..{_MAX_NEW_TOKENS}"}
        return 200, service.generate(prompt, max_new)
    if method == "POST" and path == "/attention":
        tokens = _valid_tokens(body.get("tokens"), cfg.vocab_size, cfg.context)
        if isinstance(tokens, str):
            return 400, {"error": f"'tokens' {tokens}"}
        layer = body.get("layer", 0)
        head = body.get("head", 0)
        if not isinstance(layer, int) or isinstance(layer, bool) or not (0 <= layer < cfg.n_layer):
            return 400, {"error": f"'layer' must be an integer in 0..{cfg.n_layer - 1}"}
        if not isinstance(head, int) or isinstance(head, bool) or not (0 <= head < cfg.n_head):
            return 400, {"error": f"'head' must be an integer in 0..{cfg.n_head - 1}"}
        return 200, service.attention(tokens, layer, head)
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: TransformerService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    if method == "GET" and path == "/ready":
        return 200, {"status": "ready"}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: TransformerService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": TransformerService.trained()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"transformer-from-scratch service listening on http://{host}:{port}")
    server.serve_forever()
