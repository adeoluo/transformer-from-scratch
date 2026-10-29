# Threat Model - Transformer From Scratch

## Assets
- The HTTP service's availability and the integrity of its responses.

## Trust boundaries
- The only external input is the JSON body and path of a request. The model, data, and training
  are in-process and trusted.

## Entry points and mitigations

### `POST /generate` and `POST /attention`
- **Malformed JSON** is rejected with 400 before routing.
- **Validation:** token ids must be integers within the vocabulary and within the context
  length; `max_new_tokens`, `layer`, and `head` are range-checked; booleans are rejected
  explicitly (a Python `bool` is an `int` subclass). Invalid input returns 400 with a message,
  never a stack trace.
- **Bounded work:** generation is capped by `max_new_tokens` and the context length, so a single
  request cannot force unbounded computation.

### `GET` routes
- `/health`, `/ready`, `/evaluate`, and `/metrics` take no user input and return fixed-shape
  payloads. `/metrics` exposes only aggregate counters.

## Non-threats (by construction)
- **No secrets:** the service holds no credentials and makes no outbound calls.
- **No persistence:** state is the in-memory trained model; nothing is written on the request path.
- **No code execution:** inputs are integers; no user-supplied string is evaluated.

## Supply chain
- Zero runtime dependencies. Development tooling is pinned via `uv.lock`. CodeQL and dependency
  scanning run in CI.

## Residual risk
- Generation is CPU-intensive; a flood of `/generate` requests could exhaust CPU. A production
  deployment would add a rate limiter and run behind a reverse proxy.
