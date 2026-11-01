"""Real-text character-level training benchmark.

Trains the transformer on a bundled public-domain prose excerpt at the character
level and reports held-out compression in bits per character against a unigram
baseline, plus a sampled continuation. This complements the synthetic copy task
with evidence that the same model learns real natural-language structure.
Deterministic and offline; run with:

    uv run python benchmarks/text_training.py
"""

from __future__ import annotations

import time

from transformer_from_scratch.text import CORPUS, generate_text, train_char_lm


def main() -> None:
    print("Character-level training on a public-domain prose excerpt:")
    print(f"  corpus: {len(CORPUS)} characters, {len(set(CORPUS))} distinct symbols")
    start = time.perf_counter()
    result = train_char_lm(context=24, d_model=32, n_head=4, n_layer=2, d_ff=64, steps=250, seed=0)
    elapsed = time.perf_counter() - start
    reduction = result.baseline_bits_per_char - result.bits_per_char
    print(
        f"  trained in {elapsed:.0f}s; held-out compression "
        f"{result.bits_per_char:.3f} bits/char "
        f"vs unigram baseline {result.baseline_bits_per_char:.3f} "
        f"({reduction:.3f} bits/char better)"
    )
    print(f"  final training loss {result.losses[-1]:.3f} (from {result.losses[0]:.3f})")
    sample = generate_text(result.model, result.vocab, "It is ", 80, seed=1)
    print(f"  sample continuation: {sample!r}")


if __name__ == "__main__":
    main()
