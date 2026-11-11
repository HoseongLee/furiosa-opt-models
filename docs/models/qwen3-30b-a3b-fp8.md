---
hide: [toc]
---

# Qwen3 30B-A3B FP8

The 128 experts keep the checkpoint's block-scaled E4M3 weights, and each quantizes its input
per group of 128 features at run time. The attention projections are dequantized to BF16 when
the checkpoint loads. Each of the checkpoint's four key and value heads is cached twice, so the
KV cache takes twice the memory of the checkpoint's own layout.

{% include "benchmarks.md" %}
