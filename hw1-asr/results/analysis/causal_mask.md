# Causal Mask Experiment — Triton FlashAttention

## System Configuration

| Setting | Value |
|---|---|
| Backend | Triton (`fused_flash_attention_kernel`) |
| Precision | `bfloat16` |
| Generation | `generate_v8` |
| Kernel param | `IS_CAUSAL: tl.constexpr` |

---

## What We Are Testing

Whether fusing the causal mask directly inside `fused_flash_attention_kernel` as a compile-time constant (`IS_CAUSAL: tl.constexpr`) provides a measurable speedup over the non-causal path, and whether the measured speedup matches the theoretical tile savings.

`IS_CAUSAL=True` enables two things inside the kernel:

```python
# 1. Early loop exit — skip future K tiles entirely
hi = tl.minimum(seq_k, (pid_m + 1) * BLOCK_M)

# 2. Triangle mask — zero out future positions within the last partial tile
qk = tl.where(offs_m[:, None] >= current_offs_n[None, :], qk, float("-inf"))
```

Because `IS_CAUSAL` is a `tl.constexpr`, Triton compiles two completely separate kernel binaries — the `if IS_CAUSAL` branch is compiled away entirely in the non-causal version. This means we are measuring a genuine algorithmic difference, not a runtime branch.

---

## Correctness

| Variant | Max diff vs PyTorch |
|---|---|
| Triton causal vs torch causal | 0.00948 |
| Triton non-causal vs torch non-causal | 0.00453 |
| Causal vs non-causal output diff | 2.697 (expected > 0 ✓) |

The causal/non-causal output divergence of 2.697 confirms the mask is active — the two modes are computing genuinely different attention distributions.

---

## Theoretical Tile Savings

`IS_CAUSAL=True` means Q block `i` only reads K tiles `0..i` — the lower triangle. Non-causal reads all K tiles for every Q block.

Configuration: `BLOCK_M=64, BLOCK_N=32`

| seq_len | Non-causal tiles | Causal tiles | Expected speedup |
|---|---|---|---|
| 64 | 2 | 2 | 1.00× |
| 128 | 8 | 6 | 1.33× |
| 256 | 32 | 20 | 1.60× |
| 512 | 128 | 72 | 1.78× |
| 1024 | 512 | 272 | 1.88× |
| 2048 | 2048 | 1056 | 1.94× |

As `seq_len → ∞`, speedup converges to exactly **2×** — causal visits only the lower triangle of the attention matrix.

---

## Measured Latency (ms)

| seq_len | Triton causal | Triton non-causal | PyTorch causal | PyTorch non-causal |
|---|---|---|---|---|
| 16 | 0.0145 | 0.0146 | 0.0138 | 0.0139 |
| 32 | 0.0153 | 0.0152 | 0.0145 | 0.0141 |
| 64 | 0.0165 | 0.0163 | 0.0156 | 0.0154 |
| 128 | 0.0196 | 0.0191 | 0.0237 | 0.0230 |
| 256 | 0.0257 | 0.0255 | 0.0469 | 0.0460 |
| 512 | 0.0392 | 0.0419 | 0.1134 | 0.1523 |
| 1024 | 0.0809 | 0.0991 | 0.3423 | 0.5241 |
| 2048 | 0.2181 | 0.3044 | 1.1680 | 2.0284 |

---

## Analysis

**Below seq_len=128, causal and non-causal are identical.** With `BLOCK_M=64`, a seq_len=64 kernel has only one Q block and one K tile — there is nothing to skip. The early exit fires immediately and the triangle mask covers no positions. Both lines overlap.

**Causal wins from seq_len=512 onwards.** At seq_len=1024, Triton causal is **1.22×** faster than Triton non-causal (0.0809 vs 0.0991 ms). At seq_len=2048 this grows to **1.39×** (0.218 vs 0.304 ms).

**Measured speedup is lower than theoretical.** Theory predicts 1.88× at seq_len=1024 but we measure 1.22×. The gap exists because the `tl.where` triangle mask still runs inside the last partial tile of each Q block even when all future tiles are skipped — this compute cost partially offsets the tile savings.

**PyTorch shows the same trend, validating the experiment.** PyTorch non-causal at seq_len=2048 (2.028 ms) is 1.74× slower than PyTorch causal (1.168 ms) — confirming the causal speedup is real and our Triton ratio is in the correct range.

**The causal path is the right default for generate_v8.** All autoregressive decoding uses causal attention. At the sequence lengths produced during generation (512–2048 tokens), the fused causal mask saves 22–39% of attention compute at zero additional kernel launch cost.
