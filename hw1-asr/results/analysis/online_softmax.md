

# Flash Attention: Online vs Standard Softmax Evaluation

## Overview

This experiment evaluates the performance impact of **online softmax vs standard (offline) softmax** within a Triton-based Flash Attention implementation.

### Key Setup Details

* **Precision:** `bf16`
* **Generation Function:** `generate_v8` (optimized autoregressive decoding)
* **Backend:** Triton kernels (no PyTorch fallback)
* **Masking:** **Causal masking integrated inside the kernel** (`IS_CAUSAL`)
* **Sequence Length:** 256
* **Model:** GLM-ASR (Audio + Text multimodal)

---

## Implementations Compared

---

## 🔹 1. Standard Softmax (Offline)

### Description

This implementation uses a **two-pass softmax strategy**:

1. Compute **row-wise max**
2. Compute **row-wise sum**
3. Normalize and accumulate values

⚠️ **Key drawback:**

* Requires **multiple passes over K/V**
* Increases **memory bandwidth usage**
* Slower for long sequences and decoding

---

### Triton Kernel (Standard Softmax)

```python
@triton.jit
def fused_standard_softmax_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    seq_q, seq_k, head_dim, scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M:   tl.constexpr,
    BLOCK_N:   tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    """
    Standard (offline) softmax implementation.

    Pass 1 — compute row max
    Pass 2 — compute row sum
    Pass 3 — normalize + accumulate V

    K/V are loaded multiple times.
    """
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + pid_bh * stride_q0
              + offs_m[:, None] * stride_q1
              + offs_d[None, :] * stride_q2,
        mask=(offs_m[:, None] < seq_q) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    # ── Pass 1a: row max ─────────────────────────────
    row_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(
            k_ptr + pid_bh * stride_k0
                  + offs_d[:, None] * stride_k2
                  + offs_n[None, :] * stride_k1,
            mask=(offs_d[:, None] < head_dim) & (offs_n[None, :] < seq_k),
            other=0.0,
        )
        qk = tl.dot(q, k) * scale

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        qk = tl.where(offs_n[None, :] < seq_k, qk, float("-inf"))
        row_max = tl.maximum(row_max, tl.max(qk, axis=1))

    # ── Pass 1b: row sum ─────────────────────────────
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(...)

        qk = tl.dot(q, k) * scale

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        qk = tl.where(offs_n[None, :] < seq_k, qk, float("-inf"))
        row_sum += tl.sum(tl.exp(qk - row_max[:, None]), axis=1)

    # ── Pass 2: normalize + accumulate V ─────────────
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(...)
        v = tl.load(...)

        qk = tl.dot(q, k) * scale

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        qk = tl.where(offs_n[None, :] < seq_k, qk, float("-inf"))

        p = tl.exp(qk - row_max[:, None]) / row_sum[:, None]
        acc += tl.dot(p.to(v.dtype), v)

    tl.store(out_ptr, acc)
```

---

## 🔹 2. Online Softmax (Flash Attention Style)

### Description

This implementation uses **streaming (online) softmax**:

* Maintains:

  * running max (`m_i`)
  * running sum (`l_i`)
* Computes attention in **one pass**
* Avoids reloading K/V

---

### Key Advantages

* ✅ Single-pass computation
* ✅ Reduced memory bandwidth
* ✅ Better GPU utilization
* ✅ Faster decoding

---

## Results

---

## 🔹 Online Softmax

### Performance Summary

| Component          | Time (ms)      | % of Total |
| ------------------ | -------------- | ---------- |
| Audio Encoder      | 4533.44        | 57.9%      |
| Projector          | 23.38          | 0.3%       |
| Decoder Prefill    | 834.83         | 10.7%      |
| Decoder (50 steps) | 2441.81        | 31.2%      |
| **Total**          | **7833.46 ms** |            |

---

## 🔹 Standard Softmax

### Performance Summary

| Component          | Time (ms)       | % of Total |
| ------------------ | --------------- | ---------- |
| Audio Encoder      | 4495.50         | 27.4%      |
| Projector          | 26.68           | 0.2%       |
| Decoder Prefill    | 1448.88         | 8.8%       |
| Decoder (50 steps) | 10451.49        | 63.6%      |
| **Total**          | **16422.55 ms** |            |

---

## Key Observations

### 🚀 1. End-to-End Speedup

* Online: **7833 ms**
* Standard: **16422 ms**

➡️ **~2.1× faster**

---

### ⚡ 2. Decode Phase Improvement

* Standard: **10451 ms**
* Online: **2441 ms**

➡️ **~4× faster decoding**

---

### 💾 3. Memory Efficiency

| Method   | Memory Access      |
| -------- | ------------------ |
| Standard | Multiple K/V loads |
| Online   | Single-pass        |

➡️ Online softmax wins due to **memory bandwidth reduction**

---

### 🧠 4. Kernel-Level Optimization

* Causal masking applied **inside kernel**
* No separate masking kernel
* Better fusion and fewer memory ops

---

## Final Conclusion

### ✅ Online Softmax (Flash Attention) is Superior

* ~2× faster overall
* ~4× faster decoding
* Better suited for autoregressive models

---

## Notes

* `bf16` precision used
* `generate_v8` used for optimized decoding
* Fully Triton backend
* No KV cache (full recompute per step)

---

If you want next level polish, I can:

* Add **diagrams of online vs offline softmax** (very useful for reports)
* Or convert this into a **paper-ready section with figures + equations**
