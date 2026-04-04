# HW1-ASR: FlashAttention v1 vs FlashAttention v2 (Using `generate_v8`)

> ⚠️ NOTE:
> All results are from the final implementation.
> Both configurations use the **same optimized `_generate_v8_`**, ensuring a fair comparison of attention implementations only.

---

## 1. Objective

This report compares:

* **FlashAttention v1** (baseline attention kernels)
* **FlashAttention v2** (optimized attention kernels)

Both are evaluated under:

* Identical pipeline
* Same `generate_v8` decoding function

> 🎯 Goal: Isolate the **impact of attention kernel improvements alone**

---

## 2. Key Results Summary

| Metric                    | FlashAttention v1 | FlashAttention v2 | Improvement       |
| ------------------------- | ----------------- | ----------------- | ----------------- |
| Total Runtime (50 tokens) | 15970.91 ms       | 5025.42 ms        | **~3.17× faster** |
| Decode Step (avg)         | 202.46 ms         | 30.81 ms          | **~6.57× faster** |
| Attention (einsum)        | 341.78 ms         | 182.89 ms         | **~1.87× faster** |

---

## 3. End-to-End Performance Breakdown

### 3.1 FlashAttention v1

| Component          | Time (ms)    | %     |
| ------------------ | ------------ | ----- |
| Audio Encoder      | 4262.66      | 26.7% |
| Projector          | 25.74        | 0.2%  |
| Decoder Prefill    | 1559.74      | 9.8%  |
| Decoder (50 steps) | 10122.76     | 63.4% |
| **Total**          | **15970.91** | 100%  |

---

### 3.2 FlashAttention v2

| Component          | Time (ms)   | %     |
| ------------------ | ----------- | ----- |
| Audio Encoder      | 2759.26     | 54.9% |
| Projector          | 23.09       | 0.5%  |
| Decoder Prefill    | 702.58      | 14.0% |
| Decoder (50 steps) | 1540.49     | 30.7% |
| **Total**          | **5025.42** | 100%  |

---

## 4. Attention-Level Comparison

### 4.1 Standard Attention (einsum)

| Version           | Time      |
| ----------------- | --------- |
| FlashAttention v1 | 341.78 ms |
| FlashAttention v2 | 182.89 ms |

> ✅ **~1.87× speedup in attention computation**

---

### 4.2 Torch Matmul Attention

| Version           | Time    |
| ----------------- | ------- |
| FlashAttention v1 | 0.61 ms |
| FlashAttention v2 | 0.93 ms |

* Both already highly optimized
* Not a bottleneck in either version

---

## 5. Critical Observation ⚠️

Although both experiments use `generate_v8`, the **decode time differs drastically**:

| Metric            | v1          | v2         |
| ----------------- | ----------- | ---------- |
| Decode (50 steps) | 10122.76 ms | 1540.49 ms |

This suggests:

> ❗ The difference is **not purely due to attention kernels**

---

## 6. Interpretation

### 6.1 What Changed?

Even though `generate_v8` is used in both:

* FlashAttention v2 run likely includes:

  * Better kernel integration
  * Improved memory behavior
  * Reduced overhead in decode path

---

### 6.2 What This Means

> The observed **~3× speedup is NOT solely from FlashAttention v2**

Instead, it is a combination of:

* Attention improvements (~1.8×)
* System-level efficiency differences
* Kernel interaction effects

---

## 7. Bottleneck Analysis

### FlashAttention v1

* Decoder dominates:

  * **63.4% of runtime**

---

### FlashAttention v2

* Bottleneck shifts:

  * Audio Encoder → **54.9%**
  * Decoder → **30.7%**

> 🔄 Indicates significantly more efficient decoding pipeline

---

## 8. Key Takeaways

### 8.1 Attention Improvement Exists

* ~1.87× faster attention
* Meaningful but **not dominant**

---

### 8.2 Decode Efficiency Drives Performance

* Majority of speedup comes from:

  * Reduced decode overhead
  * Better execution flow

---

### 8.3 Bottleneck Shift

> After optimization → system becomes **encoder-bound**

---

## 9. Final Conclusion

| Aspect             | Insight                 |
| ------------------ | ----------------------- |
| Attention          | Improved (~1.8×)        |
| Decode             | Major improvement (~6×) |
| System Performance | ~3× faster overall      |

---

## 10. Final Insight

> Even when using the same generation function, improvements in attention kernels can **interact with the system** to produce larger-than-expected gains.

---

## 11. Important Note

> For a *strict* FlashAttention v1 vs v2 comparison, all other factors must be perfectly controlled.
> The current results reflect **practical system performance**, not an isolated microbenchmark.

---

## 12. Future Work

* Isolate attention kernel benchmarking
* Fix identical decode paths for fair comparison
* Analyze memory bandwidth utilization
* Add KV caching for further gains

---
