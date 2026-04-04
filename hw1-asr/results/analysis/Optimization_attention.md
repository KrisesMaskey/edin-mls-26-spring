# HW1-ASR Optimization Report


---

## 1. Baseline Reproduction (Template Implementation)

### Objective

The goal of this experiment was to **verify the correctness and performance of the unmodified implementation** by running the ASR pipeline using the template implementation without any changes.

---

### Setup

* Repository: Template implementation (no modifications)
* Execution: `benchmark.sh`
* Model weights loaded from HuggingFace
* No authentication token (HF_TOKEN not set)

---

### Observations

During execution, the following warning was observed:

> *"You are sending unauthenticated requests to the HF Hub..."*

This does **not affect correctness**, but may:

* Reduce download speed
* Impose stricter rate limits

---

### Performance Results

| Metric           | Value          |
| ---------------- | -------------- |
| Average Time     | **282.6 ms**   |
| Std Dev          | ± 9.5 ms       |
| Tokens Generated | 13             |
| Time per Token   | 21.74 ms/token |

---

### Output

* **Transcription:**
  *"Concord returned to its place amidst the tents."*

* **Accuracy:** 100.0%

* **Status:** PASS ✅

---

### Key Takeaways

* The baseline implementation is **fully functional and correct**.
* The pipeline produces **perfect transcription accuracy (100%)**.
* Runtime (~282 ms) serves as the **reference point** for all future optimizations.
* No numerical instability or kernel errors were observed.

---

### Conclusion

This experiment establishes a **reliable baseline**. All subsequent optimizations will be compared against:

* **Baseline runtime:** 282.6 ms
* **Baseline accuracy:** 100%

---

## 4. Attempted Optimization: Decode GEMV Fast Path for Small Sequence Lengths

### Objective

The goal of this experiment was to **optimize attention during decoding** by introducing a fast path for very small query sequence lengths (`seq_q <= 16`), which is common in autoregressive generation.

---

### Proposed Optimization

The idea was based on the observation that:

* During decoding, `seq_q` is often **very small (typically 1)**
* Launching full **FlashAttention-style kernels** may introduce unnecessary overhead
* A simpler **matrix-vector multiplication (GEMV)** can be faster for small sizes

---

### Implementation

A conditional fast path was introduced:

```python
if seq_q <= 16:
    if n_heads_q != n_heads_k:
        k = k.repeat_interleave(n_heads_q // n_heads_k, dim=1)
        v = v.repeat_interleave(n_heads_q // n_heads_k, dim=1)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if is_causal:
        m = torch.triu(torch.ones(seq_q, seq_k, device=q.device), diagonal=1).bool()
        scores.masked_fill_(m, float("-inf"))
        
    if attention_mask is not None:
        scores += attention_mask
        
    attn_weights = F.softmax(scores, dim=-1).to(v.dtype)
    return torch.matmul(attn_weights, v)
```

---

### Motivation

* Reduce kernel launch overhead for small inputs
* Avoid expensive Triton kernels when not needed
* Leverage efficient **PyTorch GEMV operations** for small tensors

---

### Status

❌ **Not included in final implementation**

---

### Insight

This highlights an important systems concept:

* **Different algorithms/kernels are optimal at different scales**

  * Large tensors → FlashAttention / tiled kernels
  * Small tensors → GEMV / simpler operations

This reflects real-world system design where **hybrid execution strategies** are often used.

---

### Conclusion

* The idea is **theoretically sound** and widely used in practice
* Not pursued — our Triton kernel is fully implemented and handles all cases

---
## 5. Mixed Precision Attention: BF16 Kernels

### Objective

The goal of this experiment was to **improve performance of attention kernels** by using **BF16 (bfloat16) precision** instead of FP32.

---

### Configuration

| Component      | Source                                   |
| -------------- | ---------------------------------------- |
| `layers.py`    | Optimized (FP32/TF32)                            |
| `attention.py` | Custom implementation (**BF16 kernels**) |
| `rope.py`      | Custom implementation                    |

---

### Change Implemented

The following modification was introduced in the attention pipeline:

```python
dtype = torch.bfloat16  # if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
```

This ensures that all attention computations operate in **BF16 precision**.

---

### Performance Results

| Metric           | Value          |
| ---------------- | -------------- |
| Average Time     | **286.6 ms**   |
| Std Dev          | **± 7.3 ms**   |
| Tokens Generated | 13             |
| Time per Token   | 22.05 ms/token |

---

### Output

* **Transcription:**
  *"Concord returned to its place amidst the tents."*

* **Accuracy:** 100.0%

* **Status:** PASS ✅

---

### Comparison with Previous Experiments

| Configuration               | Time         | Std Dev      | Change vs Baseline |
| --------------------------- | ------------ | ------------ | ------------------ |
| Baseline (all optimized layers)     | 282.6 ms     | ± 9.5 ms     | —                  |
| Custom Attention (FP32)     | 293.3 ms     | ± 0.9 ms     | +3.8%              |
| **Custom Attention (BF16)** | **286.6 ms** | **± 7.3 ms** | **+1.4%**          |

---

### Key Insights

#### 1. No Accuracy Degradation

* BF16 maintains **100% accuracy**
* Confirms stability of reduced precision for attention

---

#### 2. Moderate Variance Observed

* Standard deviation: **±7.3 ms**
* Higher than FP32 custom attention (±0.9 ms), but lower than previous BF16 attempt
* Indicates **some instability**, but not severe

---

#### 3. Slight Performance Improvement over FP32 Custom Attention

* Improved from **293.3 ms → 286.6 ms**
* However, still **slightly slower than baseline**

---

#### 4. Interpretation

* BF16 provides **partial benefit**, but not enough to beat baseline
* Suggests:

  * Some reduction in memory cost
  * But still limited by **kernel efficiency**

---

### Conclusion

* BF16 attention is:

  * ✅ Correct
  * ⚠️ Slightly unstable
  * ⚠️ Only marginally beneficial

* Performance gains are **limited without deeper kernel optimization**

---

### Next Steps

* Reduce dtype casting overhead
* Optimize attention kernels (tiling, fusion)
* Combine BF16 with better memory access strategies

---

