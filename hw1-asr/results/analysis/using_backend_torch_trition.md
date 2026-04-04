Here’s a clean **Markdown file** you can directly use for your experiment:

---

````md
# Torch vs Triton Backend Ablation

## Objective
This experiment evaluates whether using **PyTorch (cuBLAS)** vs **Triton kernels** for linear/GEMM operations affects performance.

We specifically test:
1. **Hybrid backend (Torch + Triton)**  
   - Torch used for **small batch sizes (M ≤ 4)**
   - Triton used otherwise

2. **Pure Triton backend**
   - Triton used for **all batch sizes**

---

## Backend Selection Logic

### Hybrid (Torch + Triton)
```python
def __call__(self, x: torch.Tensor) -> torch.Tensor:
    M = int(np.prod(x.shape[:-1]))
    print(M, '=================================batchsize==================')

    # Use Torch (cuBLAS GEMV) for very small batch sizes
    if M <= 4:
        print('using torch')
        return self._forward_torch(x)

    return self._forward_triton(x)
````

### Pure Triton

```python
def __call__(self, x: torch.Tensor) -> torch.Tensor:
    M = int(np.prod(x.shape[:-1]))
    print(M, '=================================batchsize==================')

    return self._forward_triton(x)
```

---

## Key Insight

* **Small batch sizes (M ≤ 4)** → Torch is preferred

  * Triton kernels underutilize GPU (very low occupancy)
  * cuBLAS GEMV is more efficient

* **Larger batch sizes** → Triton is preferred

  * Better parallelism and throughput

---

## Results

---

# 1. Hybrid Backend (Torch for small, Triton for large)

### Performance Summary

| Component             | Time (ms)      | % of Total |
| --------------------- | -------------- | ---------- |
| Audio Encoder         | 3062.18        | 51.2%      |
| Multi-modal Projector | 25.30          | 0.4%       |
| Decoder (Prefill)     | 779.83         | 13.0%      |
| Decoder (50 steps)    | 2110.24        | 35.3%      |
| **Total**             | **5977.55 ms** |            |

---

### Decode Performance

* Decode Step Avg: **42.20 ms**
* Variance: **± 49.48 ms**

---

### Attention Performance

| Method            | Time      |
| ----------------- | --------- |
| Standard (einsum) | 170.07 ms |
| Torch matmul      | 0.58 ms   |

---

### GEMM Performance

| Method       | Time     |
| ------------ | -------- |
| Torch matmul | 1.39 ms  |
| Torch einsum | 1.10 ms  |
| Torch GEMM   | 1.07 ms  |
| Full MLP     | 64.32 ms |

---

### Observations

* Torch is triggered frequently during **decode phase (M = 1)**
* Significant speedup in decode vs pure Triton
* Lower total latency

---

# 2. Pure Triton Backend (All batch sizes)

### Performance Summary

| Component             | Time (ms)      | % of Total |
| --------------------- | -------------- | ---------- |
| Audio Encoder         | 3026.77        | 48.5%      |
| Multi-modal Projector | 27.01          | 0.4%       |
| Decoder (Prefill)     | 896.35         | 14.4%      |
| Decoder (50 steps)    | 2284.80        | 36.6%      |
| **Total**             | **6234.93 ms** |            |

---

### Decode Performance

* Decode Step Avg: **45.70 ms**
* Variance: **± 40.59 ms**

---

### Attention Performance

| Method            | Time      |
| ----------------- | --------- |
| Standard (einsum) | 178.00 ms |
| Torch matmul      | 0.60 ms   |

---

### GEMM Performance

| Method       | Time     |
| ------------ | -------- |
| Torch matmul | 4.79 ms  |
| Torch einsum | 1.11 ms  |
| Torch GEMM   | 1.07 ms  |
| Full MLP     | 69.57 ms |

---

### Observations

* Triton is used even for **M = 1 (decode)**
* Leads to:

  * Poor GPU utilization
  * Higher latency in decode steps
* Slightly worse total performance vs hybrid

---

## Comparison

| Metric        | Hybrid (Torch + Triton) | Pure Triton |
| ------------- | ----------------------- | ----------- |
| Total Time    | **5977.55 ms**          | 6234.93 ms  |
| Decode Step   | **42.20 ms**            | 45.70 ms    |
| Decoder Total | **2110.24 ms**          | 2284.80 ms  |

---

## Key Takeaways

1. **Torch is critical for small batch sizes**

   * Especially during autoregressive decoding (M = 1)

2. **Triton alone is not optimal**

   * Kernel launch + tiling overhead dominates at low M

3. **Hybrid approach is best**

   * Combines:

     * Torch efficiency (small M)
     * Triton scalability (large M)

---

## Final Conclusion

Using **dynamic backend selection (Torch + Triton)** provides the best performance.

* ✔ Faster decode
* ✔ Lower total latency
* ✔ Better GPU utilization

This confirms that **kernel specialization based on batch size is essential** for optimal inference performance.

## End-to-End Benchmark (Full Triton Backend)

This benchmark measures full inference latency using:
- **Pure Triton backend (all kernels)**
- **generate_v8 implementation**

---

### Setup
- Audio Duration: 3.50s  
- Expected Output:  
  `CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS`

---

### Results

| Metric | Value |
|------|------|
| Time | **471.1 ms** |
| Variance | ± 14.9 ms |
| Tokens | 13 |
| Speed | 36.24 ms/token |
| Accuracy | 100% |
| Status | PASS |

---

### Run Breakdown
- Run 1: 492.1 ms  
- Run 2: 459.1 ms  
- Run 3: 462.2 ms  

---

### Observations

- **Higher variance** compared to microbenchmarks  
  - Likely due to:
    - Kernel launch overhead
    - GPU scheduling variability

- **Slower than hybrid backend**
  - Decode phase dominated by **small batch (M = 1)**
  - Triton underperforms vs Torch in this regime

- Despite slower speed:
  - ✔ Correct transcription  
  - ✔ Stable execution  
  - ✔ No accuracy degradation  

---

### Key Insight

Even with highly optimized Triton kernels:
- **End-to-end latency is bottlenecked by decode steps**
- For **autoregressive generation**, kernel choice at small batch sizes is critical

---

### Conclusion (Full Triton)

- ✔ Works correctly  
- ✖ Not optimal for latency  
- ⚠ Suffers in low-batch decode scenarios  

➡ Reinforces need for **hybrid Torch + Triton execution**

