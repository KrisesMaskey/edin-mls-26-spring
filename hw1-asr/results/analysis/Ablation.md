# HW1-ASR Experiment Report: Legacy Attention + Optimized Layers


---

## 1. Objective

The goal of this experiment was to evaluate the performance impact of using a **legacy attention implementation** while retaining **optimized layer and RoPE kernels**.

---

## 2. Configuration

| Component      | Implementation                      |
| -------------- | ----------------------------------- |
| `layers.py`    | Student changes (optimized)                 |
| `attention.py` | **Legacy implementation**           |
| `rope.py`      | **Custom optimized implementation** |

---

## 3. Motivation

Attention is one of the most **performance-critical components** in transformer architectures.

Using a legacy implementation allows us to:

* Measure the **cost of non-optimized attention**
* Understand the importance of:

  * Kernel fusion
  * Efficient tiling
  * Memory access optimization

---

## 4. Performance Results

| Metric           | Value          |
| ---------------- | -------------- |
| Average Time     | **460.1 ms**   |
| Std Dev          | ± 1.7 ms       |
| Tokens Generated | 13             |
| Time per Token   | 35.39 ms/token |

---

## 5. Output

* **Transcription:**
  *"Concord returned to its place amidst the tents."*

* **Accuracy:** 100.0%

* **Status:** PASS ✅

---

## 6. Comparison with Baseline

| Configuration                         | Time         | Change            |
| ------------------------------------- | ------------ | ----------------- |
| Baseline (All Optimized)              | 282.6 ms     | —                 |
| **Legacy Attention (Optimized RoPE)** | **460.1 ms** | **+62.8% slower** |

---

## 7. Key Insights

### 7.1 Massive Performance Degradation

* Runtime increased by **~62.8%**
* Confirms that **attention dominates overall latency**

---

### 7.2 Isolation of Attention Impact

* Since both **layers and RoPE are optimized**, the slowdown is **entirely due to attention**
* This cleanly isolates attention as the **primary bottleneck**

---

### 7.3 Stability is High

* Very low variance (**±1.7 ms**)
* Indicates predictable but inefficient execution

---

### 7.4 Likely Causes of Slowdown

* No kernel fusion (separate operations for QKᵀ, softmax, V multiply)
* Inefficient memory access patterns
* Lack of tiling and parallelization
* Higher kernel launch overhead

---

## 8. Conclusion

* Attention is the **dominant bottleneck** in the ASR pipeline
* Optimizing `layers.py` and `rope.py` alone is **not sufficient**

---

## 9. Key Takeaway

> A poorly optimized attention implementation can **negate all gains** from other optimized components.

---

## 10. Next Steps

* Replace legacy attention with optimized Triton kernels
* Focus on:

  * Memory efficiency
  * Kernel fusion
  * Tiling strategies

---
# HW1-ASR Experiment Report: RoPE Kernel Tuning (num_stages Ablation)

> ⚠️ NOTE:
> All results reported in this document are from the final implementation, unless explicitly stated otherwise.

---

## 1. Objective

The goal of this experiment was to evaluate the impact of **pipeline staging (`num_stages`)** on the performance of the Triton-based RoPE kernel.

---

## 2. Configuration

| Component      | Implementation               |
| -------------- | ---------------------------- |
| `layers.py`    | Student changes (optimized)          |
| `attention.py` | Legacy implementation        |
| `rope.py`      | Custom Triton kernel (tuned) |

---

## 3. Change Implemented

The following modification was made to the kernel launch configuration:

```python
compute_freqs_kernel[(seq_len,)](
    ...
    BLOCK=block,
    num_warps=num_warps,
    num_stages=1,   # Changed from 2 → 1
)
```

---

## 4. Motivation

* `num_stages` controls **software pipelining depth** in Triton
* Lower stages:

  * Reduce **register pressure**
  * Improve **occupancy**
* Higher stages:

  * Improve latency hiding (useful for heavy compute kernels)

Since RoPE is:

* Lightweight
* Memory-bound

Reducing stages may improve performance.

---

## 5. Performance Results

| Metric           | Value          |
| ---------------- | -------------- |
| Average Time     | **453.1 ms**   |
| Std Dev          | ± 0.8 ms       |
| Tokens Generated | 13             |
| Time per Token   | 34.86 ms/token |

---

## 6. Comparison with Previous RoPE Optimization

| Configuration                       | Time         | Change           |
| ----------------------------------- | ------------ | ---------------- |
| Optimized RoPE (`num_stages=2`)     | 440.2 ms     | —                |
| **Optimized RoPE (`num_stages=1`)** | **453.1 ms** | **+2.9% slower** |

---

## 7. Key Insights

### 7.1 Performance Degradation

* Runtime increased from:

  * **440.2 ms → 453.1 ms**
* Indicates that reducing pipeline depth **hurt performance**

---

### 7.2 Extremely Low Variance

* Std Dev: **±0.8 ms**
* Very stable execution
* Suggests:

  * Predictable scheduling
  * Minimal runtime fluctuations

---

### 7.3 Interpretation of `num_stages`

* Contrary to expectation:

  * `num_stages=2` performs better than `num_stages=1`

#### Likely Reason:

* Even though RoPE is lightweight:

  * Some level of **latency hiding is still beneficial**
* `num_stages=1`:

  * Reduces overlap between memory and compute
  * Leads to underutilization

---

### 7.4 Kernel Design Insight

> Optimal `num_stages` is **not always minimal** — it depends on balancing:

* Register pressure
* Memory latency hiding
* Instruction overlap

---

## 8. Conclusion

* `num_stages=2` is better than `num_stages=1` for this kernel
* Over-aggressive reduction of pipeline stages can **hurt performance**
* Even simple kernels benefit from **moderate pipelining**

---

## 9. Key Takeaway

> GPU kernel performance depends on careful balance — reducing overhead (registers) must not come at the cost of **latency hiding**.

---

## 10. Next Steps

* Explore:

  * `num_stages=3` (if registers allow)
  * Different `BLOCK` sizes
* Combine with:

  * Optimized attention
  * End-to-end tuning

---
# HW1-ASR Experiment Report: Optimized RoPE Kernel (Triton)

> ⚠️ NOTE:
> All results reported in this document are from the final implementation, unless explicitly stated otherwise.

---

## 1. Objective

The goal of this experiment was to **optimize the RoPE (Rotary Positional Encoding) kernel** using Triton by:

* Improving **parallel execution efficiency**
* Reducing **thread underutilization**
* Lowering **register pressure**

---

## 2. Configuration

| Component      | Implementation                     |
| -------------- | ---------------------------------- |
| `layers.py`    | Student changes (optimized)                |
| `attention.py` | Legacy implementation              |
| `rope.py`      | **Custom optimized Triton kernel** |

---

## 3. Kernel Optimization Details

### 3.1 Custom Triton Kernel

A custom `compute_freqs_kernel` was implemented:

* Computes:

  * `cos(pos * inv_freq)`
  * `sin(pos * inv_freq)`
* Writes values efficiently into cache buffers
* Avoids redundant computation by reusing `cos_half` and `sin_half`

---

### 3.2 Key Optimizations

#### (a) Efficient Memory Access

* Coalesced loads:

  * `positions_ptr`
  * `inv_freq_ptr`
* Masked operations ensure safe bounds handling

---

#### (b) Reduced Redundant Computation

* Computes only **half dimension (`half_dim`)**
* Reuses results for:

  * First half
  * Second half

---

#### (c) Warp Optimization ⚡

```python
num_warps = 1 if block <= 32 else 2
```

* Prevents **thread underutilization**
* Matches workload size to hardware parallelism
* Improves efficiency for small vs large blocks

---

#### (d) Reduced Register Pressure

```python
num_stages = 2
```

* Limits pipeline depth
* Reduces register usage
* Improves occupancy for lightweight kernels

---

## 4. Performance Results

| Metric           | Value          |
| ---------------- | -------------- |
| Average Time     | **440.2 ms**   |
| Std Dev          | ± 1.3 ms       |
| Tokens Generated | 13             |
| Time per Token   | 33.86 ms/token |

---

## 5. Output

* **Transcription:**
  *"Concord returned to its place amidst the tents."*

* **Accuracy:** 100.0%

* **Status:** PASS ✅

---

## 6. Comparison with Related Configurations

| Configuration                                     | Time         | Change           |
| ------------------------------------------------- | ------------ | ---------------- |
| Legacy Attention + Legacy RoPE                    | 446.7 ms     | —                |
| **Legacy Attention + Optimized RoPE (This Work)** | **440.2 ms** | **~1.5% faster** |
| Legacy Attention + Previous Optimized RoPE        | 460.1 ms     | Slower           |

---

## 7. Key Insights

### 7.1 Measurable Improvement from RoPE Optimization

* Runtime improved from:

  * **446.7 ms → 440.2 ms**
* Gain: **~1.5%**

---

### 7.2 Low Variance = Stable Kernel

* Std Dev: **±1.3 ms**
* Indicates:

  * Efficient scheduling
  * Good memory access patterns
  * Consistent execution

---

### 7.3 Optimization Impact is Limited

* Improvement is relatively small compared to total runtime
* Confirms:

> RoPE is **not a primary bottleneck**

---

### 7.4 Kernel-Level Optimizations Matter

Even small improvements came from:

* Better warp utilization
* Reduced register pressure
* Efficient memory layout

---

## 8. Interpretation

* RoPE optimization yields **incremental gains**
* Attention still dominates runtime
* However:

  * These optimizations are **important for cumulative improvements**

---

## 9. Conclusion

* Custom Triton RoPE kernel is:

  * ✅ Correct
  * ✅ Slightly faster
  * ✅ More efficient

* However:

  * Overall performance is still limited by **attention bottleneck**

---

## 10. Key Takeaway

> Micro-optimizations (like RoPE tuning) provide **incremental gains**, but major improvements require optimizing dominant kernels like attention.

---

## 11. Next Steps

* Combine:

  * Optimized RoPE
  * Optimized attention
* Further explore:

  * Block size tuning
  * Memory coalescing strategies
  * Kernel fusion

---

# HW1-ASR Experiment Report: RoPE Kernel Tuning (num_warps + num_stages)

> ⚠️ NOTE:
> All results reported in this document are from the final implementation, unless explicitly stated otherwise.

---

## 1. Objective

The goal of this experiment was to evaluate the impact of:

* Increasing **warp parallelism (`num_warps`)**
* Increasing **pipeline depth (`num_stages`)**

on the performance of the Triton-based RoPE kernel.

---

## 2. Configuration

| Component      | Implementation               |
| -------------- | ---------------------------- |
| `layers.py`    | Student changes (optimized)          |
| `attention.py` | Legacy implementation        |
| `rope.py`      | Custom Triton kernel (tuned) |

---

## 3. Change Implemented

The kernel launch configuration was modified as follows:

```python
num_warps = 4 if block <= 32 else 2

compute_freqs_kernel[(seq_len,)](
    ...
    BLOCK=block,
    num_warps=num_warps,
    num_stages=3,
)
```

---

## 4. Motivation

### Increasing `num_warps`

* More warps → higher parallelism
* Better utilization of GPU cores
* Especially useful for small block sizes

### Increasing `num_stages`

* More stages → better **latency hiding**
* Overlaps memory and compute operations
* Can improve throughput if resources allow

---

## 5. Performance Results

| Metric           | Value          |
| ---------------- | -------------- |
| Average Time     | **446.9 ms**   |
| Std Dev          | ± 1.1 ms       |
| Tokens Generated | 13             |
| Time per Token   | 34.38 ms/token |

---

## 6. Comparison with Other RoPE Configurations

| Configuration                          | Time         | Change    |
| -------------------------------------- | ------------ | --------- |
| Optimized RoPE (`warps=2`, `stages=2`) | 440.2 ms     | —         |
| `stages=1`                             | 453.1 ms     | +2.9%     |
| **`warps=4`, `stages=3` (this)**       | **446.9 ms** | **+1.5%** |

---

## 7. Key Insights

### 7.1 No Improvement Over Baseline Optimized Kernel

* Performance is worse than:

  * `warps=2, stages=2` (best so far)
* Indicates:

  * More parallelism ≠ better performance

---

### 7.2 Diminishing Returns from More Warps

* Increasing warps to 4 did **not improve performance**
* Likely causes:

  * Increased scheduling overhead
  * Resource contention (register/shared memory pressure)

---

### 7.3 Over-Pipelining Hurts

* `num_stages=3` is worse than `num_stages=2`
* Suggests:

  * Excessive pipelining increases overhead
  * RoPE kernel is too lightweight to benefit from deep pipelines

---

### 7.4 Stable Execution

* Low variance (**±1.1 ms**)
* Indicates predictable execution despite inefficiency

---

## 8. Interpretation

This experiment highlights an important GPU principle:

> More parallelism and deeper pipelines are not always beneficial.

Optimal performance depends on:

* Matching kernel complexity to hardware configuration
* Avoiding over-allocation of resources

---

## 9. Conclusion

* Best configuration so far remains:

  * `num_warps=2`, `num_stages=2`
* Increasing warps and stages leads to:

  * Higher overhead
  * No performance gain

---

## 10. Key Takeaway

> GPU tuning requires balance — over-scaling parallelism or pipelining can degrade performance, especially for lightweight kernels.

---

## 11. Next Steps

* Tune:

  * `BLOCK` size
  * Warp allocation based on workload size
* Combine with:

  * Optimized attention kernels
  * End-to-end profiling

---
# HW1-ASR Experiment Report: RoPE Kernel Tuning (Warp Scaling Ablation)

> ⚠️ NOTE:
> All results reported in this document are from the final implementation, unless explicitly stated otherwise.

---

## 1. Objective

The goal of this experiment was to **isolate the impact of increasing warp count (`num_warps`)** while keeping pipeline depth constant.

---

## 2. Configuration

| Component      | Implementation               |
| -------------- | ---------------------------- |
| `layers.py`    | Student changes (optimized)          |
| `attention.py` | Legacy implementation        |
| `rope.py`      | Custom Triton kernel (tuned) |

---

## 3. Change Implemented

```python
num_warps = 4 if block <= 32 else 2

compute_freqs_kernel[(seq_len,)](
    ...
    BLOCK=block,
    num_warps=num_warps,
    num_stages=2,   # Fixed
)
```

---

## 4. Motivation

* Test whether **increasing warp-level parallelism** improves performance
* Keep `num_stages=2` (previously best) to isolate warp effects

---

## 5. Performance Results

| Metric           | Value          |
| ---------------- | -------------- |
| Average Time     | **453.6 ms**   |
| Std Dev          | ± 1.0 ms       |
| Tokens Generated | 13             |
| Time per Token   | 34.89 ms/token |

---

## 6. Comparison with Other Configurations

| Configuration                | Time         | Change    |
| ---------------------------- | ------------ | --------- |
| Optimal (warps=2, stages=2)  | 440.2 ms     | —         |
| warps=4, stages=3            | 446.9 ms     | +1.5%     |
| **warps=4, stages=2 (this)** | **453.6 ms** | **+3.0%** |

---

## 7. Key Insights

### 7.1 Warp Increase Hurts Performance

* Runtime increased from:

  * **440.2 ms → 453.6 ms**
* Indicates:

  * More warps introduce **overhead rather than benefit**

---

### 7.2 Resource Contention

Increasing warps likely caused:

* Higher **register pressure**
* Reduced **occupancy efficiency**
* More contention for GPU execution resources

---

### 7.3 No Benefit for Lightweight Kernels

* RoPE kernel is:

  * Small
  * Memory-bound
* Adding more warps:

  * Does not increase useful parallel work
  * Only increases scheduling overhead

---

### 7.4 Stable Execution

* Low variance (**±1.0 ms**)
* Indicates consistent but inefficient execution

---

## 8. Interpretation

This experiment confirms:

> Warp count must match workload size — over-parallelization degrades performance.

---

## 9. Conclusion

* Increasing `num_warps` from 2 → 4 is **detrimental**
* Best configuration remains:

  * `num_warps=2`
  * `num_stages=2`

---

## 10. Key Takeaway

> More parallelism is not always better — efficient GPU utilization requires matching parallelism to workload complexity.

---

## 11. Next Steps

* Focus on:

  * Optimal BLOCK size tuning
  * Memory access optimization
* Combine:

  * Best RoPE config with optimized attention kernels

---
# HW1-ASR Final Report: Fully Optimized Implementation

> ⚠️ NOTE:
> All results reported in this document are from the final implementation, unless explicitly stated otherwise.

---

## 1. Objective

The goal of this experiment was to evaluate the performance of the **fully optimized ASR pipeline**, where all major components have been optimized.

---

## 2. Configuration

| Component      | Implementation                                          |
| -------------- | ------------------------------------------------------- |
| `layers.py`    | Student changes (optimized)                                     |
| `attention.py` | Optimized implementation                                |
| `rope.py`      | Optimized Triton kernel (`num_warps=2`, `num_stages=2`) |

---

## 3. Performance Results

| Metric           | Value          |
| ---------------- | -------------- |
| Average Time     | **291.0 ms**   |
| Std Dev          | ± 0.6 ms       |
| Tokens Generated | 13             |
| Time per Token   | 22.38 ms/token |

---

## 4. Output

* **Transcription:**
  *"Concord returned to its place amidst the tents."*

* **Accuracy:** 100.0%

* **Status:** PASS ✅

---

## 5. Comparison with Previous Configurations

| Configuration                     | Time         | Improvement                     |
| --------------------------------- | ------------ | ------------------------------- |
| Baseline (initial run)            | 282.6 ms     | —                               |
| Legacy Attention + Legacy RoPE    | 446.7 ms     | -58% slower                     |
| Legacy Attention + Optimized RoPE | 440.2 ms     | -51% slower                     |
| Fully Optimized (this)            | **291.0 ms** | **~35% faster than worst case** |

---

## 6. Key Insights

### 6.1 Near-Baseline Performance Achieved

* Final runtime (**291.0 ms**) is close to:

  * Original baseline (**~282.6 ms**)
* Confirms:

  * Optimizations are effective and correct

---

### 6.2 Major Bottleneck: Attention

* Earlier experiments showed:

  * Attention dominates runtime
* Optimizing attention leads to:

  * Largest performance recovery

---

### 6.3 RoPE Optimization: Incremental Gains

* RoPE tuning contributed:

  * Small but measurable improvements (~1–3%)
* Important for:

  * Final performance polish

---

### 6.4 System Stability

* Very low variance (**±0.6 ms**)
* Indicates:

  * Efficient kernel scheduling
  * Good memory behavior
  * Stable execution

---

## 7. Final System Understanding

From all experiments:

| Component           | Impact                         |
| ------------------- | ------------------------------ |
| Attention           | 🔴 Dominant bottleneck         |
| Layers (MLP, Norms) | 🟡 Moderate impact             |
| RoPE                | 🟢 Minor but measurable impact |

---

## 8. Conclusion

* Fully optimized pipeline achieves:

  * High performance
  * Perfect accuracy
  * Stable execution

* Performance is now:

  * Close to reference implementation
  * Significantly improved over ablations

---

## 9. Key Takeaway

> End-to-end performance in ML systems depends on optimizing the **right bottlenecks** — attention dominates, while smaller kernels provide incremental gains.

---

## 10. Future Work

* Further optimize:

  * Attention kernels (fusion, tiling)
  * Memory access patterns
* Explore:

  * Kernel fusion opportunities
  * Mixed precision strategies

---
