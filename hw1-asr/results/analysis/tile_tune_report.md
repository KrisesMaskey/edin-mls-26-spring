# Phase 2 – Mandatory Optimization 1: Tile Size Tuning

## Overview

We systematically benchmarked **12 tile-size configurations** for `linear_kernel_tf32`
to identify the optimal `(TILE_M, TILE_N, TILE_K, num_warps, num_stages)` combination
on the target hardware.

**Hardware:** NVIDIA H200 MIG 1g.18gb (1/7 H200 slice, ~18 GB HBM3)
**Software:** Triton 3.6.0 · PyTorch 2.10.0+cu128 · CUDA 12.8
**Script:** `tile_tune.py` — standalone benchmark, no protected files modified

---

## Matrix Shapes Benchmarked

Three shapes representative of the most compute-intensive linear layers in GLM-ASR:

| Shape label | M | K | N | Layer |
|---|---|---|---|---|
| audio-attn | 512 | 1280 | 1280 | Audio encoder Q/K/V + output projection |
| audio-mlp | 512 | 1280 | 5120 | Audio encoder MLP up-projection (4× hidden) |
| text-decode | 1 | 3584 | 3584 | Text decoder single-token GEMV |

---

## Configurations Tested

| Config | TILE_M | TILE_N | TILE_K | num_warps | num_stages | Notes |
|---|---|---|---|---|---|---|
| Default | 64 | 64 | 32 | 4 | 2 | Assignment default |
| Spec-A | 128 | 64 | 32 | 4 | 3 | Assignment Config A |
| Spec-B | 64 | 128 | 64 | 8 | 3 | Assignment Config B / friend's best |
| Spec-C | 128 | 128 | 32 | 8 | 4 | Assignment Config C |
| Tiny | 32 | 32 | 32 | 2 | 2 | Low register pressure |
| Extra-1 | 64 | 64 | 64 | 8 | 3 | Larger K tile |
| Extra-2 | 128 | 64 | 64 | 4 | 3 | Wide M, large K |
| Extra-3 | 64 | 128 | 32 | 4 | 2 | Wide N, fewer stages |
| Extra-4 | 128 | 128 | 64 | 8 | 4 | Large all tiles — **SKIP** (OOM) |
| Extra-5 | 256 | 64 | 32 | 8 | 4 | Very wide M |
| Extra-6 | 64 | 256 | 32 | 8 | 4 | Very wide N |
| Extra-7 | 128 | 64 | 32 | 8 | 2 | Wide M, fewer stages |

---

## Raw Results (ms)

| Config | TILE_M | TILE_N | TILE_K | wrps | stgs | audio-attn (ms) | audio-mlp (ms) | text-decode (ms) | Sum (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Default | 64 | 64 | 32 | 4 | 2 | 0.291 | 1.158 | 0.327 | 1.776 |
| Spec-A | 128 | 64 | 32 | 4 | 3 | 0.173 | 0.675 | 0.382 | 1.230 |
| Spec-B | 64 | 128 | 64 | 8 | 3 | 0.328 | 1.297 | 0.351 | 1.976 |
| Spec-C | 128 | 128 | 32 | 8 | 4 | 0.226 | 0.714 | 0.399 | 1.339 |
| **Tiny** | **32** | **32** | **32** | **2** | **2** | **0.156** | **0.573** | **0.131** | **0.860** ✓ |
| Extra-1 | 64 | 64 | 64 | 8 | 3 | 0.319 | 1.260 | 0.354 | 1.933 |
| Extra-2 | 128 | 64 | 64 | 4 | 3 | 0.183 | 0.710 | 0.387 | 1.280 |
| Extra-3 | 64 | 128 | 32 | 4 | 2 | 0.351 | 1.214 | 0.343 | 1.908 |
| Extra-4 | 128 | 128 | 64 | 8 | 4 | SKIP | SKIP | SKIP | — |
| Extra-5 | 256 | 64 | 32 | 8 | 4 | 0.133 | 0.436 | 0.469 | 1.038 |
| Extra-6 | 64 | 256 | 32 | 8 | 4 | 0.368 | 1.233 | 0.330 | 1.931 |
| Extra-7 | 128 | 64 | 32 | 8 | 2 | 0.186 | 0.721 | 0.480 | 1.387 |

Extra-4 was skipped on all shapes: shared memory required (262 144 B) exceeded hardware limit (232 448 B).

---

## Best Config Per Shape

| Shape | Best Config | Latency (ms) |
|---|---|---|
| audio-attn (512×1280×1280) | Extra-5 (256,64,32) | 0.133 |
| audio-mlp (512×1280×5120) | Extra-5 (256,64,32) | 0.436 |
| text-decode (1×3584×3584) | Tiny (32,32,32) | 0.131 |

---

## Winner: Tiny (TILE_M=32, TILE_N=32, TILE_K=32)

**Winning config applied in `layers.py`:**

```python
TILE_M     = 32
TILE_N     = 32
TILE_K     = 32
NUM_WARPS  = 2
NUM_STAGES = 2
```

**Sum latency: 0.860 ms** — lowest across all three shapes combined.

### Why small tiles win on H200 MIG 1g.18gb

The H200 MIG `1g.18gb` partition is 1/7th of the full H200:
- ~2 SM compute slices (vs 132 SMs on the full H200)
- Proportionally reduced L1/shared-memory capacity (hardware limit 232 KB vs ~835 KB full)

With so few SMs, a large tile like `128×128` keeps each SM busy for a long time but
leaves the rest idle — the occupancy-vs-parallelism trade-off flips. `32×32` tiles
produce far more CTAs (thread blocks), which pipeline better across the available SMs
and keep the memory subsystem fed. They also require only 2 warps and 2 pipeline stages,
minimising register and shared-memory pressure on the constrained slice.

Extra-5 (TILE_M=256) wins for the two audio shapes because M=512 produces just two
output tiles in the M-dimension, maximising output reuse; however it hurts the GEMV
case (M=1) which only needs a single tile and cannot exploit M-dimension parallelism.

---

## Speedup vs Default

| Shape | Default (ms) | Tiny (ms) | Speedup |
|---|---|---|---|
| audio-attn | 0.291 | 0.156 | **1.87×** |
| audio-mlp | 1.158 | 0.573 | **2.02×** |
| text-decode | 0.327 | 0.131 | **2.50×** |

Average speedup of the winning config over the assignment default: **~2.1×**.

---

## Notes on Friend's Config

The friend's reported best (Spec-B: 64,128,64) performs **well below default** on this
GPU (sum 1.976 ms vs 1.776 ms). This is expected: larger tiles with 8 warps and 3
pipeline stages are designed for fully-provisioned A100/H100 GPUs. On the constrained
MIG slice their shared-memory demands create contention, hurting rather than helping.
