"""
Phase 2 – Mandatory Optimization 1: Tile Size Tuning for linear_kernel_tf32
=============================================================================
This script benchmarks 12 (TILE_M, TILE_N, TILE_K, num_warps, num_stages)
configurations on the three representative matrix shapes that appear most
frequently in GLM-ASR inference:

  Shape 1  – Audio encoder attention/MLP:  M=512, K=1280, N=1280
  Shape 2  – Audio encoder MLP up-proj:    M=512, K=1280, N=5120
  Shape 3  – Text decoder GEMV (1-token):  M=1,   K=3584, N=3584

Results are printed as a table and saved to tile_tune_results.txt so they
can be copy-pasted directly into the coursework report.

Usage (from hw1-asr/glm_asr_triton_template/):
    python tile_tune.py

Protected files NOT modified:  model.py, weight_loader.py, conv.py
"""

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Import the kernel from layers.py (read-only, not modified)
# ---------------------------------------------------------------------------
_here = Path(__file__).parent
sys.path.insert(0, str(_here))
from layers import linear_kernel_tf32  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration table – 12 (TILE_M, TILE_N, TILE_K, warps, stages) entries
# ---------------------------------------------------------------------------
@dataclass
class TileConfig:
    name: str
    TILE_M: int
    TILE_N: int
    TILE_K: int
    num_warps: int
    num_stages: int
    note: str = ""


CONFIGS: List[TileConfig] = [
    # ── Assignment baselines (from coursework spec) ──────────────────────────
    TileConfig("Default",   64,  64,  32, 4, 2, "assignment default"),
    TileConfig("Spec-A",   128,  64,  32, 4, 3, "assignment Config A"),
    TileConfig("Spec-B",    64, 128,  64, 8, 3, "assignment Config B / friend best"),
    TileConfig("Spec-C",   128, 128,  32, 8, 4, "assignment Config C"),

    # ── Additional sweep ─────────────────────────────────────────────────────
    TileConfig("Tiny",      32,  32,  32, 2, 2, "low register pressure"),
    TileConfig("Extra-1",   64,  64,  64, 8, 3, "larger K tile"),
    TileConfig("Extra-2",  128,  64,  64, 4, 3, "wide M, large K"),
    TileConfig("Extra-3",   64, 128,  32, 4, 2, "wide N, fewer stages"),
    TileConfig("Extra-4",  128, 128,  64, 8, 4, "large all tiles"),
    TileConfig("Extra-5",  256,  64,  32, 8, 4, "very wide M"),
    TileConfig("Extra-6",   64, 256,  32, 8, 4, "very wide N"),
    TileConfig("Extra-7",  128,  64,  32, 8, 2, "wide M, fewer stages"),
]

# ---------------------------------------------------------------------------
# Representative matrix shapes for GLM-ASR
# (long_label used in per-config output; short_label used in summary table)
# ---------------------------------------------------------------------------
SHAPES: List[Tuple[int, int, int, str, str]] = [
    (512, 1280, 1280, "audio-attn  M=512  K=1280 N=1280", "attn(512x1280)"),
    (512, 1280, 5120, "audio-mlp   M=512  K=1280 N=5120", "mlp(512x5120) "),
    (  1, 3584, 3584, "text-decode M=1    K=3584 N=3584", "txt-dec(1x3584)"),
]


# ---------------------------------------------------------------------------
# Helper: run one (config, shape) combination and return ms latency
# ---------------------------------------------------------------------------
def _pad(size: int, tile: int) -> int:
    return ((size + tile - 1) // tile) * tile


def _bench_config(cfg: TileConfig, M: int, K: int, N: int, device: torch.device) -> float:
    """
    Return median latency (ms) for linear_kernel_tf32 with the given tile config
    on an (M, K) x (K, N) matmul.  Returns float('inf') on error.
    """
    try:
        # Build padded input tensors (same logic as Linear._forward_triton)
        M_pad = _pad(M, cfg.TILE_M)
        K_pad = _pad(K, cfg.TILE_K)
        N_pad = _pad(N, cfg.TILE_N)

        a = torch.zeros((M_pad, K_pad), dtype=torch.float32, device=device)
        b = torch.zeros((K_pad, N_pad), dtype=torch.float32, device=device)
        c = torch.zeros((M_pad, N_pad), dtype=torch.float32, device=device)

        a[:M, :K] = torch.randn(M, K, dtype=torch.float32, device=device)
        b[:K, :N] = torch.randn(K, N, dtype=torch.float32, device=device)

        grid = (triton.cdiv(M_pad, cfg.TILE_M), triton.cdiv(N_pad, cfg.TILE_N))

        def _run():
            linear_kernel_tf32[grid](
                a, b, c,
                M_pad, N_pad, K_pad,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=cfg.TILE_M,
                BLOCK_N=cfg.TILE_N,
                BLOCK_K=cfg.TILE_K,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )

        # Warm-up compilation
        _run()
        torch.cuda.synchronize()

        ms = triton.testing.do_bench(_run, warmup=25, rep=100)
        return ms

    except Exception as exc:  # noqa: BLE001
        # Some configs may exceed register/shared-memory limits on small GPUs
        print(f"    [SKIP] {cfg.name} on ({M},{K},{N}): {exc}", flush=True)
        return float("inf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available – aborting.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    print(f"Triton version: {triton.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"\n{'='*80}")
    print("Phase 2 – Mandatory Optimization 1: Tile Size Tuning")
    print(f"{'='*80}\n")

    # ------------------------------------------------------------------
    # One warm-up compilation pass so the first timing isn't contaminated
    # ------------------------------------------------------------------
    print("Compiling kernels (first-run JIT) …", flush=True)
    dummy_cfg = CONFIGS[0]
    _ = _bench_config(dummy_cfg, 64, 64, 64, device)
    print("Done.\n")

    # ------------------------------------------------------------------
    # Benchmark every (config, shape) pair
    # ------------------------------------------------------------------
    results = {}  # cfg_name -> {shape_label -> ms}

    for cfg in CONFIGS:
        results[cfg.name] = {}
        print(f"Config: {cfg.name:<10}  TILE=({cfg.TILE_M},{cfg.TILE_N},{cfg.TILE_K})"
              f"  warps={cfg.num_warps}  stages={cfg.num_stages}"
              f"  [{cfg.note}]")
        for M, K, N, long_label, _short in SHAPES:
            ms = _bench_config(cfg, M, K, N, device)
            results[cfg.name][long_label] = ms
            tag = f"{ms:.3f} ms" if ms != float("inf") else "SKIP"
            print(f"    {long_label}  →  {tag}")
        print()

    # ------------------------------------------------------------------
    # Summary table  (use short labels so columns don't truncate)
    # ------------------------------------------------------------------
    long_labels  = [s[3] for s in SHAPES]
    short_labels = [s[4] for s in SHAPES]
    col_w = 15

    sep = "-" * (12 + 8 + 8 + 8 + 8 + 8 + (col_w + 2) * len(SHAPES))
    lines = []
    lines.append("\n" + "=" * len(sep))
    lines.append("SUMMARY TABLE – Latency (ms) per shape")
    lines.append("=" * len(sep))

    # header
    hdr = f"{'Config':<12}{'TILE_M':>6}{'TILE_N':>6}{'TILE_K':>6}{'wrps':>5}{'stgs':>5}"
    for slbl in short_labels:
        hdr += f"  {slbl:>{col_w}}"
    lines.append(hdr)
    lines.append(sep)

    best_total = {ll: (float("inf"), "–") for ll in long_labels}

    for cfg in CONFIGS:
        row = f"{cfg.name:<12}{cfg.TILE_M:>6}{cfg.TILE_N:>6}{cfg.TILE_K:>6}{cfg.num_warps:>5}{cfg.num_stages:>5}"
        for ll in long_labels:
            ms = results[cfg.name].get(ll, float("inf"))
            cell = f"{ms:.3f}" if ms != float("inf") else "SKIP"
            row += f"  {cell:>{col_w}}"
            if ms < best_total[ll][0]:
                best_total[ll] = (ms, cfg.name)
        lines.append(row)

    lines.append(sep)

    # Best-per-shape row
    best_row = f"{'BEST':<12}{'':>6}{'':>6}{'':>6}{'':>5}{'':>5}"
    for ll in long_labels:
        ms, name = best_total[ll]
        cell = name if ms != float("inf") else "–"
        best_row += f"  {cell:>{col_w}}"
    lines.append(best_row)
    lines.append("=" * len(sep))

    # Overall recommendation — lowest sum latency across shapes
    finite_sums = {}
    for cfg in CONFIGS:
        total = sum(results[cfg.name].get(ll, float("inf")) for ll in long_labels)
        finite_sums[cfg.name] = total
    best_overall_name = min(finite_sums, key=finite_sums.get)
    best_overall_cfg = next(c for c in CONFIGS if c.name == best_overall_name)

    # Also print per-config sum latency
    lines.append("\nSum latency across all shapes (lower = better):")
    for cfg in CONFIGS:
        s = finite_sums[cfg.name]
        marker = "  ← BEST" if cfg.name == best_overall_name else ""
        val = f"{s:.3f} ms" if s != float("inf") else "SKIP"
        lines.append(f"  {cfg.name:<10}  {val}{marker}")

    lines.append(
        f"\nOverall best config: {best_overall_name}  "
        f"TILE_M={best_overall_cfg.TILE_M} TILE_N={best_overall_cfg.TILE_N} "
        f"TILE_K={best_overall_cfg.TILE_K} "
        f"num_warps={best_overall_cfg.num_warps} num_stages={best_overall_cfg.num_stages}"
    )
    lines.append(
        f"  → In layers.py set:  TILE_M={best_overall_cfg.TILE_M}  "
        f"TILE_N={best_overall_cfg.TILE_N}  TILE_K={best_overall_cfg.TILE_K}  "
        f"NUM_WARPS={best_overall_cfg.num_warps}  NUM_STAGES={best_overall_cfg.num_stages}\n"
    )

    output = "\n".join(lines)
    print(output)

    # ------------------------------------------------------------------
    # Save to file
    # ------------------------------------------------------------------
    out_path = _here / "tile_tune_results.txt"
    with open(out_path, "w") as fh:
        fh.write(f"GPU: {gpu_name}\n")
        fh.write(f"Triton: {triton.__version__}   PyTorch: {torch.__version__}\n")
        fh.write(output)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
