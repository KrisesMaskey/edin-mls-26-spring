"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple
import torch.nn.functional as F
import math

def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None



# benchmark function — perf_report goes here, completely separate
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[16, 64, 128, 256, 512, 1024, 2048],
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='ms',
        plot_name='attention-benchmark',
        args={'batch': 1, 'heads': 4, 'head_dim': 64},
    )
)
def benchmark(seq_len, provider, batch, heads, head_dim):
    device = torch.device('cuda')
    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)

    if provider == 'torch':
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
    else:
        fn = lambda: scaled_dot_product_attention(q, k, v)

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms



def print_full_autotune_report(batch, num_heads, seq_len, head_dim):
    device = "cuda"
    dtype = torch.float16
    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)
    
    q_flat = q.reshape(-1, seq_len, head_dim).contiguous()
    k_flat = k.reshape(-1, seq_len, head_dim).contiguous()
    v_flat = v.reshape(-1, seq_len, head_dim).contiguous()
    out = torch.empty_like(q_flat)

    print(f"\n--- Full Performance Report (Seq: {seq_len}) ---")
    print(f"{'BLOCK_M':<8} | {'BLOCK_N':<8} | {'Warps':<6} | {'Stages':<8} | {'Time (ms)':<10}")
    print("-" * 55)

    configs = get_autotune_configs()
    
    for config in configs:
        kw = config.kwargs
        bm, bn = kw['BLOCK_M'], kw['BLOCK_N']
        warps, stages = config.num_warps, config.num_stages
        
        # We call .fn to bypass the Autotuner wrapper and manualy pass params
        def benchmark_call():
            grid = (batch * num_heads, triton.cdiv(seq_len, bm))
            fused_flash_attention_kernel.fn[grid](
                q_flat, k_flat, v_flat, out, None,
                q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
                k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
                v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                0, 0, 0, seq_len, seq_len, head_dim, float(scale),
                HAS_MASK=False, IS_CAUSAL=True, 
                BLOCK_D=128, # Keep this at 128 for padding alignment
                BLOCK_M=bm, 
                BLOCK_N=bn,
                num_warps=warps,
                num_stages=stages,
            )

        try:
            ms = triton.testing.do_bench(benchmark_call)
            print(f"{bm:<8} | {bn:<8} | {warps:<6} | {stages:<8} | {ms:<10.4f}")
        except Exception as e:
            # Some older GPUs or configurations with too many stages/warps might OOM the SRAM
            print(f"{bm:<8} | {bn:<8} | {warps:<6} | {stages:<8} | FAILED")

def get_autotune_configs():
    return [
        # 64x64 variants (The heavy hitters)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        
        # 64x32 variants (Good for memory bandwidth)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=5),
        
        # 32x64 variants
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        
        # 32x32 variants (Small & Fast)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=3),
        
        # 16x64 or 64x16 (Extreme cases)
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16}, num_warps=4, num_stages=2),
    ]


@triton.heuristics({
    # We define BLOCK_M and BLOCK_N as heuristics based on input args
    'BLOCK_M': lambda args: 64, #if args['seq_q'] >= 64 else 32,
    'BLOCK_N': lambda args: 32,
    # num_warps and num_stages can also be set here if not passed at launch
    'num_warps': lambda args: 4,
    'num_stages': lambda args: 3# if torch.cuda.get_device_capability()[0] >= 8 else 2,
})
@triton.jit
def fused_flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, mask_ptr,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    stride_m0, stride_m1, stride_m2,
    seq_q, seq_k, head_dim,
    scale,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    #tl.static_print("BLOCK_M    =", BLOCK_M, "bLOCKN", BLOCK_N)

    q_ptrs = q_ptr + pid_bh * stride_q0 + offs_m[:, None] * stride_q1 + offs_d[None, :] * stride_q2
    k_ptrs = k_ptr + pid_bh * stride_k0 + offs_d[:, None] * stride_k2 + offs_n[None, :] * stride_k1
    v_ptrs = v_ptr + pid_bh * stride_v0 + offs_n[:, None] * stride_v1 + offs_d[None, :] * stride_v2

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_q) & (offs_d[None, :] < head_dim), other=0.0)

    hi = seq_k
    if IS_CAUSAL:
        hi = tl.minimum(seq_k, (pid_m + 1) * BLOCK_M)

    for start_n in range(0, hi, BLOCK_N):
        current_offs_n = start_n + offs_n
        
        k = tl.load(k_ptrs + start_n * stride_k1, mask=(offs_d[:, None] < head_dim) & (current_offs_n[None, :] < seq_k), other=0.0)
        v = tl.load(v_ptrs + start_n * stride_v1, mask=(current_offs_n[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0)

        qk = tl.dot(q, k) * scale

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= current_offs_n[None, :], qk, float("-inf"))
        qk = tl.where(current_offs_n[None, :] < seq_k, qk, float("-inf"))

        if HAS_MASK:
            mask_ptrs = mask_ptr + pid_bh * stride_m0 + offs_m[:, None] * stride_m1 + current_offs_n[None, :] * stride_m2
            mask_vals = tl.load(mask_ptrs, mask=(offs_m[:, None] < seq_q) & (current_offs_n[None, :] < seq_k), other=float("-inf"))
            qk += mask_vals

        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        #acc += tl.dot(p.to(v.dtype.element_ty), v)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc = acc / l_i[:, None]
    
    out_ptrs = out_ptr + pid_bh * stride_o0 + offs_m[:, None] * stride_o1 + offs_d[None, :] * stride_o2
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=(offs_m[:, None] < seq_q) & (offs_d[None, :] < head_dim))


# ──────────────────────────────────────────────────────────────
# Flash Attention V1 — Triton Kernel
# ──────────────────────────────────────────────────────────────
 
# @triton.jit
# def flash_attention_v1_kernel(
#     # ── Pointers ──────────────────────────────────────────────
#     q_ptr, k_ptr, v_ptr,
#     out_ptr,
#     lse_ptr,                    # [BH, seq_q]  log-sum-exp saved for backward
#     # ── Strides for Q / K / V / Out (BH x seq x dim) ─────────
#     stride_q0, stride_q1, stride_q2,
#     stride_k0, stride_k1, stride_k2,
#     stride_v0, stride_v1, stride_v2,
#     stride_o0, stride_o1, stride_o2,
#     # ── Strides for LSE (BH x seq) ───────────────────────────
#     stride_lse0, stride_lse1,
#     # ── Sequence / dimension sizes ────────────────────────────
#     seq_q, seq_k, head_dim,
#     scale,
#     # ── Compile-time constants ────────────────────────────────
#     IS_CAUSAL: tl.constexpr,
#     BLOCK_M:   tl.constexpr,    # rows of Q processed per program
#     BLOCK_N:   tl.constexpr,    # cols of K/V processed per inner iteration
#     BLOCK_D:   tl.constexpr,    # head_dim rounded to next power-of-two
# ):
#     """
#     Each Triton program handles one (batch x head, q_block) tile.
#     Grid: (batch * num_heads,  ceil(seq_q / BLOCK_M))
#     """
 
#     # ── Program IDs ───────────────────────────────────────────
#     pid_bh = tl.program_id(0)   # which batch x head
#     pid_m  = tl.program_id(1)   # which Q-block row tile
 
#     # ── Index ranges ──────────────────────────────────────────
#     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query row indices
#     offs_d = tl.arange(0, BLOCK_D)                      # head-dim indices
 
#     # ── Load Q tile [BLOCK_M, BLOCK_D] into SRAM ──────────────
#     # Q stays in registers for the entire K/V loop — no re-loads.
#     q_ptrs = (q_ptr
#               + pid_bh          * stride_q0
#               + offs_m[:, None] * stride_q1
#               + offs_d[None, :] * stride_q2)
#     q = tl.load(
#         q_ptrs,
#         mask=(offs_m[:, None] < seq_q) & (offs_d[None, :] < head_dim),
#         other=0.0,
#     )
 
#     # ── V1 running accumulators (per query row) ───────────────
#     m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # row max
#     l_i = tl.zeros([BLOCK_M],               dtype=tl.float32)  # normaliser
#     acc = tl.zeros([BLOCK_M, BLOCK_D],       dtype=tl.float32)  # output
 
#     # ── Inner loop: iterate over K/V tiles ────────────────────
#     #
#     #   V1 NOTE: The original paper describes an *outer* loop over
#     #   K/V tiles with Q tiles on the inside. In this kernel both
#     #   loops collapse into one because each program *is* one Q tile,
#     #   so we only need the K/V loop here. The parallelism across Q
#     #   tiles is expressed via pid_m in the grid.
#     #
#     for start_n in range(0, seq_k, BLOCK_N):
#         offs_n = start_n + tl.arange(0, BLOCK_N)
 
#         # Load K tile as [BLOCK_D, BLOCK_N] — pre-transposed for tl.dot(q, k)
#         k_ptrs = (k_ptr
#                   + pid_bh          * stride_k0
#                   + offs_d[:, None] * stride_k2    # dim  -> rows
#                   + offs_n[None, :] * stride_k1)   # seq  -> cols
#         k = tl.load(
#             k_ptrs,
#             mask=(offs_d[:, None] < head_dim) & (offs_n[None, :] < seq_k),
#             other=0.0,
#         )
 
#         # Load V tile as [BLOCK_N, BLOCK_D]
#         v_ptrs = (v_ptr
#                   + pid_bh          * stride_v0
#                   + offs_n[:, None] * stride_v1
#                   + offs_d[None, :] * stride_v2)
#         v = tl.load(
#             v_ptrs,
#             mask=(offs_n[:, None] < seq_k) & (offs_d[None, :] < head_dim),
#             other=0.0,
#         )
 
#         # ── Scaled dot-product: [BLOCK_M, BLOCK_N] ────────────
#         # k is already [BLOCK_D, BLOCK_N], so tl.dot(q, k) = Q * Kt
#         qk = tl.dot(q, k, input_precision="ieee") * scale
 
#         # ── Causal mask ───────────────────────────────────────
#         # Positions where key index > query index are masked to -inf
#         if IS_CAUSAL:
#             qk = tl.where(
#                 offs_m[:, None] >= offs_n[None, :],
#                 qk,
#                 float("-inf"),
#             )
 
#         # Always mask padding beyond seq_k
#         qk = tl.where(offs_n[None, :] < seq_k, qk, float("-inf"))
 
#         # ── Online softmax update (V1 Algorithm 1) ────────────
 
#         # 1. New per-row running maximum
#         m_ij    = tl.max(qk, axis=1)               # [BLOCK_M]
#         m_i_new = tl.maximum(m_i, m_ij)            # [BLOCK_M]
 
#         # 2. Rescale correction: old exp-values shift when max changes
#         #    alpha = exp(m_old - m_new)
#         alpha = tl.exp(m_i - m_i_new)              # [BLOCK_M]
 
#         # 3. Unnormalised attention weights for this K block
#         #    p = exp(S - m_new)  ->  shape [BLOCK_M, BLOCK_N]
#         p = tl.exp(qk - m_i_new[:, None])
 
#         # 4. Rescale accumulator, then add p*V contribution
#         #    O_new = diag(alpha) * O_old + p * V
#         acc  = acc * alpha[:, None]
#         acc += tl.dot(p.to(v.dtype), v, input_precision="ieee")
 
#         # 5. Update running normaliser
#         #    l_new = alpha * l_old + rowsum(p)
#         l_i = l_i * alpha + tl.sum(p, axis=1)
 
#         m_i = m_i_new
 
#     # ── Final normalisation ────────────────────────────────────
#     acc = acc / l_i[:, None]
 
#     # ── Store LSE = m + log(l) — required for V1 backward pass ──
#     #
#     #   V1 KEY DIFFERENCE: V1 saves the log-sum-exp to HBM so the
#     #   backward kernel can recover softmax weights without re-running
#     #   the forward. V2 instead recomputes them on-the-fly, saving
#     #   the HBM write at the cost of extra compute in the backward.
#     #
#     lse = m_i + tl.log(l_i)
#     lse_ptrs = lse_ptr + pid_bh * stride_lse0 + offs_m * stride_lse1
#     tl.store(lse_ptrs, lse, mask=offs_m < seq_q)
 
#     # ── Write output tile ─────────────────────────────────────
#     out_ptrs = (out_ptr
#                 + pid_bh          * stride_o0
#                 + offs_m[:, None] * stride_o1
#                 + offs_d[None, :] * stride_o2)
#     tl.store(
#         out_ptrs,
#         acc.to(out_ptr.dtype.element_ty),
#         mask=(offs_m[:, None] < seq_q) & (offs_d[None, :] < head_dim),
#     )
 

@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1

MAX_ATTENTION_DIM = 256

 #this is for flashattentionv1
# def scaled_dot_product_attention(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,   # accepted but unused (V1 only supports causal)
#     is_causal: bool = False,
#     scale: Optional[float] = None,
# ) -> torch.Tensor:
#     """
#     Flash Attention V1 — forward pass.
#     Drop-in replacement for the V2 scaled_dot_product_attention signature.
 
#     Args:
#         q              : [batch, heads, seq_q, head_dim]  query tensor
#         k              : [batch, heads, seq_k, head_dim]  key tensor
#         v              : [batch, heads, seq_k, head_dim]  value tensor
#         attention_mask : ignored in V1 (V2 feature); kept for API compatibility
#         is_causal      : apply autoregressive (causal) masking
#         scale          : softmax scale factor, defaults to 1/sqrt(head_dim)
 
#     Returns:
#         out : [batch, heads, seq_q, head_dim]  attention output
 
#     Note — LSE:
#         V1 computes lse = m + log(l) internally and writes it to a temporary
#         buffer (needed by the backward kernel). It is NOT returned here so
#         this function is a drop-in for the V2 wrapper. If you need lse for
#         a custom backward, promote the lse tensor out of this function.
 
#     V1 block-size choice (BLOCK_M = BLOCK_N = 64):
#         The original FA-V1 paper derives block sizes from an SRAM budget M:
#             BLOCK_M = BLOCK_N = floor(M / (4 * head_dim))
#         64 is a safe symmetric default for A100/V100 configs.
#     """
#     batch, num_heads, seq_q, head_dim = q.shape
#     _, _, seq_k, _ = k.shape
 
#     if scale is None:
#         scale = 1.0 / math.sqrt(head_dim)
 
#     head_dim_padded = next_power_of_two(head_dim)
#     use_triton      = q.is_cuda and head_dim_padded <= MAX_ATTENTION_DIM
 
#     # Cast to bfloat16 for peak throughput on Ampere+
#     dtype   = torch.bfloat16
#     q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
 
#     if use_triton:
#         BH     = batch * num_heads
#         q_flat = q.reshape(BH, seq_q, head_dim).contiguous()
#         k_flat = k.reshape(BH, seq_k, head_dim).contiguous()
#         v_flat = v.reshape(BH, seq_k, head_dim).contiguous()
 
#         output = torch.empty((BH, seq_q, head_dim), dtype=dtype,         device=q.device)
#         lse    = torch.empty((BH, seq_q),           dtype=torch.float32, device=q.device)
 
#         # ── V1-style block sizes: symmetric squares ────────────
#         # V1 NOTE: V2 decouples BLOCK_M and BLOCK_N and autotuens them
#         # via a META grid lambda. V1 fixes both to the same value.
#         BLOCK_M = 64
#         BLOCK_N = 64
#         BLOCK_D = head_dim_padded
 
#         # Static grid tuple
#         # V1 NOTE: V2 uses `grid = lambda META: (...)` to expose BLOCK_M
#         # as an autotune parameter. V1 fixes the grid at dispatch time.
#         grid = (BH, triton.cdiv(seq_q, BLOCK_M))
 
#         flash_attention_v1_kernel[grid](
#             q_flat, k_flat, v_flat, output, lse,
#             q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
#             k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
#             v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
#             output.stride(0), output.stride(1), output.stride(2),
#             lse.stride(0),    lse.stride(1),
#             seq_q, seq_k, head_dim,
#             float(scale),
#             IS_CAUSAL=is_causal,
#             BLOCK_M=BLOCK_M,
#             BLOCK_N=BLOCK_N,
#             BLOCK_D=BLOCK_D,
#             num_warps=4,
#             # V1 NOTE: num_stages=1 means no software pipelining of
#             # HBM->SRAM loads. V2 raises this to 2-3 to prefetch the
#             # next K/V tile while the current one is being processed,
#             # hiding memory latency at the cost of more SRAM usage.
#             num_stages=1,
#         )
 
#         # lse is written to HBM by the kernel (V1 backward requirement)
#         # but not returned — keeps this a drop-in for the V2 wrapper.
#         return output.reshape(batch, num_heads, seq_q, head_dim)
 
#     # ── PyTorch fallback (CPU or oversized head_dim) ──────────
#     scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
 
#     if is_causal:
#         causal_mask = torch.triu(
#             torch.ones(seq_q, seq_k, dtype=torch.float32, device=q.device),
#             diagonal=1,
#         ) * -1e9
#         scores = scores + causal_mask[None, None, :, :]
 
#     if attention_mask is not None:
#         scores = scores + attention_mask
 
#     scores      = scores - torch.max(scores, dim=-1, keepdim=True).values
#     attn        = torch.exp(scores)
#     attn        = attn / torch.sum(attn, dim=-1, keepdim=True)
#     return torch.einsum("bnqk,bnkd->bnqd", attn, v).to(q.dtype)
 

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    head_dim_padded = next_power_of_two(head_dim)
    
    # Triton configuration heuristics
    use_triton = q.is_cuda and head_dim_padded <= MAX_ATTENTION_DIM


    dtype = torch.bfloat16 #if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)

    if use_triton:
        # Note: FlashAttention achieves peak performance using float16/bfloat16. 
        # For pure speedup, cast your inputs before passing them into this function.
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).contiguous()
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).contiguous()
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).contiguous()

        #output = torch.empty((batch * num_heads, seq_q, head_dim), dtype=q.dtype, device=q.device)
        output = torch.empty((batch * num_heads, seq_q, head_dim), dtype=dtype, device=q.device)
        # Prepare external mask if provided
        has_mask = attention_mask is not None
        m_strides = (0, 0, 0)
        mask_ptr = None
        if has_mask:
            if attention_mask.ndim == 4:
                attention_mask = attention_mask.reshape(batch * num_heads, seq_q, seq_k).contiguous()
            mask_ptr = attention_mask
            m_strides = (attention_mask.stride(0), attention_mask.stride(1), attention_mask.stride(2))

        # Block dimensions for SRAM tiling
        # BLOCK_M = 64 # Size of Q chunk
        # BLOCK_N = 32 # Size of K/V chunk
        BLOCK_D = head_dim_padded

        #grid = (batch * num_heads, triton.cdiv(seq_q, BLOCK_M))
        grid = lambda META: (batch * num_heads, triton.cdiv(seq_q, META['BLOCK_M']))

    #fused_flash_attention_kernel[grid](

        fused_flash_attention_kernel[grid](
            q_flat, k_flat, v_flat, output, mask_ptr,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            m_strides[0], m_strides[1], m_strides[2],
            seq_q, seq_k, head_dim,
            float(scale),
            HAS_MASK=has_mask,
            IS_CAUSAL=is_causal,
            # BLOCK_M=BLOCK_M,
            # BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            # num_warps=4,    # <--- ADDED HERE
            # num_stages=3,   # <--- ADDED HERE
        )

        return output.reshape(batch, num_heads, seq_q, head_dim)

    # ... Fallback to standard PyTorch implementation (as you had it) ...
    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
    if is_causal:
        mask = torch.triu(torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device), diagonal=1) * -1e9
        scores = scores + mask[None, None, :, :]
    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    return torch.einsum("bnqk,bnkd->bnqd", attn_weights, v).to(q.dtype)

if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2 :] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    print("\nOutput statistics:")
    print(f"  Mean: {float(output.mean()):.4f}")
    print(f"  Std:  {float(output.std()):.4f}")
    print(f"  Min:  {float(output.min()):.4f}")
    print(f"  Max:  {float(output.max()):.4f}")

    print("\nTriton Attention working!")


    # if device.type == "cuda":
    #     print("\nRunning benchmarks...")
    #     benchmark.run(print_data=True, show_plots=False)

    # if not torch.cuda.is_available():
    #     print("CUDA is required to benchmark Triton kernels.")
    # else:
    #     print("Running benchmark... (this takes a moment as autotuner compiles combinations)")
    #     benchmark.run(print_data=True, show_plots=False)
        
    #     print("\n--- Revealing the Best Configuration ---")
        # To see what the autotuner actually picked, we do a dummy run to trigger compilation
        #dummy_q = torch.randn(2, 4, 1024, 64, device='cuda', dtype=torch.float16)
        #triton_flash_attention(dummy_q, dummy_q, dummy_q)
        # scaled_dot_product_attention(q, k, v)
        
        # # Now we can inspect the best config chosen for seq_len=1024
        # best = fused_flash_attention_kernel.best_config

        # print_full_autotune_report(batch=1, num_heads=8, seq_len=256, head_dim=64)
        # print(f"For seq_len=1024, the autotuner chose:")
        # print(f"  BLOCK_M: {best.kwargs['BLOCK_M']}")
        # print(f"  BLOCK_N: {best.kwargs['BLOCK_N']}")
        # print(f"  num_warps: {best.num_warps}")
        # print(f"  num_stages: {best.num_stages}")