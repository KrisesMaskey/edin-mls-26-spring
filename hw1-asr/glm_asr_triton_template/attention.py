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


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)

    *** TODO: Implement this kernel ***
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention score computation
    # ============================================================================
    #
    # Step 1: Load query vector for this position
    # Step 2: Load all keys for this batch_head
    # Step 3: Compute dot-product scores and scale
    # Step 4: Store scores

    # YOUR CODE HERE
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    )

    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    scores = tl.sum(k * q[None, :], axis=1) * scale

    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=offs_k < seq_k,
    )


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads * seq_q,)
    """
    row = tl.program_id(0)

    # ============================================================================
    # TODO: Implement softmax
    # ============================================================================
    #
    # Step 1: Load scores row with masking
    # Step 2: Subtract max for stability
    # Step 3: Compute exp and normalize
    # Step 4: Store back

    # YOUR CODE HERE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k
    row_ptr = scores_ptr + row * stride_s

    scores = tl.load(row_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
    scores = scores - tl.max(scores, axis=0)
    exp_scores = tl.math.exp(scores)
    softmax = exp_scores / tl.sum(exp_scores, axis=0)
    tl.store(row_ptr + offs, softmax, mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr,
    v_ptr,
    output_ptr,
    seq_k,
    head_dim,
    stride_w0,
    stride_w1,
    stride_w2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute attention output: attn_weights @ V
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention output computation
    # ============================================================================
    #
    # Step 1: Load attention weights for this query
    # Step 2: Load all values for this batch_head
    # Step 3: Compute weighted sum
    # Step 4: Store output

    # YOUR CODE HERE
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    w = tl.load(
        attn_ptr
        + pid_bh * stride_w0
        + pid_q * stride_w1
        + offs_k * stride_w2,
        mask=offs_k < seq_k,
        other=0.0,
    )

    v = tl.load(
        v_ptr
        + pid_bh * stride_v0
        + offs_k[:, None] * stride_v1
        + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    out = tl.sum(v * w[:, None], axis=0)

    tl.store(
        output_ptr
        + pid_bh * stride_o0
        + pid_q * stride_o1
        + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )


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
# FLASH ATTENTION KERNEL IMPLEMENTATION
# ============================================================================
@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr,
    mask_ptr,           # additive float mask (BH, seq_q, seq_k); only read when has_mask=True
    out_ptr,
    seq_q, seq_k, head_dim,
    scale,
    stride_qbh, stride_qq, stride_qd,
    stride_kbh, stride_kk, stride_kd,
    stride_vbh, stride_vk, stride_vd,
    stride_mbh, stride_mq, stride_mk,
    stride_obh, stride_oq, stride_od,
    is_causal:  tl.constexpr,
    has_mask:   tl.constexpr,
    BLOCK_Q:    tl.constexpr,   # query tile rows
    BLOCK_KV:   tl.constexpr,   # key/value tile rows
    BLOCK_D:    tl.constexpr,   # head_dim (padded to power-of-2)
):
    """
    Flash Attention v1 — fused tiled attention without materialising the full
    (seq_q × seq_k) score matrix.
 
    Grid: (batch * num_heads,  triton.cdiv(seq_q, BLOCK_Q))
 
    Each program handles BLOCK_Q query positions and iterates over all
    KV tiles of size BLOCK_KV, maintaining the online softmax accumulators
    (m, l, O) entirely in registers.
 
    Algorithm (per program)
    ───────────────────────
    m_i  ← −∞              running row-max      (BLOCK_Q,)
    l_i  ← 0               running row-sum       (BLOCK_Q,)
    O_i  ← 0               running output        (BLOCK_Q, BLOCK_D)
 
    for each KV tile j:
        S_ij = Q_i @ K_j^T * scale              (BLOCK_Q, BLOCK_KV)
        [apply causal / additive masks to S_ij]
        m_ij   = rowmax(S_ij)
        m_new  = max(m_i, m_ij)
        alpha  = exp(m_i  − m_new)              (rescale old state)
        P_ij   = exp(S_ij − m_new[:, None])
        l_i    = alpha * l_i + rowsum(P_ij)
        O_i    = alpha[:, None] * O_i + P_ij @ V_j
        m_i    = m_new
 
    O_i  /= l_i[:, None]                        (final normalisation)
 
    Memory savings vs the three-kernel path
    ────────────────────────────────────────
    Three-kernel: allocates scores (BH, seq_q, seq_k_pad) — O(seq²) HBM
    Flash:        only allocates output (BH, seq_q, head_dim) — O(seq·d) HBM
    Removes the 256-token sequence cap imposed by the old single-tile kernels.
    """
    pid_bh = tl.program_id(0)   # batch * head index
    pid_q  = tl.program_id(1)   # query tile index
 
    q_start = pid_q * BLOCK_Q
    offs_q  = q_start + tl.arange(0, BLOCK_Q)   # absolute query positions
    offs_d  = tl.arange(0, BLOCK_D)
 
    # ── Load Q tile: (BLOCK_Q, BLOCK_D) ────────────────────────────────────
    q = tl.load(
        q_ptr
        + pid_bh * stride_qbh
        + offs_q[:, None] * stride_qq
        + offs_d[None, :] * stride_qd,
        mask=(offs_q[:, None] < seq_q) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
 
    # ── Initialise online-softmax accumulators ──────────────────────────────
    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
 
    # ── Iterate over KV tiles ───────────────────────────────────────────────
    for kv_start in range(0, seq_k, BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)
 
        # Load K tile (BLOCK_KV, BLOCK_D), transpose to (BLOCK_D, BLOCK_KV)
        k = tl.load(
            k_ptr
            + pid_bh * stride_kbh
            + offs_kv[:, None] * stride_kk
            + offs_d[None, :] * stride_kd,
            mask=(offs_kv[:, None] < seq_k) & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
 
        # S = Q @ K^T * scale  →  (BLOCK_Q, BLOCK_KV)
        s = tl.dot(q, tl.trans(k), allow_tf32=True) * scale
 
        # ── Out-of-bounds KV positions → −inf (excluded from softmax) ────
        s = tl.where(offs_kv[None, :] < seq_k, s, float("-inf"))
 
        # ── Causal mask ───────────────────────────────────────────────────
        # Keep position (i, j) only when j <= i  (lower triangular)
        if is_causal:
            s = tl.where(offs_kv[None, :] <= offs_q[:, None], s, float("-inf"))
 
        # ── Additive attention mask ───────────────────────────────────────
        if has_mask:
            m_tile = tl.load(
                mask_ptr
                + pid_bh * stride_mbh
                + offs_q[:, None]  * stride_mq
                + offs_kv[None, :] * stride_mk,
                mask=(offs_q[:, None] < seq_q) & (offs_kv[None, :] < seq_k),
                other=0.0,
            ).to(tl.float32)
            s = s + m_tile
 
        # ── Online softmax update ─────────────────────────────────────────
        m_ij  = tl.max(s, axis=1)                   # (BLOCK_Q,)
        m_new = tl.maximum(m_i, m_ij)               # elementwise max
 
        alpha = tl.exp(m_i  - m_new)                # rescale factor for old state
        p     = tl.exp(s    - m_new[:, None])        # (BLOCK_Q, BLOCK_KV)
 
        l_i   = alpha * l_i + tl.sum(p, axis=1)
        o_i   = alpha[:, None] * o_i
 
        # ── Load V tile (BLOCK_KV, BLOCK_D) and accumulate ───────────────
        v = tl.load(
            v_ptr
            + pid_bh * stride_vbh
            + offs_kv[:, None] * stride_vk
            + offs_d[None, :] * stride_vd,
            mask=(offs_kv[:, None] < seq_k) & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
 
        o_i = o_i + tl.dot(p, v, allow_tf32=True)   # (BLOCK_Q, BLOCK_D)
        m_i = m_new
 
    # ── Final normalisation ─────────────────────────────────────────────────
    o_i = o_i / l_i[:, None]
 
    # ── Store output ────────────────────────────────────────────────────────
    tl.store(
        out_ptr
        + pid_bh * stride_obh
        + offs_q[:, None]  * stride_oq
        + offs_d[None, :] * stride_od,
        o_i,
        mask=(offs_q[:, None] < seq_q) & (offs_d[None, :] < head_dim),
    )
 

# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""
    
    # Set True  → use flash_attention_kernel (single tiled kernel, no O(seq²) allocation)
    # Set False → use the original three-kernel path (scores → softmax → weighted-V)
    FLASH: bool = True

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
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256

# Flash attention only constrains head_dim (must fit as a register tile).
# Sequence length is unconstrained — the kernel tiles over it.
MAX_FLASH_HEAD_DIM = 128   # covers head_dim 64 (audio) and 128 (text) in this model

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention using Triton kernels.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
    
    # ── Flash Attention path ────────────────────────────────────────────────
    # Single tiled kernel; never allocates the (seq_q × seq_k) scores matrix.
    # Removes the old 256-token sequence cap — seq_k can be arbitrarily long.
    # Only constraint: head_dim must fit in a register tile (≤ MAX_FLASH_HEAD_DIM).
    head_dim_padded = next_power_of_two(head_dim)
 
    if (
        MultiHeadAttention.FLASH
        and q.is_cuda
        and head_dim_padded <= MAX_FLASH_HEAD_DIM
    ):
        BH     = batch * num_heads
        q_flat = q.reshape(BH, seq_q, head_dim).to(torch.float32).contiguous()
        k_flat = k.reshape(BH, seq_k, head_dim).to(torch.float32).contiguous()
        v_flat = v.reshape(BH, seq_k, head_dim).to(torch.float32).contiguous()
        output = torch.zeros(
            (BH, seq_q, head_dim_padded), dtype=torch.float32, device=q.device
        )
 
        # ── Prepare attention mask ──────────────────────────────────────────
        # Reshape to (BH, seq_q, seq_k) so the kernel can tile it.
        # When mask is 4D (batch, 1, seq_q, seq_k), broadcast across heads.
        has_mask = attention_mask is not None
        if has_mask:
            if attention_mask.ndim == 4:
                # (batch, 1, seq_q, seq_k) → (batch, num_heads, seq_q, seq_k)
                # → (BH, seq_q, seq_k)
                mask_flat = (
                    attention_mask
                    .expand(batch, num_heads, seq_q, seq_k)
                    .reshape(BH, seq_q, seq_k)
                    .to(torch.float32)
                    .contiguous()
                )
            else:
                mask_flat = attention_mask.reshape(BH, seq_q, seq_k).to(torch.float32).contiguous()
            mask_ptr        = mask_flat
            stride_mbh      = mask_flat.stride(0)
            stride_mq       = mask_flat.stride(1)
            stride_mk       = mask_flat.stride(2)
        else:
            # Dummy pointer — kernel won't read it when has_mask=False
            mask_ptr   = q_flat
            stride_mbh = stride_mq = stride_mk = 0
 
        # ── Tile sizes ──────────────────────────────────────────────────────
        # BLOCK_Q / BLOCK_KV must be ≥ 16 for tl.dot; use 16 as the minimum.
        # Larger tiles improve tensor-core utilisation on long sequences.
        BLOCK_Q  = max(16, min(64, next_power_of_two(seq_q)))
        BLOCK_KV = max(16, min(64, next_power_of_two(seq_k)))
 
        grid = (BH, triton.cdiv(seq_q, BLOCK_Q))
        flash_attention_kernel[grid](
            q_flat, k_flat, v_flat,
            mask_ptr,
            output,
            seq_q, seq_k, head_dim,
            float(scale),
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            stride_mbh, stride_mq, stride_mk,
            output.stride(0), output.stride(1), output.stride(2),
            is_causal=is_causal,
            has_mask=has_mask,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
            BLOCK_D=head_dim_padded,
        )
 
        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]
 
        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)
 
    # ── Original three-kernel path (fallback when FLASH=False or CPU) ───────
    seq_k_padded = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    use_triton = (
        q.is_cuda
        and seq_k_padded <= MAX_ATTENTION_DIM
        and head_dim_padded <= MAX_ATTENTION_DIM
    )

    if use_triton:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32)
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)

        if seq_k_padded != seq_k or head_dim_padded != head_dim:
            k_padded = torch.zeros(
                (batch * num_heads, seq_k_padded, head_dim_padded),
                dtype=torch.float32,
                device=q.device,
            )
            v_padded = torch.zeros_like(k_padded)
            q_padded = torch.zeros(
                (batch * num_heads, seq_q, head_dim_padded),
                dtype=torch.float32,
                device=q.device,
            )
            k_padded[:, :seq_k, :head_dim] = k_flat
            v_padded[:, :seq_k, :head_dim] = v_flat
            q_padded[:, :, :head_dim] = q_flat
            k_flat = k_padded
            v_flat = v_padded
            q_flat = q_padded

        scores = torch.empty(
            (batch * num_heads, seq_q, seq_k_padded),
            dtype=torch.float32,
            device=q.device,
        )
        output = torch.empty(
            (batch * num_heads, seq_q, head_dim_padded),
            dtype=torch.float32,
            device=q.device,
        )

        grid = (batch * num_heads, seq_q)
        attention_scores_kernel[grid](
            q_flat,
            k_flat,
            scores,
            float(scale),
            seq_k_padded,
            head_dim_padded,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if seq_k_padded != seq_k:
            scores[:, :, seq_k:] = -1e9

        if is_causal:
            mask = torch.triu(
                torch.ones((seq_q, seq_k_padded), dtype=torch.float32, device=q.device),
                diagonal=1,
            ) * -1e9
            scores = scores + mask[None, :, :]

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                attention_mask = attention_mask.reshape(
                    batch * num_heads, seq_q, seq_k
                )
            if seq_k_padded != seq_k:
                mask_padded = torch.zeros(
                    (batch * num_heads, seq_q, seq_k_padded),
                    dtype=torch.float32,
                    device=q.device,
                )
                mask_padded[:, :, :seq_k] = attention_mask
                mask_padded[:, :, seq_k:] = -1e9
                attention_mask = mask_padded
            scores = scores + attention_mask

        scores_2d = scores.reshape(batch * num_heads * seq_q, seq_k_padded)
        block = seq_k_padded
        softmax_inplace_kernel[(scores_2d.shape[0],)](
            scores_2d, scores_2d.stride(0), seq_k_padded, BLOCK_SIZE=block
        )
        scores = scores_2d.reshape(batch * num_heads, seq_q, seq_k_padded)

        attention_output_kernel[grid](
            scores,
            v_flat,
            output,
            seq_k_padded,
            head_dim_padded,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)

    return output.to(q.dtype)


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Pure-PyTorch reference implementation used to validate kernel outputs."""
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = torch.einsum("bnqd,bnkd->bnqk", q.float(), k.float()) * scale
    if is_causal:
        seq_q, seq_k = q.shape[2], k.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_q, seq_k, device=q.device), diagonal=1
        ) * -1e9
        scores = scores + causal_mask[None, None]
    if attention_mask is not None:
        scores = scores + attention_mask.float()
    scores = scores - scores.amax(dim=-1, keepdim=True)
    w = torch.exp(scores)
    w = w / w.sum(dim=-1, keepdim=True)
    return torch.einsum("bnqk,bnkd->bnqd", w, v.float()).to(q.dtype)
 
 
def _check(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float = 1e-3) -> bool:
    """Compare two tensors and print a PASS / FAIL line."""
    max_err = float((got.float() - ref.float()).abs().max())
    ok = max_err <= atol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}  |  max_err={max_err:.2e}  (tol={atol:.0e})")
    return ok
    

if __name__ == "__main__":
    print("=" * 60)
    print("Triton Attention — unit tests")
    print("=" * 60)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — Triton kernels will not run; "
              "only PyTorch fallback paths are exercised.")
 
    passed = 0
    failed = 0
 
    def record(ok: bool):
        global passed, failed
        if ok:
            passed += 1
        else:
            failed += 1
 
    torch.manual_seed(42)
 
    # ── Common small tensors used by several tests ───────────────────────────
    B, H, S, D = 2, 4, 16, 64
    q  = torch.randn(B, H, S, D, device=device)
    k  = torch.randn(B, H, S, D, device=device)
    v  = torch.randn(B, H, S, D, device=device)
 
    # =========================================================================
    # Section 1 — Original three-kernel path (FLASH=False)
    # Keeps the original tests running so regressions are caught.
    # =========================================================================
    print("\n── Section 1: original three-kernel path (FLASH=False) ──")
    MultiHeadAttention.FLASH = False
 
    print("\nBasic attention (shape check):")
    out = scaled_dot_product_attention(q, k, v)
    ok = out.shape == (B, H, S, D)
    print(f"  [{'PASS' if ok else 'FAIL'}] output shape {out.shape} == {(B, H, S, D)}")
    record(ok)
 
    print("\nCausal attention (shape check):")
    out_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    ok = out_causal.shape == (B, H, S, D)
    print(f"  [{'PASS' if ok else 'FAIL'}] output shape {out_causal.shape} == {(B, H, S, D)}")
    record(ok)
 
    print("\nWith attention mask (shape check):")
    mask_4d = torch.zeros(B, H, S, S, device=device)
    mask_4d[:, :, :, S // 2:] = -1e9
    out_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask_4d)
    ok = out_masked.shape == (B, H, S, D)
    print(f"  [{'PASS' if ok else 'FAIL'}] output shape {out_masked.shape} == {(B, H, S, D)}")
    record(ok)
 
    print("\nGrouped Query Attention — GQA (shape check):")
    KVH = 2
    k_gqa = torch.randn(B, KVH, S, D, device=device)
    v_gqa = torch.randn(B, KVH, S, D, device=device)
    attn_obj = MultiHeadAttention(hidden_size=H * D, num_heads=H, num_kv_heads=KVH)
    out_gqa = attn_obj(q, k_gqa, v_gqa)
    ok = out_gqa.shape == (B, H, S, D)
    print(f"  [{'PASS' if ok else 'FAIL'}] output shape {out_gqa.shape} == {(B, H, S, D)}")
    record(ok)
 
    print("\nOutput statistics (three-kernel):")
    print(f"  Mean={float(out.mean()):.4f}  Std={float(out.std()):.4f}"
          f"  Min={float(out.min()):.4f}  Max={float(out.max()):.4f}")
 
    # =========================================================================
    # Section 2 — Flash Attention correctness vs PyTorch reference
    # Each test: run with FLASH=True, run reference, compare numerically.
    # =========================================================================
    print("\n── Section 2: flash attention correctness vs PyTorch reference ──")
    MultiHeadAttention.FLASH = True
 
    # ── 2.1  Basic (no mask, no causal) ─────────────────────────────────────
    print("\n2.1  Basic attention:")
    flash_out  = scaled_dot_product_attention(q, k, v)
    ref_out    = _reference_attention(q, k, v)
    record(_check("output values match reference", flash_out, ref_out))
    ok = flash_out.shape == ref_out.shape
    print(f"  [{'PASS' if ok else 'FAIL'}] shape {flash_out.shape} == {ref_out.shape}")
    record(ok)
 
    # ── 2.2  Causal masking ──────────────────────────────────────────────────
    print("\n2.2  Causal attention:")
    flash_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    ref_causal   = _reference_attention(q, k, v, is_causal=True)
    record(_check("causal output matches reference", flash_causal, ref_causal))
 
    # Extra: verify that future tokens are not attended to.
    # For position 0, only position 0 should contribute.
    # We can check indirectly: output at position 0 should equal v[:,:,0,:]
    # weighted by softmax of a single score — i.e. output[:,: ,0,:] should be
    # very close to v[:,:,0,:] (it equals exactly when q[0]*k[0] is the only
    # attended position, which softmax normalises to 1.0).
    # Instead just confirm position 1 differs from the non-causal output
    # (if causality is broken they will be equal).
    causal_differs = not torch.allclose(
        flash_causal[:, :, 1, :].float(),
        flash_out[:, :, 1, :].float(),
        atol=1e-4,
    )
    ok = causal_differs
    print(f"  [{'PASS' if ok else 'FAIL'}] causal output differs from non-causal (mask is active)")
    record(ok)
 
    # ── 2.3  Additive attention mask ─────────────────────────────────────────
    print("\n2.3  Additive attention mask (4D, blocks second half of sequence):")
    flash_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask_4d)
    ref_masked   = _reference_attention(q, k, v, attention_mask=mask_4d)
    record(_check("masked output matches reference", flash_masked, ref_masked))
 
    # Verify mask is actually active: masked output should differ from unmasked
    mask_differs = not torch.allclose(
        flash_masked.float(), flash_out.float(), atol=1e-4
    )
    ok = mask_differs
    print(f"  [{'PASS' if ok else 'FAIL'}] masked output differs from unmasked (mask is active)")
    record(ok)
 
    # ── 2.4  GQA — key/value head expansion ─────────────────────────────────
    print("\n2.4  Grouped Query Attention (GQA, num_kv_heads=2):")
    MultiHeadAttention.FLASH = True
    flash_gqa = attn_obj(q, k_gqa, v_gqa)
    MultiHeadAttention.FLASH = False
    ref_gqa   = attn_obj(q, k_gqa, v_gqa)
    MultiHeadAttention.FLASH = True
    record(_check("GQA flash matches three-kernel", flash_gqa, ref_gqa))
    ok = flash_gqa.shape == (B, H, S, D)
    print(f"  [{'PASS' if ok else 'FAIL'}] GQA output shape {flash_gqa.shape} == {(B, H, S, D)}")
    record(ok)
 
    # ── 2.5  head_dim = 128 (text decoder size) ──────────────────────────────
    print("\n2.5  head_dim=128 (text decoder):")
    D2 = 128
    q2 = torch.randn(B, H, S, D2, device=device)
    k2 = torch.randn(B, H, S, D2, device=device)
    v2 = torch.randn(B, H, S, D2, device=device)
    flash_d128 = scaled_dot_product_attention(q2, k2, v2)
    ref_d128   = _reference_attention(q2, k2, v2)
    record(_check("head_dim=128 output matches reference", flash_d128, ref_d128))
 
    # ── 2.6  Non-power-of-2 sequence length ──────────────────────────────────
    print("\n2.6  Non-power-of-2 sequence length (seq_len=50):")
    S3 = 50
    q3 = torch.randn(B, H, S3, D, device=device)
    k3 = torch.randn(B, H, S3, D, device=device)
    v3 = torch.randn(B, H, S3, D, device=device)
    flash_s50 = scaled_dot_product_attention(q3, k3, v3)
    ref_s50   = _reference_attention(q3, k3, v3)
    record(_check("seq_len=50 output matches reference", flash_s50, ref_s50))
 
    # ── 2.7  Cross-attention (seq_q != seq_k) ────────────────────────────────
    print("\n2.7  Cross-attention (seq_q=8, seq_k=32):")
    SQ, SK = 8, 32
    qc = torch.randn(B, H, SQ, D, device=device)
    kc = torch.randn(B, H, SK, D, device=device)
    vc = torch.randn(B, H, SK, D, device=device)
    flash_cross = scaled_dot_product_attention(qc, kc, vc)
    ref_cross   = _reference_attention(qc, kc, vc)
    ok_shape = flash_cross.shape == (B, H, SQ, D)
    print(f"  [{'PASS' if ok_shape else 'FAIL'}] output shape {flash_cross.shape} == {(B, H, SQ, D)}")
    record(ok_shape)
    record(_check("cross-attention output matches reference", flash_cross, ref_cross))
 
    # ── 2.8  Long sequence — beyond the old 256-token cap ────────────────────
    print("\n2.8  Long sequence (seq_len=512, beyond old 256-token cap):")
    SL = 512
    ql = torch.randn(1, H, SL, D, device=device)
    kl = torch.randn(1, H, SL, D, device=device)
    vl = torch.randn(1, H, SL, D, device=device)
    MultiHeadAttention.FLASH = True
    flash_long = scaled_dot_product_attention(ql, kl, vl)
    ref_long   = _reference_attention(ql, kl, vl)
    ok_shape = flash_long.shape == (1, H, SL, D)
    print(f"  [{'PASS' if ok_shape else 'FAIL'}] output shape {flash_long.shape} == {(1, H, SL, D)}")
    record(ok_shape)
    record(_check("long-sequence output matches reference", flash_long, ref_long))
 
    # Confirm old path would have fallen back (seq_k_padded=512 > MAX_ATTENTION_DIM=256)
    MultiHeadAttention.FLASH = False
    old_long = scaled_dot_product_attention(ql, kl, vl)
    MultiHeadAttention.FLASH = True
    ok = old_long.shape == (1, H, SL, D)
    print(f"  [{'PASS' if ok else 'FAIL'}] old path also produces correct shape (PyTorch fallback)")
    record(ok)
 
    # ── 2.9  Softmax normalisation — rows sum to 1 ────────────────────────────
    # We cannot inspect the internal attention weights directly, but we can
    # construct a V = identity: output[:,h,q,:] should equal the q-th row of
    # the attention weight matrix, which must sum to 1 per head.
    # Easier proxy: use V = ones → output should be all-ones (sum of weights = 1).
    print("\n2.9  Softmax normalisation (V=ones → output must be all-ones):")
    v_ones = torch.ones(B, H, S, D, device=device)
    flash_norm = scaled_dot_product_attention(q, k, v_ones)
    ref_ones   = torch.ones_like(flash_norm)
    record(_check("rows sum to 1 (output ≈ 1.0 when V=ones)", flash_norm, ref_ones, atol=1e-3))
 
    # ── 2.10  Flash == three-kernel for standard config ───────────────────────
    print("\n2.10 Flash output == three-kernel output (same inputs, seq_len=16):")
    MultiHeadAttention.FLASH = False
    three_k = scaled_dot_product_attention(q, k, v)
    MultiHeadAttention.FLASH = True
    flash_k  = scaled_dot_product_attention(q, k, v)
    record(_check("flash == three-kernel", flash_k, three_k))
 
    # =========================================================================
    # Section 3 — Model-relevant configurations
    # =========================================================================
    print("\n── Section 3: model-relevant configurations ──")
    MultiHeadAttention.FLASH = True
 
    print("\n3.1  Audio encoder (hidden=1280, heads=20, head_dim=64, seq=750):")
    AH, AD, AS = 20, 64, 750
    qa = torch.randn(1, AH, AS, AD, device=device)
    ka = torch.randn(1, AH, AS, AD, device=device)
    va = torch.randn(1, AH, AS, AD, device=device)
    out_audio = scaled_dot_product_attention(qa, ka, va)
    ok = out_audio.shape == (1, AH, AS, AD)
    print(f"  [{'PASS' if ok else 'FAIL'}] output shape {out_audio.shape} == {(1, AH, AS, AD)}")
    record(ok)
    ref_audio = _reference_attention(qa, ka, va)
    record(_check("audio encoder output matches reference", out_audio, ref_audio))
 
    print("\n3.2  Text decoder (hidden=3584, heads=28, kv_heads=4, head_dim=128, seq=32):")
    TH, TKV, TD, TS = 28, 4, 128, 32
    qt = torch.randn(1, TH,  TS, TD, device=device)
    kt = torch.randn(1, TKV, TS, TD, device=device)
    vt = torch.randn(1, TKV, TS, TD, device=device)
    text_attn = MultiHeadAttention(
        hidden_size=TH * TD, num_heads=TH, num_kv_heads=TKV, head_dim=TD
    )
    out_text = text_attn(qt, kt, vt, is_causal=True)
    ok = out_text.shape == (1, TH, TS, TD)
    print(f"  [{'PASS' if ok else 'FAIL'}] output shape {out_text.shape} == {(1, TH, TS, TD)}")
    record(ok)
     # Replicate KV heads to match Q heads (mirrors MultiHeadAttention._expand_kv)
    repeats = TH // TKV
    kt_exp = kt.repeat_interleave(repeats, dim=1)
    vt_exp = vt.repeat_interleave(repeats, dim=1)
    ref_text = _reference_attention(qt, kt_exp, vt_exp, is_causal=True)
    record(_check("text decoder output matches reference", out_text, ref_text))
 
    # =========================================================================
    # Summary
    # =========================================================================
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  — all tests passed ✓")
    print("=" * 60)

    """print("Testing Triton Attention...")

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

    print("\nTriton Attention working!")"""
