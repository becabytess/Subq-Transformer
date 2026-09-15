"""
OpenAI Triton Accelerated Kernel for Sub-Quadratic (SubQ) Multi-Hop Attention.

Features:
- Fused forward pass with online softmax and reciprocal key/value gathering.
- Two-phase backward pass:
    Phase 1: Computes dQ and writes dS globally with guaranteed GPU-wide barrier.
    Phase 2: Computes dK and dV using reciprocal offsets reading clean dS with zero atomic locks.
- Automatic fallback to vectorized PyTorch autograd if Triton or CUDA is unavailable.
"""

import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _subq_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Offsets_ptr, Biases_ptr, Out_ptr, P_ptr,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_ob, stride_oh, stride_ol, stride_od,
        stride_pb, stride_ph, stride_pl, stride_pk,
        scale,
        B, H, L,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_OFFSETS: tl.constexpr,
    ):
        start_m = tl.program_id(0) * BLOCK_M
        off_hz = tl.program_id(1)
        batch_id = off_hz // H
        head_id = off_hz % H

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = Q_ptr + (batch_id * stride_qb + head_id * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd)
        mask_m = offs_m < L
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Pass 1: Compute scores and find max for online softmax
        for k_idx in range(NUM_OFFSETS):
            offset_val = tl.load(Offsets_ptr + k_idx)
            bias_val = tl.load(Biases_ptr + k_idx)

            target_pos = offs_m - offset_val
            target_mask = mask_m & (target_pos >= 0) & (target_pos < L)
            safe_target = tl.where(target_mask, target_pos, 0)

            k_ptrs = K_ptr + (batch_id * stride_kb + head_id * stride_kh + safe_target[:, None] * stride_kl + offs_d[None, :] * stride_kd)
            v_ptrs = V_ptr + (batch_id * stride_vb + head_id * stride_vh + safe_target[:, None] * stride_vl + offs_d[None, :] * stride_vd)

            k = tl.load(k_ptrs, mask=target_mask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=target_mask[:, None], other=0.0)

            score = tl.sum(q * k, axis=1) * scale + bias_val
            score = tl.where(target_mask, score, -1e9)

            m_new = tl.maximum(m_i, score)
            alpha = tl.exp(m_i - m_new)
            p = tl.where(target_mask, tl.exp(score - m_new), 0.0)

            acc = acc * alpha[:, None] + p[:, None] * v
            l_i = l_i * alpha + p
            m_i = m_new

        # Normalize output
        acc = acc / (l_i[:, None] + 1e-6)
        out_ptrs = Out_ptr + (batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od)
        tl.store(out_ptrs, acc, mask=mask_m[:, None])

        # Pass 2: Store exact normalized weights P for backward pass
        for k_idx in range(NUM_OFFSETS):
            offset_val = tl.load(Offsets_ptr + k_idx)
            bias_val = tl.load(Biases_ptr + k_idx)
            target_pos = offs_m - offset_val
            target_mask = mask_m & (target_pos >= 0) & (target_pos < L)
            safe_target = tl.where(target_mask, target_pos, 0)

            k_ptrs = K_ptr + (batch_id * stride_kb + head_id * stride_kh + safe_target[:, None] * stride_kl + offs_d[None, :] * stride_kd)
            k = tl.load(k_ptrs, mask=target_mask[:, None], other=0.0)
            score = tl.sum(q * k, axis=1) * scale + bias_val
            score = tl.where(target_mask, score, -1e9)
            p = tl.where(target_mask, tl.exp(score - m_i) / (l_i + 1e-6), 0.0)

            p_ptrs = P_ptr + (batch_id * stride_pb + head_id * stride_ph + offs_m * stride_pl + k_idx * stride_pk)
            tl.store(p_ptrs, p, mask=mask_m)

    @triton.jit
    def _subq_bwd_dq_kernel(
        dOut_ptr, Q_ptr, K_ptr, V_ptr, P_ptr, Out_ptr, Offsets_ptr,
        dQ_ptr, dS_ptr,
        stride_ob, stride_oh, stride_ol, stride_od,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_pb, stride_ph, stride_pl, stride_pk,
        stride_dqb, stride_dqh, stride_dql, stride_dqd,
        stride_dsb, stride_dsh, stride_dsl, stride_dsk,
        scale,
        B, H, L,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_OFFSETS: tl.constexpr,
    ):
        start_m = tl.program_id(0) * BLOCK_M
        off_hz = tl.program_id(1)
        batch_id = off_hz // H
        head_id = off_hz % H

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < L

        dout_ptrs = dOut_ptr + (batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od)
        out_ptrs = Out_ptr + (batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od)
        dout = tl.load(dout_ptrs, mask=mask_m[:, None], other=0.0)
        out_val = tl.load(out_ptrs, mask=mask_m[:, None], other=0.0)

        D_i = tl.sum(dout * out_val, axis=1)

        dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for k_idx in range(NUM_OFFSETS):
            offset_val = tl.load(Offsets_ptr + k_idx)
            target_pos = offs_m - offset_val
            target_mask = mask_m & (target_pos >= 0) & (target_pos < L)
            safe_target = tl.where(target_mask, target_pos, 0)

            p_ptrs = P_ptr + (batch_id * stride_pb + head_id * stride_ph + offs_m * stride_pl + k_idx * stride_pk)
            p = tl.load(p_ptrs, mask=mask_m, other=0.0)

            v_ptrs = V_ptr + (batch_id * stride_vb + head_id * stride_vh + safe_target[:, None] * stride_vl + offs_d[None, :] * stride_vd)
            k_ptrs = K_ptr + (batch_id * stride_kb + head_id * stride_kh + safe_target[:, None] * stride_kl + offs_d[None, :] * stride_kd)
            v = tl.load(v_ptrs, mask=target_mask[:, None], other=0.0)
            k = tl.load(k_ptrs, mask=target_mask[:, None], other=0.0)

            dP = tl.sum(dout * v, axis=1)
            dS = p * (dP - D_i) * scale
            dS = tl.where(target_mask, dS, 0.0)

            ds_ptrs = dS_ptr + (batch_id * stride_dsb + head_id * stride_dsh + offs_m * stride_dsl + k_idx * stride_dsk)
            tl.store(ds_ptrs, dS, mask=mask_m)

            dq_acc += dS[:, None] * k

        dq_ptrs = dQ_ptr + (batch_id * stride_dqb + head_id * stride_dqh + offs_m[:, None] * stride_dql + offs_d[None, :] * stride_dqd)
        tl.store(dq_ptrs, dq_acc, mask=mask_m[:, None])

    @triton.jit
    def _subq_bwd_dkv_kernel(
        dOut_ptr, Q_ptr, P_ptr, dS_ptr, Offsets_ptr,
        dK_ptr, dV_ptr,
        stride_ob, stride_oh, stride_ol, stride_od,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_pb, stride_ph, stride_pl, stride_pk,
        stride_dsb, stride_dsh, stride_dsl, stride_dsk,
        stride_dkb, stride_dkh, stride_dkl, stride_dkd,
        stride_dvb, stride_dvh, stride_dvl, stride_dvd,
        B, H, L,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_OFFSETS: tl.constexpr,
    ):
        start_m = tl.program_id(0) * BLOCK_M
        off_hz = tl.program_id(1)
        batch_id = off_hz // H
        head_id = off_hz % H

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < L

        dk_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for k_idx in range(NUM_OFFSETS):
            offset_val = tl.load(Offsets_ptr + k_idx)
            recip_pos = offs_m + offset_val
            recip_mask = mask_m & (recip_pos >= 0) & (recip_pos < L)
            safe_recip = tl.where(recip_mask, recip_pos, 0)

            ds_recip_ptrs = dS_ptr + (batch_id * stride_dsb + head_id * stride_dsh + safe_recip * stride_dsl + k_idx * stride_dsk)
            ds_recip = tl.load(ds_recip_ptrs, mask=recip_mask, other=0.0)

            p_recip_ptrs = P_ptr + (batch_id * stride_pb + head_id * stride_ph + safe_recip * stride_pl + k_idx * stride_pk)
            p_recip = tl.load(p_recip_ptrs, mask=recip_mask, other=0.0)

            q_recip_ptrs = Q_ptr + (batch_id * stride_qb + head_id * stride_qh + safe_recip[:, None] * stride_ql + offs_d[None, :] * stride_qd)
            dout_recip_ptrs = dOut_ptr + (batch_id * stride_ob + head_id * stride_oh + safe_recip[:, None] * stride_ol + offs_d[None, :] * stride_od)
            q_recip = tl.load(q_recip_ptrs, mask=recip_mask[:, None], other=0.0)
            dout_recip = tl.load(dout_recip_ptrs, mask=recip_mask[:, None], other=0.0)

            dk_acc += ds_recip[:, None] * q_recip
            dv_acc += p_recip[:, None] * dout_recip

        dk_ptrs = dK_ptr + (batch_id * stride_dkb + head_id * stride_dkh + offs_m[:, None] * stride_dkl + offs_d[None, :] * stride_dkd)
        dv_ptrs = dV_ptr + (batch_id * stride_dvb + head_id * stride_dvh + offs_m[:, None] * stride_dvl + offs_d[None, :] * stride_dvd)
        tl.store(dk_ptrs, dk_acc, mask=mask_m[:, None])
        tl.store(dv_ptrs, dv_acc, mask=mask_m[:, None])

    class SubQTritonFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, K, V, offsets, biases):
            B, H, L, D = Q.shape
            scale = 1.0 / math.sqrt(D)
            NUM_OFFSETS = offsets.shape[0]

            out = torch.empty_like(Q)
            P = torch.empty((B, H, L, NUM_OFFSETS), device=Q.device, dtype=torch.float32)

            BLOCK_M = 16
            BLOCK_D = D
            grid = (triton.cdiv(L, BLOCK_M), B * H)

            _subq_fwd_kernel[grid](
                Q, K, V, offsets, biases, out, P,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                P.stride(0), P.stride(1), P.stride(2), P.stride(3),
                scale,
                B, H, L,
                BLOCK_M=BLOCK_M,
                BLOCK_D=BLOCK_D,
                NUM_OFFSETS=NUM_OFFSETS,
            )

            ctx.save_for_backward(Q, K, V, P, out, offsets, biases)
            ctx.scale = scale
            return out

        @staticmethod
        def backward(ctx, dOut):
            Q, K, V, P, out, offsets, biases = ctx.saved_tensors
            B, H, L, D = Q.shape
            scale = ctx.scale
            NUM_OFFSETS = offsets.shape[0]

            dOut = dOut.contiguous()
            dQ = torch.empty_like(Q)
            dK = torch.empty_like(K)
            dV = torch.empty_like(V)
            dS = torch.empty((B, H, L, NUM_OFFSETS), device=Q.device, dtype=torch.float32)

            BLOCK_M = 16
            BLOCK_D = D
            grid = (triton.cdiv(L, BLOCK_M), B * H)

            # Phase 1: Compute dQ and store dS globally across all positions
            _subq_bwd_dq_kernel[grid](
                dOut, Q, K, V, P, out, offsets,
                dQ, dS,
                dOut.stride(0), dOut.stride(1), dOut.stride(2), dOut.stride(3),
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                P.stride(0), P.stride(1), P.stride(2), P.stride(3),
                dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
                dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
                scale,
                B, H, L,
                BLOCK_M=BLOCK_M,
                BLOCK_D=BLOCK_D,
                NUM_OFFSETS=NUM_OFFSETS,
            )

            # Phase 2: Compute dK and dV using reciprocal offsets reading clean dS
            _subq_bwd_dkv_kernel[grid](
                dOut, Q, P, dS, offsets,
                dK, dV,
                dOut.stride(0), dOut.stride(1), dOut.stride(2), dOut.stride(3),
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                P.stride(0), P.stride(1), P.stride(2), P.stride(3),
                dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
                dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
                dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
                B, H, L,
                BLOCK_M=BLOCK_M,
                BLOCK_D=BLOCK_D,
                NUM_OFFSETS=NUM_OFFSETS,
            )

            dbiases = (dS / scale).sum(dim=(0, 1, 2))
            return dQ, dK, dV, None, dbiases
else:
    class SubQTritonFunction:
        @staticmethod
        def apply(*args, **kwargs):
            raise NotImplementedError("OpenAI Triton is not available in this environment.")


class FastSubQAttentionFunction(torch.autograd.Function):
    """Vectorized reciprocal autograd fallback when Triton is unavailable."""
    @staticmethod
    def forward(ctx, Q, K, V, bias, scale, clamped, valid_mask, clamped_rev, rev_valid):
        K_c = K[:, :, clamped, :]
        V_c = V[:, :, clamped, :]
        scores = torch.einsum('bhld,bhlkd->bhlk', Q, K_c) * scale + bias.view(1, 1, 1, -1)
        scores = scores.masked_fill(~valid_mask.view(1, 1, *valid_mask.shape), -1e9)
        weights = F.softmax(scores, dim=-1)
        weights_clean = torch.where(valid_mask.view(1, 1, *valid_mask.shape), weights, torch.zeros_like(weights))
        out = torch.einsum('bhlk,bhlkd->bhld', weights_clean, V_c)

        ctx.save_for_backward(Q, K, V, weights_clean, out, bias)
        ctx.scale = scale
        ctx.clamped = clamped
        ctx.valid_mask = valid_mask
        ctx.clamped_rev = clamped_rev
        ctx.rev_valid = rev_valid
        return out

    @staticmethod
    def backward(ctx, dOut):
        Q, K, V, weights_clean, out, bias = ctx.saved_tensors
        scale = ctx.scale
        clamped = ctx.clamped
        valid_mask = ctx.valid_mask
        clamped_rev = ctx.clamped_rev
        rev_valid = ctx.rev_valid
        K_total = clamped.shape[-1]

        K_c = K[:, :, clamped, :]
        V_c = V[:, :, clamped, :]

        D_i = (dOut * out).sum(dim=-1, keepdim=True)
        dP = torch.einsum('bhld,bhlkd->bhlk', dOut, V_c)
        dS = weights_clean * (dP - D_i) * scale
        dS = torch.where(valid_mask.view(1, 1, *valid_mask.shape), dS, torch.zeros_like(dS))

        dQ = torch.einsum('bhlk,bhlkd->bhld', dS, K_c)

        Q_rev = Q[:, :, clamped_rev, :]
        dS_rev = dS[:, :, clamped_rev, torch.arange(K_total).unsqueeze(0)]
        dS_rev = torch.where(rev_valid.view(1, 1, *rev_valid.shape), dS_rev, torch.zeros_like(dS_rev))
        dK = torch.einsum('bhlk,bhlkd->bhld', dS_rev, Q_rev)

        dOut_rev = dOut[:, :, clamped_rev, :]
        weights_rev = weights_clean[:, :, clamped_rev, torch.arange(K_total).unsqueeze(0)]
        weights_rev = torch.where(rev_valid.view(1, 1, *rev_valid.shape), weights_rev, torch.zeros_like(weights_rev))
        dV = torch.einsum('bhlk,bhlkd->bhld', weights_rev, dOut_rev)

        dbias = (weights_clean * (dP - D_i)).sum(dim=(0, 1, 2))
        return dQ, dK, dV, dbias, None, None, None, None, None


def subq_attention(Q, K, V, offsets, biases, scale=None):
    """
    High-level dispatch function for SubQ attention:
    Uses Triton if CUDA and Triton are available; otherwise falls back to vectorized reciprocal autograd.
    """
    B, H, L, D = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if HAS_TRITON and Q.is_cuda:
        return SubQTritonFunction.apply(Q, K, V, offsets.to(torch.int32), biases.float())
    else:
        pos = torch.arange(L, device=Q.device).unsqueeze(1)
        raw_targets = pos - offsets.unsqueeze(0).long()
        valid_mask = (raw_targets >= 0) & (raw_targets < L)
        clamped = torch.clamp(raw_targets, 0, L - 1)

        raw_rev_targets = pos + offsets.unsqueeze(0).long()
        rev_valid = (raw_rev_targets >= 0) & (raw_rev_targets < L)
        clamped_rev = torch.clamp(raw_rev_targets, 0, L - 1)

        return FastSubQAttentionFunction.apply(
            Q, K, V, biases, scale, clamped, valid_mask, clamped_rev, rev_valid
        )
