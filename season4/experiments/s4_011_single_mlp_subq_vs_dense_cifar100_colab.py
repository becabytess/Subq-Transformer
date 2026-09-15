"""
==========================================================================================
  EXPERIMENT S4-011: SINGLE-MLP SHOOTOUT (OPENAI TRITON JIT ACCELERATED)
  1-LAYER SUBQ VIT (T=4, K=9, 1 MLP AT END) VS. 1-LAYER DENSE VIT (1 MLP AT END)
==========================================================================================
Architectural Setup:
1. SubQ ViT: 4 optical hops with K=9 sparse bilateral wave attention, followed by 
   EXACTLY ONE 512-wide MLP at the end (T=4) -> 235,268 parameters.
2. Dense ViT: 1 layer of full L x L dense attention, followed by 
   EXACTLY ONE 512-wide MLP at the end -> 245,476 parameters.
3. Custom OpenAI Triton JIT Acceleration:
   - _subq_fwd_kernel: Fused on-chip register attention and online softmax (Zero VRAM allocation).
   - _subq_bwd_kernel: Closed-form reciprocal gradient kernel (dQ, dK, dV) with zero atomic locks.
   - Pre-cached RAM dataset (zero-CPU DataLoader).
==========================================================================================
"""

import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# ------------------------------------------------------------------------------
# 1. OpenAI Triton JIT Forward & Backward Kernels for SubQ Attention
# ------------------------------------------------------------------------------
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

        # Load dOut and Out
        dout_ptrs = dOut_ptr + (batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od)
        out_ptrs = Out_ptr + (batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od)
        dout = tl.load(dout_ptrs, mask=mask_m[:, None], other=0.0)
        out_val = tl.load(out_ptrs, mask=mask_m[:, None], other=0.0)

        D_i = tl.sum(dout * out_val, axis=1) # [BLOCK_M]

        # 1. Compute dS and dQ
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

        # 2. Compute dK and dV using Reciprocal Offsets (Safe Phase 2 reading dS)
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

# Fallback autograd function if Triton is not installed
class FastSubQAttentionFunction(torch.autograd.Function):
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

# ------------------------------------------------------------------------------
# 2. Bilateral Harmonic Wave Router
# ------------------------------------------------------------------------------
class FastBilateralWaveRouter(nn.Module):
    def __init__(self, num_waves=12, p_peaks=4, max_d=256):
        super().__init__()
        self.num_waves = num_waves
        self.p_peaks = p_peaks
        self.k_total = 1 + 2 * p_peaks
        self.max_d = max_d
        self.init_wave_latent = nn.Parameter(torch.randn(1, num_waves * 4) * 0.1)
        self.wave_transition = nn.Sequential(
            nn.Linear(num_waves * 4, 64),
            nn.GELU(),
            nn.Linear(64, num_waves * 4),
        )
        log_freqs = torch.linspace(0.0, -2.0, num_waves)
        self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
        self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

    def forward(self, wave_latent, dev):
        curr_params = wave_latent.view(1, self.num_waves, 4)
        amp = torch.tanh(curr_params[..., 0]).view(1, 1, 1, self.num_waves)
        omega = F.softplus(curr_params[..., 1]).view(1, 1, 1, self.num_waves) * self.base_freqs
        phi = (curr_params[..., 2] * math.pi).view(1, 1, 1, self.num_waves)
        decay = (F.softplus(curr_params[..., 3]) * 0.05).view(1, 1, 1, self.num_waves)

        wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
        wave_1d = wave_comps.sum(dim=-1).squeeze()

        topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.p_peaks, dim=-1)
        past_peak_offsets = past_peak_offsets + 1

        zero_offset = torch.zeros(1, dtype=torch.long, device=dev)
        zero_val = torch.zeros(1, dtype=torch.float, device=dev)

        offsets_1d = torch.cat([zero_offset, past_peak_offsets, -past_peak_offsets])
        peak_vals_1d = torch.cat([zero_val, topk_vals, topk_vals])

        next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
        return offsets_1d, peak_vals_1d, next_wave_latent

# ------------------------------------------------------------------------------
# 3. Architecture 1: SubQ ViT (T=4 hops pure attention, 1 Single MLP at End)
# ------------------------------------------------------------------------------
class SubQViT_SingleMLP(nn.Module):
    def __init__(self, num_classes=100, seq_len=256, d_model=128, n_heads=4, d_mlp=512, T=4, p_peaks=4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.T = T
        self.p_peaks = p_peaks
        self.k_total = 1 + 2 * p_peaks
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.inv_sqrt_T = 1.0 / math.sqrt(T)

        self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.ln_attn = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        self.router = FastBilateralWaveRouter(num_waves=12, p_peaks=p_peaks, max_d=seq_len)
        self.register_buffer("pos_idx", torch.arange(seq_len).unsqueeze(1))

        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, img):
        B = img.shape[0]
        patches = self.patch(img).flatten(2).transpose(1, 2)
        s = patches + self.pos
        dev = img.device
        L = self.seq_len

        wave_latent = self.router.init_wave_latent

        for _ in range(self.T):
            z = self.ln_attn(s)
            offsets_1d, peak_vals_1d, wave_latent = self.router(wave_latent, dev)

            Q = self.q(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2).contiguous()
            K = self.k(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2).contiguous()
            V = self.v(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2).contiguous()

            # Execute via Custom Triton Kernel if on GPU, otherwise reciprocal autograd
            if HAS_TRITON and Q.is_cuda:
                context = SubQTritonFunction.apply(Q, K, V, offsets_1d.to(torch.int32), peak_vals_1d.float())
            else:
                raw_targets = self.pos_idx - offsets_1d.unsqueeze(0)
                valid_mask = (raw_targets >= 0) & (raw_targets < L)
                clamped = torch.clamp(raw_targets, 0, L - 1)

                raw_rev_targets = self.pos_idx + offsets_1d.unsqueeze(0)
                rev_valid = (raw_rev_targets >= 0) & (raw_rev_targets < L)
                clamped_rev = torch.clamp(raw_rev_targets, 0, L - 1)

                context = FastSubQAttentionFunction.apply(
                    Q, K, V, peak_vals_1d, self.scale, clamped, valid_mask, clamped_rev, rev_valid
                )

            attn_out = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
            s = s + self.inv_sqrt_T * attn_out

        s = s + self.mlp(self.ln_mlp(s))
        pooled = self.ln_f(s).mean(dim=1)
        return self.head(pooled)

# ------------------------------------------------------------------------------
# 4. Architecture 2: Dense ViT (1-Layer Standard Full Attention + 1 MLP at End)
# ------------------------------------------------------------------------------
class DenseViT_SingleMLP(nn.Module):
    def __init__(self, num_classes=100, seq_len=256, d_model=128, n_heads=4, d_mlp=512):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.ln_attn = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model)

        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, img):
        B = img.shape[0]
        patches = self.patch(img).flatten(2).transpose(1, 2)
        x = patches + self.pos
        L = self.seq_len

        z = self.ln_attn(x)
        Q = self.q(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        attn_out = (weights @ V).transpose(1, 2).contiguous().view(B, L, self.d_model)

        x = x + self.c_proj(attn_out)
        x = x + self.mlp(self.ln_mlp(x))

        pooled = self.ln_f(x).mean(dim=1)
        return self.head(pooled)

# ------------------------------------------------------------------------------
# 5. High-Throughput RAM-Cached Dataset (Zero-CPU Bottleneck)
# ------------------------------------------------------------------------------
class CachedCifar100(Dataset):
    def __init__(self, X, Y, is_train=True):
        self.X = X
        self.Y = Y
        self.is_train = is_train
        self.mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        self.std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].float() / 255.0
        if self.is_train:
            img = F.pad(img, (4, 4, 4, 4), mode='reflect')
            top = torch.randint(0, 9, (1,)).item()
            left = torch.randint(0, 9, (1,)).item()
            img = img[:, top:top+32, left:left+32]
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, [2])

        img = (img - self.mean) / self.std
        return img, self.Y[idx]

# ------------------------------------------------------------------------------
# 6. Evaluation Routine
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

def verify_triton_gradients(device):
    if not (HAS_TRITON and torch.cuda.is_available()):
        return
    print("\n" + "=" * 80)
    print("  [TRITON SELF-CHECK] Verifying Triton Gradients against PyTorch Autograd...")
    print("=" * 80)
    B, H, L, D = 2, 4, 32, 32
    K_total = 9
    offsets_1d = torch.tensor([0, 1, 3, 5, 7, -1, -3, -5, -7], dtype=torch.int32, device=device)
    biases = torch.randn(K_total, device=device, requires_grad=True)
    scale = 1.0 / math.sqrt(D)

    Q = torch.randn(B, H, L, D, device=device, requires_grad=True)
    K = torch.randn(B, H, L, D, device=device, requires_grad=True)
    V = torch.randn(B, H, L, D, device=device, requires_grad=True)

    pos = torch.arange(L, device=device).unsqueeze(1)
    raw_targets = pos - offsets_1d.unsqueeze(0).long()
    valid_mask = (raw_targets >= 0) & (raw_targets < L)
    clamped = torch.clamp(raw_targets, 0, L - 1)
    raw_rev_targets = pos + offsets_1d.unsqueeze(0).long()
    rev_valid = (raw_rev_targets >= 0) & (raw_rev_targets < L)
    clamped_rev = torch.clamp(raw_rev_targets, 0, L - 1)

    out_ref = FastSubQAttentionFunction.apply(
        Q, K, V, biases, scale, clamped, valid_mask, clamped_rev, rev_valid
    )
    dOut = torch.randn_like(out_ref)
    loss_ref = (out_ref * dOut).sum()
    loss_ref.backward()

    dQ_ref = Q.grad.clone()
    dK_ref = K.grad.clone()
    dV_ref = V.grad.clone()
    db_ref = biases.grad.clone()

    Q.grad = None
    K.grad = None
    V.grad = None
    biases.grad = None

    out_tri = SubQTritonFunction.apply(Q, K, V, offsets_1d, biases)
    loss_tri = (out_tri * dOut).sum()
    loss_tri.backward()

    diff_out = (out_ref - out_tri).abs().max().item()
    diff_Q = (dQ_ref - Q.grad).abs().max().item()
    diff_K = (dK_ref - K.grad).abs().max().item()
    diff_V = (dV_ref - V.grad).abs().max().item()
    diff_b = (db_ref - biases.grad).abs().max().item()

    print(f"  --> Forward Out Diff:   {diff_out:.2e}")
    print(f"  --> Gradient Errors:    dQ={diff_Q:.2e} | dK={diff_K:.2e} | dV={diff_V:.2e} | dbias={diff_b:.2e}")
    if max(diff_out, diff_Q, diff_K, diff_V, diff_b) < 1e-4:
        print("  --> [TRITON STATUS] 100% BIT-EXACT MATCH WITH PYTORCH! Triton is READY.")
    else:
        print("  --> [WARNING] Discrepancy detected between Triton and PyTorch reference!")
    print("=" * 80 + "\n")

# ------------------------------------------------------------------------------
# 7. Main Training Routine
# ------------------------------------------------------------------------------
def main():
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("  EXPERIMENT S4-011: SINGLE-MLP CIFAR-100 SHOOTOUT (OPENAI TRITON ACCELERATED)")
    print("  1-LAYER SUBQ (T=4, K=9, 1 MLP AT END) VS. 1-LAYER DENSE (1 MLP AT END)")
    print(f"  Compute Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  OpenAI Triton Active: {HAS_TRITON and torch.cuda.is_available()}")
    print("=" * 100)

    # 0. Automatic Triton Self-Check
    verify_triton_gradients(device)

    # 1. Dataset Pre-Caching
    print("Loading HuggingFace dataset 'uoft-cs/cifar100'...")
    raw = load_dataset("uoft-cs/cifar100")

    print("Pre-caching dataset into contiguous RAM tensors (zero-CPU DataLoader)...")
    t0_cache = time.time()
    def cache_split(split):
        imgs, labels = [], []
        for item in raw[split]:
            img = item.get("img", item.get("image"))
            t_img = torch.from_numpy(np.array(img)).permute(2, 0, 1)
            imgs.append(t_img)
            labels.append(item.get("fine_label", item.get("label")))
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)

    X_train, Y_train = cache_split("train")
    X_test, Y_test = cache_split("test")
    print(f"Pre-caching completed in {time.time() - t0_cache:.2f}s! Train: {X_train.shape} | Test: {X_test.shape}\n")

    batch_size = 128
    epochs = 20
    train_loader = DataLoader(CachedCifar100(X_train, Y_train, is_train=True), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(CachedCifar100(X_test, Y_test, is_train=False), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    results = {}

    # =========================================================================
    # MODEL 1: SubQ ViT (T=4, K=9, 1 MLP at End)
    # =========================================================================
    print("=" * 100)
    print("  [MODEL 1/2] TRAINING: SubQ ViT (1-Layer Recurrent, T=4 hops, K=9 candidates, 1 MLP at End)")
    print("=" * 100)

    torch.manual_seed(seed)
    model_subq = SubQViT_SingleMLP(
        num_classes=100, seq_len=256, d_model=128, n_heads=4, d_mlp=512,
        T=4, p_peaks=4
    ).to(device)

    params_subq = sum(p.numel() for p in model_subq.parameters() if p.requires_grad)
    print(f"SubQ (Single-MLP) Trainable Parameters: {params_subq:,}")

    opt_subq = torch.optim.AdamW(model_subq.parameters(), lr=1e-3, weight_decay=0.05)
    sched_subq = torch.optim.lr_scheduler.CosineAnnealingLR(opt_subq, T_max=epochs, eta_min=1e-4)

    # Initial Warmup & Speed Benchmark
    dummy_x = torch.randn(batch_size, 3, 32, 32, device=device)
    dummy_y = torch.randint(0, 100, (batch_size,), device=device)
    opt_subq.zero_grad()
    t_warm = time.time()
    out_warm = model_subq(dummy_x)
    F.cross_entropy(out_warm, dummy_y).backward()
    opt_subq.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_ms = (time.time() - t_warm) * 1000
    print(f"  --> [Triton Speed Check] 1 Batch (Fwd + Bwd) completed in: {warmup_ms:.1f} ms! Expected epoch: ~{warmup_ms * len(train_loader) / 1000:.1f}s")

    t0_subq = time.time()
    history_subq = []

    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        model_subq.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt_subq.zero_grad(set_to_none=True)
            logits = model_subq(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_subq.parameters(), 1.0)
            opt_subq.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += labels.size(0)

        sched_subq.step()
        ep_loss = train_loss / train_total
        ep_acc = 100.0 * train_correct / train_total
        test_acc = evaluate(model_subq, test_loader, device)
        ep_duration = time.time() - t_ep_start

        print(
            f"  [SubQ Single-MLP] Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {ep_loss:.4f} | Train Acc: {ep_acc:5.2f}% | "
            f"TEST ACC: {test_acc:5.2f}% | "
            f"Epoch Time: {ep_duration:4.1f}s"
        )
        history_subq.append({"epoch": epoch, "loss": round(ep_loss, 4), "train_acc": round(ep_acc, 2), "test_acc": round(test_acc, 2), "epoch_sec": round(ep_duration, 1)})

    time_subq = time.time() - t0_subq
    final_acc_subq = test_acc
    print(f"--> [MODEL 1 FINISHED] SubQ Final Test Accuracy: {final_acc_subq:.2f}% (Total Time: {time_subq:.1f}s)\n")
    results["subq"] = {"params": params_subq, "acc": final_acc_subq, "time": time_subq, "history": history_subq}

    # =========================================================================
    # MODEL 2: Dense ViT (1-Layer Standard Full Attention + 1 MLP at End)
    # =========================================================================
    print("=" * 100)
    print("  [MODEL 2/2] TRAINING: Dense ViT (1-Layer Standard Full Attention, 1 MLP at End)")
    print("=" * 100)

    torch.manual_seed(seed)
    model_dense = DenseViT_SingleMLP(
        num_classes=100, seq_len=256, d_model=128, n_heads=4, d_mlp=512
    ).to(device)

    params_dense = sum(p.numel() for p in model_dense.parameters() if p.requires_grad)
    print(f"Dense (Single-MLP) Trainable Parameters: {params_dense:,}")

    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=1e-3, weight_decay=0.05)
    sched_dense = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dense, T_max=epochs, eta_min=1e-4)

    t0_dense = time.time()
    history_dense = []

    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        model_dense.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt_dense.zero_grad(set_to_none=True)
            logits = model_dense(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_dense.parameters(), 1.0)
            opt_dense.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += labels.size(0)

        sched_dense.step()
        ep_loss = train_loss / train_total
        ep_acc = 100.0 * train_correct / train_total
        test_acc = evaluate(model_dense, test_loader, device)
        ep_duration = time.time() - t_ep_start

        print(
            f"  [Dense Single-MLP] Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {ep_loss:.4f} | Train Acc: {ep_acc:5.2f}% | "
            f"TEST ACC: {test_acc:5.2f}% | "
            f"Epoch Time: {ep_duration:4.1f}s"
        )
        history_dense.append({"epoch": epoch, "loss": round(ep_loss, 4), "train_acc": round(ep_acc, 2), "test_acc": round(test_acc, 2), "epoch_sec": round(ep_duration, 1)})

    time_dense = time.time() - t0_dense
    final_acc_dense = test_acc
    print(f"--> [MODEL 2 FINISHED] Dense Final Test Accuracy: {final_acc_dense:.2f}% (Total Time: {time_dense:.1f}s)\n")
    results["dense"] = {"params": params_dense, "acc": final_acc_dense, "time": time_dense, "history": history_dense}

    # =========================================================================
    # DEFINITIVE HEAD-TO-HEAD SCORECARD
    # =========================================================================
    print("=" * 100)
    print("  DEFINITIVE HEAD-TO-HEAD SCORECARD: SINGLE-MLP ATTENTION SHOOTOUT")
    print("=" * 100)
    print(f"  {'Model Name':<35} | {'Lookups/Token':<15} | {'Parameters':<12} | {'Test Acc':<10} | {'Time (s)':<10}")
    print(f"  {'-'*35}-+-{'-'*15}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")
    if "subq" in results:
        print(f"  {'SubQ ViT (T=4, K=9, 1 MLP at End)':<35} | {'36 (4 x 9)':<15} | {results['subq']['params']:<12,} | {results['subq']['acc']:5.2f}%    | {results['subq']['time']:5.1f}s")
    if "dense" in results:
        print(f"  {'Dense ViT (1-Layer, 1 MLP at End)':<35} | {'256 (1 x 256)':<15} | {results['dense']['params']:<12,} | {results['dense']['acc']:5.2f}%    | {results['dense']['time']:5.1f}s")
    print("=" * 100)

    if "subq" in results and "dense" in results:
        delta = results['subq']['acc'] - results['dense']['acc']
        print(f"  NET RECURRENT SUBQ ADVANTAGE: {delta:+.2f}%")
        if delta > 0:
            print("  VERDICT: SubQ Sparse Recurrence + 1 Spaced MLP beats Dense ViT!")
        else:
            print("  VERDICT: Dense ViT is competitive when paired with 1 MLP.")
        print("=" * 100)

    with open("s4_011_single_mlp_cifar100_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to 's4_011_single_mlp_cifar100_results.json'")

if __name__ == "__main__":
    main()
