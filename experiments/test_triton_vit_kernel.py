import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import triton
    import triton.language as tl

    @triton.jit
    def harmonic_subq_vision_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Offsets_ptr, Biases_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_off_h, stride_bias_h,
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

        q_ptrs = Q_ptr + (batch_id * stride_qb + head_id * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        mask_m = offs_m < L
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        offsets_base = Offsets_ptr + head_id * stride_off_h
        biases_base = Biases_ptr + head_id * stride_bias_h

        k_offs = tl.load(offsets_base + tl.arange(0, NUM_OFFSETS))
        biases = tl.load(biases_base + tl.arange(0, NUM_OFFSETS))

        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for k_idx in range(NUM_OFFSETS):
            offset_val = tl.load(offsets_base + k_idx)
            bias_val = tl.load(biases_base + k_idx)

            target_pos = (offs_m - offset_val) % L

            k_ptrs = K_ptr + (batch_id * stride_kb + head_id * stride_kh + target_pos[:, None] * stride_kn + offs_d[None, :] * stride_kd)
            v_ptrs = V_ptr + (batch_id * stride_vb + head_id * stride_vh + target_pos[:, None] * stride_vn + offs_d[None, :] * stride_vd)

            k = tl.load(k_ptrs, mask=mask_m[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_m[:, None], other=0.0)

            score = tl.sum(q * k, axis=1) * scale + bias_val

            m_new = tl.maximum(m_i, score)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(score - m_new)

            acc = acc * alpha[:, None] + p[:, None] * v
            l_i = l_i * alpha + p
            m_i = m_new

        acc = acc / l_i[:, None]
        out_ptrs = Out_ptr + (batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
        tl.store(out_ptrs, acc, mask=mask_m[:, None])

except ImportError:
    pass
