import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "triton>=2.2.0",
        "transformers>=4.40.0",
        "numpy"
    )
)

app = modal.App("exp-flashattn-vs-subq-triton", image=image)

@app.function(gpu="A10G", timeout=1800)
def benchmark_flashattn_vs_subq_triton():
    import math
    import time
    import gc
    import torch
    import torch.nn.functional as F
    import triton
    import triton.language as tl

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 135)
    print("  STUDY 41: DIRECT 3-WAY BENCHMARK — DENSE FLASHATTENTION-2 VS. PYTORCH EAGER SUBQ VS. OPENAI TRITON SUBQ KERNEL")
    print("  Comparing O(L^2) Dense FlashAttention-2 against O(L*K) SubQ Across L = 1,024 -> 65,536 Tokens on NVIDIA A10G (24GB)")
    print("=" * 135)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Custom Triton JIT Kernel for Causal SubQ Attention
    @triton.jit
    def _subq_causal_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        offsets_ptr,
        scale,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_ob, stride_oh, stride_ol, stride_od,
        B, H, L,
        NUM_OFFSETS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch_idx = pid_bh // H
        head_idx = pid_bh % H

        offs_d = tl.arange(0, HEAD_DIM)
        q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
        k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
        v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
        out_base = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh

        for i in range(BLOCK_SIZE):
            token_idx = pid_m * BLOCK_SIZE + i
            if token_idx < L:
                q_ptrs = q_base + token_idx * stride_ql + offs_d * stride_qd
                q = tl.load(q_ptrs)

                m_prev = -float("inf")
                d_prev = 0.0
                acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

                for k_idx in range(NUM_OFFSETS):
                    offset_val = tl.load(offsets_ptr + k_idx)
                    source_idx = token_idx - offset_val

                    if source_idx >= 0:
                        k_ptrs = k_base + source_idx * stride_kl + offs_d * stride_kd
                        k = tl.load(k_ptrs)
                        score = tl.sum(q * k) * scale

                        m_curr = tl.maximum(m_prev, score)
                        exp_score = tl.exp(score - m_curr)
                        alpha = tl.exp(m_prev - m_curr)

                        v_ptrs = v_base + source_idx * stride_vl + offs_d * stride_vd
                        v = tl.load(v_ptrs)

                        acc = acc * alpha + exp_score * v
                        d_prev = d_prev * alpha + exp_score
                        m_prev = m_curr

                if d_prev > 0.0:
                    out = acc / d_prev
                else:
                    out = tl.zeros([HEAD_DIM], dtype=tl.float32)

                out_ptrs = out_base + token_idx * stride_ol + offs_d * stride_od
                tl.store(out_ptrs, out.to(tl.float16))

    def subq_triton_fwd(q, k, v, offsets):
        B, H, L, D = q.shape
        out = torch.empty_like(q)
        offsets_tensor = torch.tensor(offsets, dtype=torch.int32, device=q.device)
        NUM_OFFSETS = len(offsets)
        scale = 1.0 / math.sqrt(D)
        BLOCK_SIZE = 32
        grid = (triton.cdiv(L, BLOCK_SIZE), B * H)
        _subq_causal_fwd_kernel[grid](
            q, k, v, out,
            offsets_tensor,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, L,
            NUM_OFFSETS=NUM_OFFSETS,
            HEAD_DIM=D,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out

    # 2. PyTorch Eager Reference
    def subq_pytorch_eager(q, k, v, offsets):
        B, H, L, D = q.shape
        scale = 1.0 / math.sqrt(D)
        scores_list, valid_masks = [], []
        for d in offsets:
            if d == 0:
                s = (q * k).sum(dim=-1) * scale
                m = torch.ones(B, H, L, device=q.device, dtype=torch.bool)
            elif d < L:
                q_slice = q[:, :, d:, :]
                k_slice = k[:, :, :-d, :]
                s_valid = (q_slice * k_slice).sum(dim=-1) * scale
                s = F.pad(s_valid, (d, 0), value=-1e4)
                m = torch.cat([torch.zeros(B, H, d, device=q.device, dtype=torch.bool), torch.ones(B, H, L - d, device=q.device, dtype=torch.bool)], dim=-1)
            else:
                s = torch.full((B, H, L), -1e4, device=q.device)
                m = torch.zeros(B, H, L, device=q.device, dtype=torch.bool)
            scores_list.append(s)
            valid_masks.append(m)

        scores = torch.stack(scores_list, dim=-1)
        valid_mask = torch.stack(valid_masks, dim=-1)
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        weights = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)

        out = torch.zeros_like(q)
        for k_idx, d in enumerate(offsets):
            w_k = weights[:, :, :, k_idx : k_idx + 1]
            if d == 0:
                out = out + w_k * v
            elif d < L:
                v_shifted = F.pad(v[:, :, :-d, :], (0, 0, d, 0))
                out = out + w_k * v_shifted
        return out

    # 3. Dense FlashAttention-2 (PyTorch native scaled_dot_product_attention with is_causal=True)
    def dense_flashattention_fwd(q, k, v):
        # q, k, v shape: [B, H, L, D]
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    offsets_test = [0, 1, 2, 4, 8, 16, 32, 64]
    test_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    benchmark_results = []

    print("\n" + "=" * 135)
    print("  RUNNING SYSTEMATIC 3-WAY BENCHMARK (Batch=1, Heads=12, HeadDim=64, FP16)")
    print("=" * 135)

    for L in test_lengths:
        print(f"\n>>> TESTING SEQUENCE LENGTH L = {L:,} TOKENS:")
        q = torch.randn(1, 12, L, 64, dtype=torch.float16, device=device)
        k = torch.randn(1, 12, L, 64, dtype=torch.float16, device=device)
        v = torch.randn(1, 12, L, 64, dtype=torch.float16, device=device)

        # -------------------------------------------------------------
        # A. Dense FlashAttention-2
        # -------------------------------------------------------------
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        flash_time, flash_vram, flash_tok_s = "OOM", "OOM", "0 tok/s"
        try:
            _ = dense_flashattention_fwd(q, k, v)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            iters = 10 if L <= 8192 else (5 if L <= 32768 else 2)
            t0 = time.time()
            for _ in range(iters):
                _ = dense_flashattention_fwd(q, k, v)
            torch.cuda.synchronize()
            elapsed = (time.time() - t0) / iters
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            flash_time = f"{elapsed*1000:.2f} ms"
            flash_vram = f"{peak_mb:.1f} MB"
            flash_tok_s = f"{L / elapsed:,.0f} tok/s"
            print(f"  [1. Dense FlashAttention-2] -> Time: {flash_time:>10} | Peak VRAM: {flash_vram:>9} | Speed: {flash_tok_s}")
        except Exception as e:
            flash_time = "💥 OOM Crash"
            print(f"  [1. Dense FlashAttention-2] -> 💥 CRASHED AT L = {L:,}: {e}")

        # -------------------------------------------------------------
        # B. PyTorch Eager SubQ (Previous Prototype)
        # -------------------------------------------------------------
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        eager_time, eager_vram, eager_tok_s = "OOM", "OOM", "0 tok/s"
        try:
            _ = subq_pytorch_eager(q, k, v, offsets_test)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            iters = 5 if L <= 8192 else 2
            t0 = time.time()
            for _ in range(iters):
                _ = subq_pytorch_eager(q, k, v, offsets_test)
            torch.cuda.synchronize()
            elapsed = (time.time() - t0) / iters
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            eager_time = f"{elapsed*1000:.2f} ms"
            eager_vram = f"{peak_mb:.1f} MB"
            eager_tok_s = f"{L / elapsed:,.0f} tok/s"
            print(f"  [2. PyTorch Eager SubQ]     -> Time: {eager_time:>10} | Peak VRAM: {eager_vram:>9} | Speed: {eager_tok_s}")
        except Exception as e:
            eager_time = "💥 OOM Crash"
            print(f"  [2. PyTorch Eager SubQ]     -> 💥 CRASHED AT L = {L:,}: {e}")

        # -------------------------------------------------------------
        # C. Custom OpenAI Triton SubQ Kernel
        # -------------------------------------------------------------
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        triton_time, triton_vram, triton_tok_s = "OOM", "OOM", "0 tok/s"
        try:
            _ = subq_triton_fwd(q, k, v, offsets_test)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            iters = 10 if L <= 16384 else 5
            t0 = time.time()
            for _ in range(iters):
                _ = subq_triton_fwd(q, k, v, offsets_test)
            torch.cuda.synchronize()
            elapsed = (time.time() - t0) / iters
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            triton_time = f"{elapsed*1000:.2f} ms"
            triton_vram = f"{peak_mb:.1f} MB"
            triton_tok_s = f"{L / elapsed:,.0f} tok/s"
            print(f"  [3. OpenAI Triton SubQ]     -> Time: {triton_time:>10} | Peak VRAM: {triton_vram:>9} | Speed: {triton_tok_s}")
        except Exception as e:
            triton_time = "💥 Error"
            print(f"  [3. OpenAI Triton SubQ]     -> 💥 CRASHED: {e}")

        benchmark_results.append({
            "L": L,
            "flash_time": flash_time,
            "flash_vram": flash_vram,
            "flash_speed": flash_tok_s,
            "eager_time": eager_time,
            "eager_vram": eager_vram,
            "eager_speed": eager_tok_s,
            "triton_time": triton_time,
            "triton_vram": triton_vram,
            "triton_speed": triton_tok_s
        })

    # Master Scorecard
    print("\n" + "=" * 135)
    print("  FINAL 3-WAY COMPARISON SCORECARD: FLASHATTENTION-2 VS. PYTORCH EAGER SUBQ VS. TRITON SUBQ")
    print("=" * 135)
    print(f"{'Length (L)':<12} | {'Dense FlashAttn-2 Time':<24} | {'Eager SubQ Time':<18} | {'Triton SubQ Time':<18} | {'Triton Speed':<16} | {'Complexity Advantage':<20}")
    print("-" * 135)

    for r in benchmark_results:
        adv = "O(L*K) Linear"
        print(f"{r['L']:<12,} | {r['flash_time']:<24} | {r['eager_time']:<18} | {r['triton_time']:<18} | {r['triton_speed']:<16} | {adv:<20}")

    print("=" * 135)

@app.local_entrypoint()
def main():
    benchmark_flashattn_vs_subq_triton.remote()
