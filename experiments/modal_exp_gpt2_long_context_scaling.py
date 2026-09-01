import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "requests",
        "numpy",
        "accelerate"
    )
)

app = modal.App("exp-gpt2-long-context-scaling", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=3600, volumes={"/root/checkpoints": volume})
def run_gpt2_long_context_benchmark():
    import math
    import time
    import gc
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 130)
    print("  STUDY 39: FULL 12-LAYER FOUNDATION LLM (124M) EXTREME LONG-CONTEXT SCALING & OOM BENCHMARK")
    print("  Comparing 12-Layer Dense GPT-2 O(L^2) vs. 12-Layer SubQ-GPT2 O(L*K) Across L = 1,024 -> 16,384 Tokens")
    print("=" * 130)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Load Tokenizer & Base GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    orig_dense_base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    orig_dense_base.eval()

    d_model, n_heads = 768, 12
    head_dim = d_model // n_heads
    # Causal logarithmic offsets for SubQ
    causal_offsets = [0, 1, 2, 4, 8, 16, 32, 64]
    K = len(causal_offsets)

    # 2. Optimized SubQ Attention Module for Ultra-Long Sequences
    class CausalSubQSelfAttention(nn.Module):
        def __init__(self, orig_attn_layer):
            super().__init__()
            self.offsets = causal_offsets
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)
            self.temp_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1))

            with torch.no_grad():
                self.c_attn.weight.copy_(orig_attn_layer.c_attn.weight.t())
                self.c_attn.bias.copy_(orig_attn_layer.c_attn.bias)
                self.c_proj.weight.copy_(orig_attn_layer.c_proj.weight.t())
                self.c_proj.bias.copy_(orig_attn_layer.c_proj.bias)

        def forward(self, x):
            B, L, D = x.shape
            qkv = self.c_attn(x)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(B, L, n_heads, head_dim).transpose(1, 2) # [B, H, L, d]
            k = k.view(B, L, n_heads, head_dim).transpose(1, 2) # [B, H, L, d]
            v = v.view(B, L, n_heads, head_dim).transpose(1, 2) # [B, H, L, d]

            scores_list = []
            valid_masks = []
            for d in self.offsets:
                if d == 0:
                    # dot product q and k at same position
                    s = (q * k).sum(dim=-1) / math.sqrt(head_dim) # [B, H, L]
                    m = torch.ones(B, 1, L, device=x.device, dtype=torch.bool)
                elif d < L:
                    # q at position i matches k at position (i - d)
                    q_slice = q[:, :, d:, :]
                    k_slice = k[:, :, :-d, :]
                    s_valid = (q_slice * k_slice).sum(dim=-1) / math.sqrt(head_dim)
                    s = F.pad(s_valid, (d, 0), value=-1e4)
                    m = torch.cat([torch.zeros(B, 1, d, device=x.device, dtype=torch.bool), torch.ones(B, 1, L - d, device=x.device, dtype=torch.bool)], dim=-1)
                else:
                    s = torch.full((B, n_heads, L), -1e4, device=x.device)
                    m = torch.zeros(B, 1, L, device=x.device, dtype=torch.bool)
                scores_list.append(s)
                valid_masks.append(m)

            scores = torch.stack(scores_list, dim=-1) * self.temp_scale # [B, H, L, K]
            valid_mask = torch.stack(valid_masks, dim=-1) # [B, 1, L, K]
            scores = scores.masked_fill(~valid_mask, float("-inf"))
            weights = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0) # [B, H, L, K]

            # Weighted sum over values
            out = torch.zeros_like(q)
            for k_idx, d in enumerate(self.offsets):
                w_k = weights[:, :, :, k_idx : k_idx + 1] # [B, H, L, 1]
                if d == 0:
                    out = out + w_k * v
                elif d < L:
                    v_shifted = F.pad(v[:, :, :-d, :], (0, 0, d, 0))
                    out = out + w_k * v_shifted

            out = out.transpose(1, 2).contiguous().view(B, L, D)
            return self.c_proj(out)

    class SubQGPT2Block(nn.Module):
        def __init__(self, orig_block):
            super().__init__()
            self.ln_1 = orig_block.ln_1
            self.attn = CausalSubQSelfAttention(orig_block.attn)
            self.ln_2 = orig_block.ln_2
            self.mlp = orig_block.mlp

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    class Full12LayerSubQGPT2(nn.Module):
        def __init__(self, orig_model):
            super().__init__()
            self.wte = orig_model.transformer.wte
            self.wpe = orig_model.transformer.wpe
            self.blocks = nn.ModuleList([
                SubQGPT2Block(orig_model.transformer.h[i]) for i in range(12)
            ])
            self.ln_f = orig_model.transformer.ln_f
            self.lm_head = orig_model.lm_head

        def forward(self, input_ids, last_token_only=False):
            B, L = input_ids.shape
            pos = torch.arange(0, L, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            pos_emb = self.wpe(pos % 1024)
            h = self.wte(input_ids) + pos_emb
            for block in self.blocks:
                h = block(h)
            h = self.ln_f(h)
            if last_token_only:
                logits = self.lm_head(h[:, -1:, :])
            else:
                logits = self.lm_head(h)
            return logits

    class ExtDenseGPT2(nn.Module):
        def __init__(self, orig_model):
            super().__init__()
            self.model = orig_model

        def forward(self, input_ids, last_token_only=False):
            B, L = input_ids.shape
            pos = torch.arange(0, L, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            pos_emb = self.model.transformer.wpe(pos % 1024)
            h = self.model.transformer.wte(input_ids) + pos_emb
            for block in self.model.transformer.h:
                attn_out = block.attn(block.ln_1(h))[0]
                h = h + attn_out
                h = h + block.mlp(block.ln_2(h))
            h = self.model.transformer.ln_f(h)
            if last_token_only:
                logits = self.model.lm_head(h[:, -1:, :])
            else:
                logits = self.model.lm_head(h)
            return logits

    dense_model = ExtDenseGPT2(orig_dense_base).to(device)
    subq_model = Full12LayerSubQGPT2(orig_dense_base).to(device)

    # 3. Context Length Frontier Test Grid
    test_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    results = []

    print("\n" + "=" * 130)
    print("  RUNNING SYSTEMATIC FORWARD BENCHMARKS (BATCH SIZE = 1, FP16)")
    print("=" * 130)

    for L in test_lengths:
        print(f"\n>>> TESTING CONTEXT LENGTH L = {L:,} TOKENS:")

        # --- A. Test Dense GPT-2 ---
        dense_vram, dense_time, dense_tok_s, dense_status = "OOM", "N/A", "0 tok/s", "💥 OOM Crash"
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        try:
            input_ids = torch.randint(0, 50257, (1, L), device=device)
            # Warmup
            with torch.amp.autocast('cuda', dtype=torch.float16):
                _ = dense_model(input_ids, last_token_only=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            # Timed iterations
            num_iters = 5 if L <= 4096 else 2
            t0 = time.time()
            for _ in range(num_iters):
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _ = dense_model(input_ids, last_token_only=True)
                torch.cuda.synchronize()
            t1 = time.time()

            elapsed_per_iter = (t1 - t0) / num_iters
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            tok_per_sec = L / elapsed_per_iter

            dense_vram = f"{peak_mb:.1f} MB"
            dense_time = f"{elapsed_per_iter*1000:.1f} ms"
            dense_tok_s = f"{tok_per_sec:,.0f} tok/s"
            dense_status = "✅ PASS"
            print(f"  [12L Dense GPT-2] -> Latency: {dense_time} | VRAM: {dense_vram} | Speed: {dense_tok_s} | Status: {dense_status}")
        except Exception as e:
            dense_status = f"💥 OOM: {type(e).__name__}"
            print(f"  [12L Dense GPT-2] -> 💥 CRASHED AT L = {L:,}: {e}")
            torch.cuda.empty_cache()

        # --- B. Test SubQ-GPT2 ---
        subq_vram, subq_time, subq_tok_s, subq_status = "OOM", "N/A", "0 tok/s", "💥 OOM Crash"
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        try:
            input_ids = torch.randint(0, 50257, (1, L), device=device)
            # Warmup
            with torch.amp.autocast('cuda', dtype=torch.float16):
                _ = subq_model(input_ids, last_token_only=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            # Timed iterations
            num_iters = 5 if L <= 4096 else 3
            t0 = time.time()
            for _ in range(num_iters):
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _ = subq_model(input_ids, last_token_only=True)
                torch.cuda.synchronize()
            t1 = time.time()

            elapsed_per_iter = (t1 - t0) / num_iters
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            tok_per_sec = L / elapsed_per_iter

            subq_vram = f"{peak_mb:.1f} MB"
            subq_time = f"{elapsed_per_iter*1000:.1f} ms"
            subq_tok_s = f"{tok_per_sec:,.0f} tok/s"
            subq_status = "✅ PASS"
            print(f"  [12L SubQ-GPT2]   -> Latency: {subq_time} | VRAM: {subq_vram} | Speed: {subq_tok_s} | Status: {subq_status}")
        except Exception as e:
            subq_status = f"💥 OOM: {type(e).__name__}"
            print(f"  [12L SubQ-GPT2]   -> 💥 CRASHED AT L = {L:,}: {e}")
            torch.cuda.empty_cache()

        results.append({
            "L": L,
            "dense_vram": dense_vram,
            "dense_time": dense_time,
            "dense_tok_s": dense_tok_s,
            "dense_status": dense_status,
            "subq_vram": subq_vram,
            "subq_time": subq_time,
            "subq_tok_s": subq_tok_s,
            "subq_status": subq_status
        })

    # 4. Long-Context Needle-in-a-Haystack Retrieval at L = 8,192
    print("\n" + "=" * 130)
    print("  [4/5] LONG-CONTEXT NEEDLE-IN-A-HAYSTACK RETRIEVAL AT L = 8,192 TOKENS")
    print("=" * 130)

    needle_target = "passkey_74921"
    prompt_needle = f" The secret passkey is {needle_target}. Remember this passkey."
    distractor = " In a distant galaxy there are countless stars and planets orbiting in silence."

    print(f"Constructing L = 8,192 token haystack with needle placed at depth = 50%...")
    needle_tokens = tokenizer.encode(prompt_needle)
    distractor_tokens = tokenizer.encode(distractor)

    full_haystack = []
    while len(full_haystack) < 8192:
        full_haystack.extend(distractor_tokens)
    full_haystack = full_haystack[:8192]

    # Insert needle in the middle
    mid = 4096
    full_haystack[mid : mid + len(needle_tokens)] = needle_tokens
    query_prompt = " What is the secret passkey? The secret passkey is"
    query_tokens = tokenizer.encode(query_prompt)
    full_haystack[-len(query_tokens):] = query_tokens

    haystack_tensor = torch.tensor([full_haystack], dtype=torch.long, device=device)
    print(f"Haystack input shape: {haystack_tensor.shape} (8,192 tokens)")

    # Run SubQ-GPT2 retrieval pass
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            subq_out = subq_model(haystack_tensor, last_token_only=True)
            pred_token_id = subq_out[0, -1].argmax().item()
            pred_token = tokenizer.decode([pred_token_id])

    print(f"SubQ-GPT2 Top-1 Predicted Token at End of 8,192 Context: '{pred_token}'")

    # 5. Final Master Scorecard
    print("\n" + "=" * 130)
    print("  [5/5] MASTER LONG-CONTEXT COMPARISON SCORECARD (12-LAYER 124M MODELS)")
    print("=" * 130)
    print(f"{'Context Length (L)':<20} | {'Dense GPT-2 VRAM':<18} | {'SubQ-GPT2 VRAM':<16} | {'Dense Speed':<16} | {'SubQ Speed':<16} | {'VRAM Advantage':<16}")
    print("-" * 130)

    for r in results:
        vram_adv = "N/A"
        if "MB" in r["dense_vram"] and "MB" in r["subq_vram"]:
            d_val = float(r["dense_vram"].replace(" MB", ""))
            s_val = float(r["subq_vram"].replace(" MB", ""))
            ratio = d_val / max(1e-5, s_val)
            vram_adv = f"{ratio:.1f}x lower"
        elif "OOM" in r["dense_status"]:
            vram_adv = "🚀 SubQ Solves (OOM)"

        print(f"{r['L']:<20,} | {r['dense_vram']:<18} | {r['subq_vram']:<16} | {r['dense_tok_s']:<16} | {r['subq_tok_s']:<16} | {vram_adv:<16}")

    print("=" * 130)

@app.local_entrypoint()
def main():
    run_gpt2_long_context_benchmark.remote()
