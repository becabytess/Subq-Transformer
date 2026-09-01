import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.38.0",
        "numpy",
        "requests",
        "accelerate"
    )
)

app = modal.App("gpt2-subq-bidirectional-diffusion", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=3000, volumes={"/root/checkpoints": volume})
def run_bidirectional_diffusion_experiment():
    import math
    import time
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2Tokenizer
    import requests
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 29: SUBQ BIDIRECTIONAL ATTRACTOR DIFFUSION WITH FREQUENCY DEBIASING & GRAPH RELAXATION")
    print("  Solving the Mode-Collapse Problem:")
    print("    1. Bidirectional Graph Hops: Virtual block tokens exchange messages both forwards and backwards.")
    print("    2. Unigram Frequency Debiasing: Subtract log P_unigram(w) to prevent the 'the' marginal mode collapse.")
    print("    3. Dynamic Nucleus & Repetition Sampling across diffusion steps.")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Tokenizer & Dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = 50257
    d_model = 768
    n_heads = 12
    head_dim = 64
    seq_len = 256
    block_size = 8

    print(f"\n[1/5] Fetching Multi-Domain Text Dataset...")
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    text_data = requests.get(url).text
    tokens = tokenizer.encode(text_data)
    data_tensor = torch.tensor(tokens, dtype=torch.long, device=device)

    split = int(len(data_tensor) * 0.9)
    train_tokens = data_tensor[:split]
    val_tokens = data_tensor[split:]

    # Compute empirical unigram token frequencies for debiased diffusion
    token_counts = torch.bincount(train_tokens, minlength=vocab_size).float()
    unigram_probs = (token_counts + 1.0) / (token_counts.sum() + vocab_size)
    log_unigram = torch.log(unigram_probs).to(device)
    print(f"Dataset: Train {len(train_tokens):,} | Val {len(val_tokens):,} tokens | Unigram Frequency Map Computed")

    # 2. SubQ Bidirectional Jump Attention
    def get_bidirectional_menu(K=32):
        # Asymmetric menu: causal backward jumps for prefix, bidirectional small jumps for block
        backward_jumps = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
        forward_jumps = [-1, -2, -3, -4, -6, -8]
        combined = backward_jumps + forward_jumps
        while len(combined) < K:
            combined.append(combined[-1] + 1)
        return torch.tensor(combined[:K], dtype=torch.long, device=device)

    class SubQBidirectionalAttention(nn.Module):
        def __init__(self, d_model=768, n_heads=12, K=32, T_max=4):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.K = K
            self.T_max = T_max

            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)
            self.register_buffer("offsets", get_bidirectional_menu(K))
            self.temp_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1))

            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=True)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=True)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=True)

        def forward(self, x, is_prefix_causal=True, prefix_len=128, T_max=4):
            B, L, D = x.shape
            s = x
            hops_taken = 0
            energy_trace = []

            for t in range(T_max):
                s_prev = s
                qkv = self.c_attn(s)
                q, k, v = qkv.chunk(3, dim=-1)

                q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

                k_shifted_list = []
                v_shifted_list = []
                valid_mask_list = []

                for delta_val in self.offsets:
                    d = delta_val.item()
                    
                    if d >= 0:
                        # Backward jump (looking past)
                        if d >= L:
                            k_s = torch.zeros_like(k)
                            v_s = torch.zeros_like(v)
                            m = torch.zeros(B, 1, L, device=x.device, dtype=torch.bool)
                        elif d == 0:
                            k_s = k
                            v_s = v
                            m = torch.ones(B, 1, L, device=x.device, dtype=torch.bool)
                        else:
                            k_s = F.pad(k[:, :, :-d, :], (0, 0, d, 0))
                            v_s = F.pad(v[:, :, :-d, :], (0, 0, d, 0))
                            m = torch.cat([
                                torch.zeros(B, 1, d, device=x.device, dtype=torch.bool),
                                torch.ones(B, 1, L - d, device=x.device, dtype=torch.bool)
                            ], dim=-1)
                    else:
                        # Forward jump (looking ahead into the block)
                        d_abs = abs(d)
                        if d_abs >= L:
                            k_s = torch.zeros_like(k)
                            v_s = torch.zeros_like(v)
                            m = torch.zeros(B, 1, L, device=x.device, dtype=torch.bool)
                        else:
                            k_s = F.pad(k[:, :, d_abs:, :], (0, 0, 0, d_abs))
                            v_s = F.pad(v[:, :, d_abs:, :], (0, 0, 0, d_abs))
                            # Mask out prefix tokens from looking forward to maintain causality for prefix
                            m = torch.cat([
                                torch.ones(B, 1, L - d_abs, device=x.device, dtype=torch.bool),
                                torch.zeros(B, 1, d_abs, device=x.device, dtype=torch.bool)
                            ], dim=-1)
                            if is_prefix_causal:
                                # Force prefix positions to False for forward jumps
                                m[:, :, :prefix_len] = False

                    k_shifted_list.append(k_s)
                    v_shifted_list.append(v_s)
                    valid_mask_list.append(m)

                K_cand = torch.stack(k_shifted_list, dim=3)
                V_cand = torch.stack(v_shifted_list, dim=3)
                valid_mask = torch.stack(valid_mask_list, dim=3)

                q_expanded = q.unsqueeze(3)
                scores = (q_expanded * K_cand).sum(dim=-1) / math.sqrt(self.head_dim)
                scores = scores * self.temp_scale
                scores = scores.masked_fill(~valid_mask, float("-inf"))
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

                out_h = (attn_weights.unsqueeze(-1) * V_cand).sum(dim=3)
                out = out_h.transpose(1, 2).contiguous().view(B, L, D)
                context = self.c_proj(out)

                gates_ih = self.w_ih(context)
                r_ih, z_ih, n_ih = gates_ih.chunk(3, dim=-1)

                gates_h = self.w_gate_h(s)
                r_h, z_h = gates_h.chunk(2, dim=-1)

                r = torch.sigmoid(r_ih + r_h)
                z = torch.sigmoid(z_ih + z_h)
                n = torch.tanh(n_ih + self.w_cand_h(r * s))

                s = (1.0 - z) * n + z * s
                hops_taken += 1
                energy_trace.append(torch.norm(s - s_prev, p=2, dim=-1).mean().item())

            return s, hops_taken, energy_trace

    class SubQBidirectionalDiffusionModel(nn.Module):
        def __init__(self, K=32, T_max=4, block_size=8):
            super().__init__()
            self.block_size = block_size
            self.wte = nn.Embedding(vocab_size, d_model)
            self.wpe = nn.Embedding(1024, d_model)
            self.drop = nn.Dropout(0.1)
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

            self.slot_embeddings = nn.Parameter(torch.randn(block_size, d_model) * 0.02)

            self.blocks = nn.ModuleList()
            for l in range(12):
                subq_attn = SubQBidirectionalAttention(d_model=768, n_heads=12, K=K, T_max=T_max)
                ln1 = nn.LayerNorm(768)
                ln2 = nn.LayerNorm(768)
                mlp = nn.Sequential(
                    nn.Linear(768, 4 * 768),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(4 * 768, 768),
                    nn.Dropout(0.1)
                )
                block_dict = nn.ModuleDict({
                    "ln_1": ln1,
                    "attn": subq_attn,
                    "ln_2": ln2,
                    "mlp": mlp
                })
                self.blocks.append(block_dict)

        def forward_with_mask_pattern(self, prefix_ids, target_ids, mask_indices, T_max=4):
            B, L_pre = prefix_ids.shape
            N = len(target_ids)
            total_len = L_pre + N

            pos = torch.arange(0, total_len, dtype=torch.long, device=prefix_ids.device).unsqueeze(0)
            prefix_emb = self.wte(prefix_ids)

            target_emb_list = []
            for j in range(N):
                if j in mask_indices:
                    slot_vec = self.slot_embeddings[j].unsqueeze(0).unsqueeze(0)
                    target_emb_list.append(slot_vec)
                else:
                    tok_id = target_ids[j].view(1, 1)
                    tok_vec = self.wte(tok_id)
                    target_emb_list.append(tok_vec)

            target_block_emb = torch.cat(target_emb_list, dim=1)
            full_emb = torch.cat([prefix_emb, target_block_emb], dim=1)

            hidden_states = full_emb + self.wpe(pos)
            hidden_states = self.drop(hidden_states)

            all_hops = 0
            layer_energy = []
            for block in self.blocks:
                norm_1 = block["ln_1"](hidden_states)
                attn_out, hops, e_trace = block["attn"](norm_1, is_prefix_causal=True, prefix_len=L_pre, T_max=T_max)
                hidden_states = hidden_states + attn_out
                all_hops += hops
                layer_energy.append(e_trace)

                norm_2 = block["ln_2"](hidden_states)
                mlp_out = block["mlp"](norm_2)
                hidden_states = hidden_states + mlp_out

            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits, all_hops / len(self.blocks), layer_energy

        @torch.no_grad()
        def generate_block_debiased(self, prefix_ids, log_unigram_freq, steps=4, debias_weight=0.35, rep_penalty=1.5, temp=0.8):
            """
            Frequency-Debiased Progressive Attractor Diffusion with Repetition Penalty
            """
            self.eval()
            N = self.block_size
            known_tokens = {}
            masked_slots = set(range(N))
            unmask_schedule = [2, 2, 2, 2] # 4 micro-steps of 2 tokens each

            for s_idx, num_to_unmask in enumerate(unmask_schedule):
                dummy_targets = torch.zeros(N, dtype=torch.long, device=prefix_ids.device)
                for slot, t_id in known_tokens.items():
                    dummy_targets[slot] = t_id

                logits, _, _ = self.forward_with_mask_pattern(
                    prefix_ids, dummy_targets, mask_indices=masked_slots, T_max=3
                )
                block_logits = logits[0, -N:, :].clone()

                # 1. Apply Unigram Frequency Debiasing: Subtract alpha * log P(w)
                block_logits = block_logits - debias_weight * log_unigram_freq

                # 2. Apply Repetition Penalty for tokens already emitted in prompt & block
                for seen_id in set(prefix_ids[0].tolist() + list(known_tokens.values())):
                    block_logits[:, seen_id] /= rep_penalty

                probs = F.softmax(block_logits / temp, dim=-1)

                # Rank masked slots by confidence
                confidences = []
                for slot in list(masked_slots):
                    # Top-k nucleus sample or argmax
                    val, pred_id = probs[slot].max(dim=-1)
                    confidences.append((val.item(), slot, pred_id.item()))

                confidences.sort(key=lambda x: x[0], reverse=True)
                commit_count = min(num_to_unmask, len(confidences))

                for c in range(commit_count):
                    _, slot_win, token_win = confidences[c]
                    known_tokens[slot_win] = token_win
                    masked_slots.remove(slot_win)

            final_block = [known_tokens[j] for j in range(N)]
            return final_block

    # 3. Load Pre-Trained Weights
    print("\n[2/5] Initializing SubQ-GPT2 Bidirectional Model...")
    model = SubQBidirectionalDiffusionModel(K=32, T_max=4, block_size=block_size).to(device)
    ckpt_path = "/root/checkpoints/subq_gpt2_best.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Successfully loaded base weights from {ckpt_path}!")

    # 4. Training Loop: 600 Steps of Bidirectional Masked Denoising
    print("\n" + "=" * 125)
    print(f"  [3/5] TRAINING SUBQ BIDIRECTIONAL BLOCK ATTRACTOR (600 Steps)")
    print("  Optimizer: AdamW (lr=2e-4), Cosine Decay, Mixed Precision (FP16)")
    print("=" * 125)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    total_steps = 600
    accum_steps = 4
    prefix_len = 128
    optimizer.zero_grad()

    print(f"{'Step':<10} | {'Loss':<14} | {'Attractor Velocity Trace (||Δs||)':<35}")
    print("-" * 75)

    for step in range(1, total_steps + 1):
        model.train()
        progress = step / total_steps
        lr_now = 1e-5 + 0.5 * (2e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g['lr'] = lr_now

        idx = (step * (prefix_len + block_size)) % (len(train_tokens) - prefix_len - block_size - 1)
        prefix_ids = train_tokens[idx : idx + prefix_len].unsqueeze(0)
        target_ids = train_tokens[idx + prefix_len : idx + prefix_len + block_size]

        num_masked = np.random.randint(1, block_size + 1)
        mask_indices = set(np.random.choice(block_size, size=num_masked, replace=False))

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, _, layer_energy = model.forward_with_mask_pattern(prefix_ids, target_ids, mask_indices, T_max=4)
            block_logits = logits[0, -block_size:, :]
            mask_tensor = torch.tensor([idx in mask_indices for idx in range(block_size)], device=device)
            loss = F.cross_entropy(block_logits[mask_tensor], target_ids[mask_tensor]) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if step % 100 == 0 or step == total_steps:
            final_trace = layer_energy[-1] if len(layer_energy) > 0 else [0.0]
            trace_str = " -> ".join([f"{v:.3f}" for v in final_trace])
            print(f"Step {step:>4}/{total_steps} | Loss: {loss.item()*accum_steps:>8.4f} | Trace: [{trace_str}]")

    # 5. Qualitative Debiased Generation Benchmark
    print("\n" + "=" * 125)
    print("  [4/5] QUALITATIVE DEBIASED BIDIRECTIONAL BLOCK GENERATION BENCHMARK")
    print("=" * 125)

    test_prompts = [
        "In artificial intelligence and deep learning, neural networks",
        "Albert Einstein published his famous theory of general",
        "The President of the United States gave an address regarding",
        "def compute_fibonacci(n):\n    if n <= 1:\n        return"
    ]

    for p in test_prompts:
        p_ids = torch.tensor(tokenizer.encode(p), dtype=torch.long, device=device).unsqueeze(0)
        
        # Test with debiased unmasking
        gen_tokens = model.generate_block_debiased(
            p_ids, log_unigram, steps=4, debias_weight=0.4, rep_penalty=1.6, temp=0.8
        )
        gen_text = tokenizer.decode(gen_tokens)

        print(f"\nPrompt: \"{p}\"")
        print(f"SubQ Debiased 8-Token Settle: \"{gen_text}\"")

    print("\n" + "=" * 125)
    print("Study 29 Completed Successfully!")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_bidirectional_diffusion_experiment.remote()
