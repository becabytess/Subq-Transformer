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

app = modal.App("gpt2-subq-iterative-block-diffusion", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=3000, volumes={"/root/checkpoints": volume})
def run_iterative_block_diffusion_experiment():
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
    print("  STUDY 28: SUBQ ITERATIVE ATTRACTOR DIFFUSION (SLOT EMBEDDINGS + DYNAMIC MASKED DENOISING)")
    print("  Goals:")
    print("    1. Break spatial symmetry using position-specific learned slot embeddings (slot_0 ... slot_7).")
    print("    2. Train on dynamic masking ratios (predicting k in [1, 8] tokens conditioned on partial context).")
    print("    3. Evaluate progressive confidence unmasking generation: generating 8 tokens in 3 parallel diffusion steps!")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset & Tokenizer
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
    print(f"Dataset: Train {len(train_tokens):,} | Val {len(val_tokens):,} tokens | Block Size N = {block_size}")

    # 2. SubQ Architecture with Slot-Specific Mask Embeddings
    def get_menu(K=32):
        local_k = 8
        local_offsets = list(range(local_k))
        long_jumps = []
        val = local_k
        while len(local_offsets) + len(long_jumps) < K and val < seq_len:
            long_jumps.append(val)
            val = int(val * 1.4) + 1
        menu = local_offsets + long_jumps
        while len(menu) < K:
            menu.append(menu[-1] + 1)
        return torch.tensor(menu[:K], dtype=torch.long, device=device)

    class SubQJumpAttention(nn.Module):
        def __init__(self, d_model=768, n_heads=12, K=32, T_max=4, eps=0.01):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.K = K
            self.T_max = T_max
            self.eps = eps

            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)
            self.register_buffer("offsets", get_menu(K))
            self.temp_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1))

            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=True)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=True)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=True)

        def forward(self, x, T_max=None, eps=None):
            B, L, D = x.shape
            T_max = T_max or self.T_max
            eps = eps or self.eps

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
                    if d >= L:
                        k_shifted = torch.zeros_like(k)
                        v_shifted = torch.zeros_like(v)
                        mask = torch.zeros(B, 1, L, device=x.device, dtype=torch.bool)
                    elif d == 0:
                        k_shifted = k
                        v_shifted = v
                        mask = torch.ones(B, 1, L, device=x.device, dtype=torch.bool)
                    else:
                        k_shifted = F.pad(k[:, :, :-d, :], (0, 0, d, 0))
                        v_shifted = F.pad(v[:, :, :-d, :], (0, 0, d, 0))
                        mask = torch.cat([
                            torch.zeros(B, 1, d, device=x.device, dtype=torch.bool),
                            torch.ones(B, 1, L - d, device=x.device, dtype=torch.bool)
                        ], dim=-1)

                    k_shifted_list.append(k_shifted)
                    v_shifted_list.append(v_shifted)
                    valid_mask_list.append(mask)

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

                vel = torch.norm(s - s_prev, p=2, dim=-1).mean().item()
                energy_trace.append(vel)

            return s, hops_taken, energy_trace

    class SubQIterativeDiffusionModel(nn.Module):
        def __init__(self, K=32, T_max=4, eps=0.01, block_size=8):
            super().__init__()
            self.block_size = block_size
            self.wte = nn.Embedding(vocab_size, d_model)
            self.wpe = nn.Embedding(1024, d_model)
            self.drop = nn.Dropout(0.1)
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

            # Slot-Specific Learned Mask Embeddings: [block_size, d_model]
            self.slot_embeddings = nn.Parameter(torch.randn(block_size, d_model) * 0.02)

            self.blocks = nn.ModuleList()
            for l in range(12):
                subq_attn = SubQJumpAttention(d_model=768, n_heads=12, K=K, T_max=T_max, eps=eps)
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
            """
            Forward pass where target positions are either real token embeddings (if unmasked)
            or slot embeddings (if masked in mask_indices).
            """
            B, L_pre = prefix_ids.shape
            N = len(target_ids)
            total_len = L_pre + N

            pos = torch.arange(0, total_len, dtype=torch.long, device=prefix_ids.device).unsqueeze(0)
            prefix_emb = self.wte(prefix_ids) # [B, L_pre, D]

            # Build target block embeddings
            target_emb_list = []
            for j in range(N):
                if j in mask_indices:
                    # Use slot-specific mask embedding + small jitter
                    slot_vec = self.slot_embeddings[j].unsqueeze(0).unsqueeze(0) # [1, 1, D]
                    target_emb_list.append(slot_vec)
                else:
                    # Use true token embedding
                    tok_id = target_ids[j].view(1, 1)
                    tok_vec = self.wte(tok_id)
                    target_emb_list.append(tok_vec)

            target_block_emb = torch.cat(target_emb_list, dim=1) # [1, N, D]
            full_emb = torch.cat([prefix_emb, target_block_emb], dim=1)

            hidden_states = full_emb + self.wpe(pos)
            hidden_states = self.drop(hidden_states)

            all_hops = 0
            layer_energy = []
            for block in self.blocks:
                norm_1 = block["ln_1"](hidden_states)
                attn_out, hops, e_trace = block["attn"](norm_1, T_max=T_max, eps=0.01)
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
        def generate_block_progressive(self, prefix_ids, steps=3, T_hops_per_step=3, temperature=0.7):
            """
            Progressive Confidence Unmasking (MaskGIT-style):
            Generates block_size=8 tokens in `steps` parallel diffusion passes.
            """
            self.eval()
            N = self.block_size
            known_tokens = {} # slot_idx -> token_id
            masked_slots = set(range(N))

            # Schedule: unmask tokens progressively (e.g. 2, then 3, then 3 = 8 tokens)
            unmask_schedule = [2, 3, 3]

            for s_idx, num_to_unmask in enumerate(unmask_schedule):
                # Run forward pass conditioned on currently known tokens
                dummy_targets = torch.zeros(N, dtype=torch.long, device=prefix_ids.device)
                for slot, t_id in known_tokens.items():
                    dummy_targets[slot] = t_id

                logits, _, _ = self.forward_with_mask_pattern(
                    prefix_ids, dummy_targets, mask_indices=masked_slots, T_max=T_hops_per_step
                )

                block_logits = logits[0, -N:, :] # [N, vocab_size]
                probs = F.softmax(block_logits / temperature, dim=-1)

                # For all currently masked slots, find confidence scores
                confidences = []
                for slot in list(masked_slots):
                    max_p, pred_id = probs[slot].max(dim=-1)
                    confidences.append((max_p.item(), slot, pred_id.item()))

                # Sort by confidence descending
                confidences.sort(key=lambda x: x[0], reverse=True)

                # Commit the top `num_to_unmask` most confident slots
                commit_count = min(num_to_unmask, len(confidences))
                for c in range(commit_count):
                    _, slot_win, token_win = confidences[c]
                    known_tokens[slot_win] = token_win
                    masked_slots.remove(slot_win)

            # Assemble final block
            final_block = [known_tokens[j] for j in range(N)]
            return final_block

    # 3. Load Pre-Trained Weights
    print("\n[2/5] Initializing SubQ-GPT2 and Loading Weights from Modal Volume...")
    model = SubQIterativeDiffusionModel(K=32, T_max=4, block_size=block_size).to(device)
    ckpt_path = "/root/checkpoints/subq_gpt2_best.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Loaded base model weights from {ckpt_path}!")

    # 4. Evaluation Function
    def evaluate_masked_prediction(model_eval, eval_data, prefix_len=128, num_eval=30):
        model_eval.eval()
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_evaluated = 0

        with torch.no_grad():
            for i in range(num_eval):
                idx = i * (prefix_len + block_size)
                if idx + prefix_len + block_size >= len(eval_data):
                    break

                prefix_ids = eval_data[idx : idx + prefix_len].unsqueeze(0)
                target_ids = eval_data[idx + prefix_len : idx + prefix_len + block_size]

                # Test 100% masked block
                all_masks = set(range(block_size))
                logits, _, _ = model_eval.forward_with_mask_pattern(prefix_ids, target_ids, all_masks, T_max=4)
                block_logits = logits[0, -block_size:, :]

                loss = F.cross_entropy(block_logits, target_ids)
                total_loss += loss.item()

                preds = block_logits.argmax(dim=-1)
                top1_acc = (preds == target_ids).float().mean().item()
                total_top1 += top1_acc

                _, top5 = torch.topk(block_logits, 5, dim=-1)
                top5_acc = (top5 == target_ids.unsqueeze(-1)).any(dim=-1).float().mean().item()
                total_top5 += top5_acc
                total_evaluated += 1

        avg_loss = total_loss / max(1, total_evaluated)
        avg_top1 = (total_top1 / max(1, total_evaluated)) * 100.0
        avg_top5 = (total_top5 / max(1, total_evaluated)) * 100.0
        avg_ppl = math.exp(min(avg_loss, 20.0))
        return avg_loss, avg_ppl, avg_top1, avg_top5

    # 5. Training Loop: Dynamic Masking Denoising (1,000 Steps)
    print("\n" + "=" * 125)
    print(f"  [3/5] TRAINING SUBQ ITERATIVE BLOCK DIFFUSION (1,000 Steps | Dynamic Masking Ratios)")
    print("  Optimizer: AdamW (lr=2.5e-4), Cosine Decay, Mixed Precision (FP16)")
    print("=" * 125)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    total_steps = 1000
    accum_steps = 4
    prefix_len = 128
    best_top1 = 0.0

    print(f"{'Step':<10} | {'Block PPL':<14} | {'Top-1 Acc (%)':<16} | {'Top-5 Acc (%)':<16} | {'Status'}")
    print("-" * 95)

    t0 = time.time()
    optimizer.zero_grad()

    for step in range(1, total_steps + 1):
        model.train()
        
        progress = step / total_steps
        lr_now = 1e-5 + 0.5 * (2.5e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g['lr'] = lr_now

        idx = (step * (prefix_len + block_size)) % (len(train_tokens) - prefix_len - block_size - 1)
        prefix_ids = train_tokens[idx : idx + prefix_len].unsqueeze(0)
        target_ids = train_tokens[idx + prefix_len : idx + prefix_len + block_size]

        # Sample random number of slots to mask: k in [1, 8]
        num_masked = np.random.randint(1, block_size + 1)
        mask_indices = set(np.random.choice(block_size, size=num_masked, replace=False))

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, _, _ = model.forward_with_mask_pattern(prefix_ids, target_ids, mask_indices, T_max=4)
            block_logits = logits[0, -block_size:, :]
            
            # Compute loss only on masked positions
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
            val_loss, val_ppl, val_top1, val_top5 = evaluate_masked_prediction(model, val_tokens, prefix_len=prefix_len)

            if val_top1 > best_top1:
                best_top1 = val_top1
                torch.save({
                    "step": step,
                    "state_dict": model.state_dict(),
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "val_ppl": val_ppl,
                }, "/root/checkpoints/subq_iterative_diffusion_best.pt")
                volume.commit()
                status = "⭐ [New Best Saved to Volume]"
            else:
                status = ""

            print(f"Step {step:>4}/{total_steps} | {val_ppl:>10.2f}     | {val_top1:>12.2f}%    | {val_top5:>12.2f}%    | {status}")

    print(f"\nTraining completed in {time.time() - t0:.1f}s")

    # 6. Comprehensive Progressive Unmasking Generation Benchmark
    print("\n" + "=" * 125)
    print("  [4/5] PROGRESSIVE ATTRACTOR DIFFUSION GENERATION DEMO (3 MICRO-STEPS -> 8 TOKENS)")
    print("=" * 125)

    prompts = [
        "In artificial intelligence, deep neural networks are designed to",
        "The theory of relativity explains how gravity affects space and",
        "The President of the United States announced a new policy on",
        "def binary_search(arr, target):\n    left, right ="
    ]

    model.eval()
    for p in prompts:
        p_ids = torch.tensor(tokenizer.encode(p), dtype=torch.long, device=device).unsqueeze(0)
        
        # Progressive 3-step unmasking
        gen_tokens = model.generate_block_progressive(p_ids, steps=3, T_hops_per_step=3, temperature=0.7)
        gen_text = tokenizer.decode(gen_tokens)

        print(f"\n--- Prompt: \"{p}\" ---")
        print(f"SubQ 8-Token Progressive Settling: \"{gen_text}\"")

    print("\n" + "=" * 125)
    print("Checkpoint saved permanently to: /root/checkpoints/subq_iterative_diffusion_best.pt")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_iterative_block_diffusion_experiment.remote()
