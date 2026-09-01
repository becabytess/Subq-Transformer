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

app = modal.App("gpt2-subq-parallel-block-diffusion", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=2400, volumes={"/root/checkpoints": volume})
def run_parallel_block_diffusion_experiment():
    import math
    import time
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import requests
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 27: SUBQ PARALLEL MULTI-TOKEN BLOCK DIFFUSION & ATTRACTOR SETTLING")
    print("  Question: Can we fine-tune SubQ-GPT2 to predict an entire block of N tokens (N=4, N=8) in ONE parallel pass")
    print("            by letting noisy virtual token states relax through T thought hops?")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Load Tokenizer & Dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = 50257
    d_model = 768
    n_heads = 12
    head_dim = 64
    seq_len = 256 # Prefix + Block length
    block_size = 8 # Predict N=8 tokens simultaneously

    print(f"\n[1/5] Fetching Multi-Domain Text Dataset...")
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    text_data = requests.get(url).text
    tokens = tokenizer.encode(text_data)
    data_tensor = torch.tensor(tokens, dtype=torch.long, device=device)

    split = int(len(data_tensor) * 0.9)
    train_tokens = data_tensor[:split]
    val_tokens = data_tensor[split:]
    print(f"Dataset: Train {len(train_tokens):,} | Val {len(val_tokens):,} tokens | Block Size N = {block_size}")

    # 2. SubQ Jump Architecture Definition
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

    class SubQBlockSettlerModel(nn.Module):
        def __init__(self, K=32, T_max=4, eps=0.01):
            super().__init__()
            self.wte = nn.Embedding(vocab_size, d_model)
            self.wpe = nn.Embedding(1024, d_model)
            self.drop = nn.Dropout(0.1)
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

            # Special Learned Mask/Virtual Token Embedding for target positions
            self.mask_token_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

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

        def forward(self, input_ids, num_virtual_tokens=0, T_max=4, eps=0.01):
            B, L = input_ids.shape
            pos = torch.arange(0, L + num_virtual_tokens, dtype=torch.long, device=input_ids.device).unsqueeze(0)

            # Prefix token embeddings
            prefix_emb = self.wte(input_ids) # [B, L, D]

            if num_virtual_tokens > 0:
                # Expand virtual mask tokens with slight initial noise
                virtual_emb = self.mask_token_emb.expand(B, num_virtual_tokens, -1)
                noise = torch.randn_like(virtual_emb) * 0.05
                full_emb = torch.cat([prefix_emb, virtual_emb + noise], dim=1)
            else:
                full_emb = prefix_emb

            hidden_states = full_emb + self.wpe(pos)
            hidden_states = self.drop(hidden_states)

            all_hops = 0
            layer_energy = []
            for block in self.blocks:
                norm_1 = block["ln_1"](hidden_states)
                attn_out, hops, e_trace = block["attn"](norm_1, T_max=T_max, eps=eps)
                hidden_states = hidden_states + attn_out
                all_hops += hops
                layer_energy.append(e_trace)

                norm_2 = block["ln_2"](hidden_states)
                mlp_out = block["mlp"](norm_2)
                hidden_states = hidden_states + mlp_out

            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits, all_hops / len(self.blocks), layer_energy

    # 3. Load Pre-Trained Weights from Modal Volume
    print("\n[2/5] Loading Pre-Trained SubQ-GPT2 Checkpoint from Modal Volume...")
    ckpt_path = "/root/checkpoints/subq_gpt2_best.pt"
    model = SubQBlockSettlerModel(K=32, T_max=4, eps=0.01).to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["state_dict"]
        # Filter mask_token_emb if missing
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded checkpoint from {ckpt_path} (step {ckpt.get('step', 'N/A')})!")
        print(f"Loaded params with missing: {missing}")
    else:
        print("Checkpoint not found in volume, initializing from base GPT-2...")
        gpt2_ref = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        model.wte.weight.data.copy_(gpt2_ref.transformer.wte.weight.data)
        model.wpe.weight.data.copy_(gpt2_ref.transformer.wpe.weight.data)
        model.ln_f.weight.data.copy_(gpt2_ref.transformer.ln_f.weight.data)
        model.ln_f.bias.data.copy_(gpt2_ref.transformer.ln_f.bias.data)
        model.lm_head.weight.data.copy_(gpt2_ref.lm_head.weight.data)

    # 4. Evaluation Function for Parallel Block Prediction
    def evaluate_block_prediction(model_eval, eval_data, N=8, prefix_len=128, num_eval=25):
        model_eval.eval()
        total_top1_acc = 0.0
        total_top5_acc = 0.0
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for i in range(num_eval):
                idx = i * (prefix_len + N)
                if idx + prefix_len + N >= len(eval_data):
                    break

                prefix_ids = eval_data[idx : idx + prefix_len].unsqueeze(0) # [1, L]
                target_ids = eval_data[idx + prefix_len : idx + prefix_len + N] # [N]

                # Run parallel settlement pass with N virtual slots
                logits, _, _ = model_eval(prefix_ids, num_virtual_tokens=N, T_max=4)
                block_logits = logits[0, -N:, :] # [N, vocab_size]

                loss = F.cross_entropy(block_logits, target_ids)
                total_loss += loss.item()

                preds = block_logits.argmax(dim=-1) # [N]
                top1_correct = (preds == target_ids).float().mean().item()
                total_top1_acc += top1_correct

                _, top5 = torch.topk(block_logits, 5, dim=-1)
                top5_correct = (top5 == target_ids.unsqueeze(-1)).any(dim=-1).float().mean().item()
                total_top5_acc += top5_correct
                total_samples += 1

        avg_loss = total_loss / max(1, total_samples)
        avg_top1 = (total_top1_acc / max(1, total_samples)) * 100.0
        avg_top5 = (total_top5_acc / max(1, total_samples)) * 100.0
        avg_ppl = math.exp(min(avg_loss, 20.0))
        return avg_loss, avg_ppl, avg_top1, avg_top5

    # 5. Measure Initial Zero-Shot Block Accuracy
    print("\n[3/5] Measuring Initial Zero-Shot Block Prediction Accuracy (Before Block Tuning)...")
    init_loss, init_ppl, init_top1, init_top5 = evaluate_block_prediction(model, val_tokens, N=block_size, prefix_len=128)
    print(f"  --> Zero-Shot Block (N={block_size}) PPL: {init_ppl:.2f} | Top-1 Accuracy: {init_top1:.2f}% | Top-5 Accuracy: {init_top5:.2f}%\n")

    # 6. Fine-Tuning Loop: Training Parallel Block Attractor Settling (800 Steps)
    print("=" * 125)
    print(f"  [4/5] FINE-TUNING SUBQ PARALLEL BLOCK SETTLER (800 Steps | Block Size N = {block_size})")
    print("  Optimizer: AdamW (lr=2e-4), Cosine Decay, Mixed Precision (FP16)")
    print("=" * 125)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    total_steps = 800
    accum_steps = 4
    prefix_len = 128
    batch_size = 1
    best_acc = init_top1

    print(f"{'Step':<10} | {'Block PPL':<14} | {'Top-1 Acc (%)':<16} | {'Top-5 Acc (%)':<16} | {'Settling Velocity (||Δs||)':<28} | {'Status'}")
    print("-" * 125)

    t0 = time.time()
    optimizer.zero_grad()

    for step in range(1, total_steps + 1):
        model.train()
        
        # Cosine LR
        progress = step / total_steps
        lr_now = 1e-5 + 0.5 * (2e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g['lr'] = lr_now

        idx = (step * (prefix_len + block_size)) % (len(train_tokens) - prefix_len - block_size - 1)
        prefix_ids = train_tokens[idx : idx + prefix_len].unsqueeze(0)
        target_ids = train_tokens[idx + prefix_len : idx + prefix_len + block_size]

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, _, layer_energy = model(prefix_ids, num_virtual_tokens=block_size, T_max=4)
            block_logits = logits[0, -block_size:, :]
            loss = F.cross_entropy(block_logits, target_ids) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if step % 100 == 0 or step == total_steps:
            val_loss, val_ppl, val_top1, val_top5 = evaluate_block_prediction(model, val_tokens, N=block_size, prefix_len=prefix_len)
            
            # Measure settling velocity of final layer
            final_layer_trace = layer_energy[-1] if len(layer_energy) > 0 else [0.0]
            trace_str = " -> ".join([f"{v:.3f}" for v in final_layer_trace])

            if val_top1 > best_acc:
                best_acc = val_top1
                torch.save({
                    "step": step,
                    "state_dict": model.state_dict(),
                    "block_size": block_size,
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "val_ppl": val_ppl,
                }, "/root/checkpoints/subq_block_diffusion_best.pt")
                volume.commit()
                status = "⭐ [New Best Saved]"
            else:
                status = ""

            print(f"Step {step:>4}/{total_steps} | {val_ppl:>10.2f}     | {val_top1:>12.2f}%    | {val_top5:>12.2f}%    | Trace: [{trace_str:<18}] | {status}")

    print(f"\nFine-Tuning completed in {time.time() - t0:.1f}s")

    # 7. Qualitative Parallel Block Generation Benchmark
    print("\n" + "=" * 125)
    print("  [5/5] QUALITATIVE PARALLEL BLOCK INFILLING & MULTI-TOKEN SETTLING DEMO")
    print("=" * 125)

    test_prompts = [
        "In artificial intelligence and machine learning, deep neural",
        "The mathematical foundation of general relativity is based on",
        "def quicksort(arr):\n    if len(arr) <="
    ]

    model.eval()
    with torch.no_grad():
        for p in test_prompts:
            p_ids = torch.tensor(tokenizer.encode(p), dtype=torch.long, device=device).unsqueeze(0)
            
            # Predict 8 tokens in 1 single parallel pass
            logits, _, energy = model(p_ids, num_virtual_tokens=block_size, T_max=4)
            block_logits = logits[0, -block_size:, :]
            pred_tokens = block_logits.argmax(dim=-1).tolist()
            pred_text = tokenizer.decode(pred_tokens)

            print(f"\nPrompt: \"{p}\"")
            print(f"Parallel 8-Token Settle: \"{pred_text}\"")
            print(f"Attractor Settling Energy (Layer 12): {energy[-1]}")

    print("\n" + "=" * 125)
    print(f"Final Model Checkpoint saved to: /root/checkpoints/subq_block_diffusion_best.pt")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_parallel_block_diffusion_experiment.remote()
