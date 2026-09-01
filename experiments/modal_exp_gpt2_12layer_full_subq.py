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

app = modal.App("exp-gpt2-12layer-full-subq", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=2400, volumes={"/root/checkpoints": volume})
def run_12layer_full_gpt2_subq():
    import math
    import time
    import os
    import requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 37: FULL 12-LAYER SUBQ-GPT2 (124M PARAMETERS) — FULL DEPTH TRANSPLANT & AUTOREGRESSIVE GENERATION")
    print("  Replacing Dense All-to-All Causal Attention in ALL 12 Layers with Causal SubQ Logarithmic Routing (O(L*K))")
    print("  Preserving Exact 1-to-1 Weights Across All 12 Layers (124M Parameters, No Destructive Averaging)")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Load Tokenizer & Original Pretrained GPT-2
    print("\n[1/5] Loading Pre-Trained GPT-2 (124M params, 12 layers)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    orig_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    orig_gpt2.eval()

    # 2. Download WikiText-2
    print("\n[2/5] Downloading WikiText-2 Dataset...")
    train_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    val_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"

    train_text = requests.get(train_url).text
    val_text = requests.get(val_url).text

    def encode_corpus(raw_text):
        tokens = tokenizer.encode(raw_text)
        return torch.tensor(tokens, dtype=torch.long, device=device)

    train_tokens = encode_corpus(train_text)
    val_tokens = encode_corpus(val_text)
    print(f"WikiText-2: Train {len(train_tokens):,} | Val {len(val_tokens):,} BPE tokens")

    d_model, n_heads, seq_len = 768, 12, 128
    head_dim = d_model // n_heads
    # Causal offsets: token at position i can ONLY look at past offsets (i - delta)
    causal_offsets = [0, 1, 2, 4, 8, 16, 32, 64]
    K = len(causal_offsets)
    batch_size = 16

    # 3. Full 12-Layer SubQ-GPT2 Architecture
    class CausalSubQSelfAttention(nn.Module):
        def __init__(self, orig_attn_layer):
            super().__init__()
            self.register_buffer("offsets", torch.tensor(causal_offsets, dtype=torch.long, device=device))
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)
            self.temp_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1))

            # Exact 1-to-1 weight copy from GPT-2 Conv1D layer
            with torch.no_grad():
                # GPT-2 uses Conv1D weight of shape [d_model, 3*d_model], transpose to Linear [3*d_model, d_model]
                self.c_attn.weight.copy_(orig_attn_layer.c_attn.weight.t())
                self.c_attn.bias.copy_(orig_attn_layer.c_attn.bias)
                self.c_proj.weight.copy_(orig_attn_layer.c_proj.weight.t())
                self.c_proj.bias.copy_(orig_attn_layer.c_proj.bias)

        def forward(self, x):
            B, L, D = x.shape
            qkv = self.c_attn(x)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
            k = k.view(B, L, n_heads, head_dim).transpose(1, 2)
            v = v.view(B, L, n_heads, head_dim).transpose(1, 2)

            k_s_list, v_s_list, m_list = [], [], []
            for delta_val in self.offsets:
                d = delta_val.item()
                if d >= L:
                    k_s = torch.zeros_like(k); v_s = torch.zeros_like(v)
                    m = torch.zeros(B, 1, L, device=x.device, dtype=torch.bool)
                elif d == 0:
                    k_s, v_s = k, v
                    m = torch.ones(B, 1, L, device=x.device, dtype=torch.bool)
                else:
                    k_s = F.pad(k[:, :, :-d, :], (0, 0, d, 0))
                    v_s = F.pad(v[:, :, :-d, :], (0, 0, d, 0))
                    m = torch.cat([torch.zeros(B, 1, d, device=x.device, dtype=torch.bool), torch.ones(B, 1, L - d, device=x.device, dtype=torch.bool)], dim=-1)
                k_s_list.append(k_s); v_s_list.append(v_s); m_list.append(m)

            K_cand = torch.stack(k_s_list, dim=3)
            V_cand = torch.stack(v_s_list, dim=3)
            valid_mask = torch.stack(m_list, dim=3)

            scores = (q.unsqueeze(3) * K_cand).sum(dim=-1) / math.sqrt(head_dim) * self.temp_scale
            scores = scores.masked_fill(~valid_mask, float("-inf"))
            attn_weights = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)

            out = (attn_weights.unsqueeze(-1) * V_cand).sum(dim=3).transpose(1, 2).contiguous().view(B, L, D)
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
            print("  Transplanting all 12 GPT-2 Layers into 12 Causal SubQ Layers (1-to-1 exact weights)...")
            self.blocks = nn.ModuleList([
                SubQGPT2Block(orig_model.transformer.h[i]) for i in range(12)
            ])
            self.ln_f = orig_model.transformer.ln_f
            self.lm_head = orig_model.lm_head

        def forward(self, input_ids):
            B, L = input_ids.shape
            pos = torch.arange(0, L, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            h = self.wte(input_ids) + self.wpe(pos)
            for block in self.blocks:
                h = block(h)
            h = self.ln_f(h)
            logits = self.lm_head(h)
            return logits

    # 4. Evaluation Function for Causal LM
    def get_causal_batch(data, B=batch_size, L=seq_len):
        starts = torch.randint(0, len(data) - L - 1, (B,))
        x = torch.stack([data[s : s + L] for s in starts])
        y = torch.stack([data[s + 1 : s + L + 1] for s in starts])
        return x, y

    def evaluate_causal_model(model_eval, val_source, num_batches=30):
        model_eval.eval()
        total_loss, total_top1, total_top5, total_tokens = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for _ in range(num_batches):
                x, y = get_causal_batch(val_source, B=batch_size, L=seq_len)
                out = model_eval(x)
                logits = out.logits if hasattr(out, "logits") else out

                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item() * y.numel()

                preds = logits.argmax(dim=-1)
                total_top1 += (preds == y).float().sum().item()

                _, top5 = torch.topk(logits, 5, dim=-1)
                total_top5 += (top5 == y.unsqueeze(-1)).any(dim=-1).float().sum().item()
                total_tokens += y.numel()

        avg_loss = total_loss / max(1, total_tokens)
        top1 = (total_top1 / max(1, total_tokens)) * 100.0
        top5 = (total_top5 / max(1, total_tokens)) * 100.0
        ppl = math.exp(min(avg_loss, 20.0))
        return avg_loss, ppl, top1, top5

    # 5. Evaluate Teacher Oracle
    print("\n" + "=" * 125)
    print("  [3/5] EVALUATING 12-LAYER DENSE GPT-2 ORACLE (124M PARAMS)")
    print("=" * 125)
    o_loss, o_ppl, o_top1, o_top5 = evaluate_causal_model(orig_gpt2, val_tokens, num_batches=40)
    print(f"Original 12L Dense GPT-2: Val Loss: {o_loss:.4f} | Causal PPL: {o_ppl:.2f} | Top-1: {o_top1:.2f}% | Top-5: {o_top5:.2f}%")

    # 6. Zero-Shot Evaluation of Full 12-Layer SubQ-GPT2
    print("\n" + "=" * 125)
    print("  [4/5] ZERO-SHOT 12-LAYER SUBQ-GPT2 (ZERO GRADIENTS)")
    print("=" * 125)
    full_subq_gpt2 = Full12LayerSubQGPT2(orig_gpt2).to(device)
    total_params = sum(p.numel() for p in full_subq_gpt2.parameters())
    print(f"Full 12-Layer SubQ-GPT2 Parameters: {total_params:,}")

    z_loss, z_ppl, z_top1, z_top5 = evaluate_causal_model(full_subq_gpt2, val_tokens, num_batches=40)
    print(f"Zero-Shot 12L SubQ-GPT2: Val Loss: {z_loss:.4f} | Causal PPL: {z_ppl:.2f} | Top-1: {z_top1:.2f}% | Top-5: {z_top5:.2f}%")

    # 7. Rapid Adaptation Training (1,000 Steps on A10G)
    print("\n" + "=" * 125)
    print("  [5/5] RAPID ADAPTATION TRAINING OF 12-LAYER SUBQ-GPT2 (1,000 STEPS)")
    print("=" * 125)

    full_subq_gpt2.wte.requires_grad_(False)
    full_subq_gpt2.wpe.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in full_subq_gpt2.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01
    )
    scaler = torch.amp.GradScaler('cuda')
    full_gpt2_ckpt = "/root/checkpoints/subq_gpt2_12layer_full_best.pt"
    if os.path.exists(full_gpt2_ckpt):
        print(f"\n  Found existing trained checkpoint at {full_gpt2_ckpt}, loading weights...")
        ckpt = torch.load(full_gpt2_ckpt, map_location=device)
        full_subq_gpt2.load_state_dict(ckpt["state_dict"])
        print(f"  Loaded! Saved Val PPL: {ckpt.get('ppl', 'N/A')}, Top-1 Acc: {ckpt.get('top1', 'N/A')}%")
    else:
        total_steps = 1000
        t0 = time.time()
        for step in range(1, total_steps + 1):
            full_subq_gpt2.train()
            progress = step / total_steps
            lr = 1e-5 + 0.5 * (1e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))
            for g in optimizer.param_groups: g['lr'] = lr

            x, y = get_causal_batch(train_tokens, B=batch_size, L=seq_len)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = full_subq_gpt2(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(full_subq_gpt2.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if step % 200 == 0:
                print(f"Step {step:>4}/{total_steps} | Train Loss: {loss.item():.4f} | LR: {lr:.6f} | Elapsed: {time.time()-t0:.1f}s")

        # Save to Modal Volume
        torch.save({
            "step": total_steps,
            "state_dict": full_subq_gpt2.state_dict(),
        }, full_gpt2_ckpt)
        volume.commit()
        print(f"⭐ Full 12-Layer SubQ-GPT2 Checkpoint permanently saved to Modal Volume: {full_gpt2_ckpt}!")

    # Final Evaluation
    f_loss, f_ppl, f_top1, f_top5 = evaluate_causal_model(full_subq_gpt2, val_tokens, num_batches=50)
    print(f"\nFinal Adapted 12-Layer SubQ-GPT2: Val Loss: {f_loss:.4f} | Causal PPL: {f_ppl:.2f} | Top-1: {f_top1:.2f}% | Top-5: {f_top5:.2f}%")

    # 8. Autoregressive Text Generation Demo (50 Tokens Forward)
    print("\n" + "=" * 125)
    print("  QUALITATIVE AUTOREGRESSIVE TEXT GENERATION (50 TOKENS FORWARD)")
    print("=" * 125)

    def generate_autoregressive(model_gen, prompt, max_new_tokens=40, temperature=0.7, top_k=40):
        model_gen.eval()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                cur_ctx = generated[:, -seq_len:]
                out = model_gen(cur_ctx)
                logits = out.logits if hasattr(out, "logits") else out
                next_token_logits = logits[:, -1, :] / temperature
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=-1)

        return tokenizer.decode(generated[0].tolist())

    sample_prompts = [
        "In artificial intelligence, neural networks are designed to",
        "The discovery of gravitational waves revealed that",
        "The history of ancient civilizations shows that",
        "She opened the dusty old book in the library and discovered"
    ]

    for p in sample_prompts:
        print(f"\n[Dense GPT-2 Oracle]:")
        print(generate_autoregressive(orig_gpt2, p, max_new_tokens=35))
        print(f"[Full 12-Layer SubQ-GPT2]:")
        print(generate_autoregressive(full_subq_gpt2, p, max_new_tokens=35))
        print("-" * 100)

    # Scorecard
    print("\n" + "=" * 125)
    print("  FINAL SCORECARD: 12-LAYER DENSE GPT-2 VS. 12-LAYER FULL SUBQ-GPT2")
    print("=" * 125)
    print(f"{'Architecture':<45} | {'Layers':<8} | {'Complexity':<14} | {'Val Loss':<10} | {'Causal PPL':<12} | {'Top-1 Acc':<12} | {'Top-5 Acc':<12}")
    print("-" * 125)
    print(f"{'1. Original GPT-2 124M (Dense Oracle)':<45} | {'12L':<8} | {'O(L^2) Dense':<14} | {o_loss:>8.4f} | {o_ppl:>10.2f} | {o_top1:>10.2f}% | {o_top5:>10.2f}%")
    print(f"{'2. Full 12-Layer SubQ-GPT2 (Zero-Shot)':<45} | {'12L':<8} | {'O(L*K) Causal':<14} | {z_loss:>8.4f} | {z_ppl:>10.2f} | {z_top1:>10.2f}% | {z_top5:>10.2f}%")
    print(f"{'3. Full 12-Layer SubQ-GPT2 (Adapted 1k)':<45} | {'12L':<8} | {'O(L*K) Causal':<14} | {f_loss:>8.4f} | {f_ppl:>10.2f} | {f_top1:>10.2f}% | {f_top5:>10.2f}%")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_12layer_full_gpt2_subq.remote()
