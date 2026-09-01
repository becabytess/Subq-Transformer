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

app = modal.App("exp-controlled-gpt2-vs-subq", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=3600, volumes={"/root/checkpoints": volume})
def run_controlled_gpt2_vs_subq():
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
    print("  STUDY 38: RIGOROUS APPLES-TO-APPLES CONTROLLED BENCHMARK")
    print("  12-Layer Dense GPT-2 (O(L^2)) vs. 12-Layer SubQ-GPT2 (O(L*K)) Under Identical Training & Evaluation Budget")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 2. Datasets
    print("\n[1/5] Downloading Datasets (WikiText-2 In-Domain + Penn Treebank Out-of-Domain)...")
    # In-Domain: WikiText-2
    train_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    val_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
    # Out-of-Domain: Penn Treebank test
    ptb_url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"

    train_text = requests.get(train_url).text
    val_text = requests.get(val_url).text
    ptb_text = requests.get(ptb_url).text

    def encode_corpus(raw_text):
        tokens = tokenizer.encode(raw_text)
        return torch.tensor(tokens, dtype=torch.long, device=device)

    train_tokens = encode_corpus(train_text)
    val_tokens = encode_corpus(val_text)
    ptb_tokens = encode_corpus(ptb_text)
    print(f"WikiText-2: Train {len(train_tokens):,} | Val {len(val_tokens):,} BPE tokens")
    print(f"Penn Treebank (Out-of-Domain): {len(ptb_tokens):,} BPE tokens")

    d_model, n_heads, seq_len = 768, 12, 128
    head_dim = d_model // n_heads
    causal_offsets = [0, 1, 2, 4, 8, 16, 32, 64]
    K = len(causal_offsets)
    batch_size = 16
    total_steps = 1000

    # 3. Model Definitions
    class CausalSubQSelfAttention(nn.Module):
        def __init__(self, orig_attn_layer):
            super().__init__()
            self.register_buffer("offsets", torch.tensor(causal_offsets, dtype=torch.long, device=device))
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

    def evaluate_model(model_eval, eval_data, num_batches=40):
        model_eval.eval()
        total_loss, total_top1, total_top5, total_tokens = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for _ in range(num_batches):
                starts = torch.randint(0, len(eval_data) - seq_len - 1, (batch_size,))
                x = torch.stack([eval_data[s : s + seq_len] for s in starts])
                y = torch.stack([eval_data[s + 1 : s + seq_len + 1] for s in starts])

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

    # 4. Train Dense GPT-2 (Baseline Control)
    print("\n" + "=" * 125)
    print("  [2/5] TRAINING MODEL A: 12-LAYER DENSE GPT-2 (124M PARAMS, O(L^2) ATTENTION)")
    print(f"  Exact Training Budget: 1,000 Steps on WikiText-2 Train | LR=1e-4 -> 1e-5 Cosine")
    print("=" * 125)

    dense_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    dense_gpt2.transformer.wte.requires_grad_(False)
    dense_gpt2.transformer.wpe.requires_grad_(False)

    opt_dense = torch.optim.AdamW([p for p in dense_gpt2.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    scaler_dense = torch.amp.GradScaler('cuda')

    # Fix seed for training batch synchronization
    torch.manual_seed(42)
    cached_batches = []
    for _ in range(total_steps):
        starts = torch.randint(0, len(train_tokens) - seq_len - 1, (batch_size,))
        x = torch.stack([train_tokens[s : s + seq_len] for s in starts])
        y = torch.stack([train_tokens[s + 1 : s + seq_len + 1] for s in starts])
        cached_batches.append((x, y))

    t0 = time.time()
    for step in range(1, total_steps + 1):
        dense_gpt2.train()
        progress = step / total_steps
        lr = 1e-5 + 0.5 * (1e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))
        for g in opt_dense.param_groups: g['lr'] = lr

        x, y = cached_batches[step - 1]
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = dense_gpt2(x).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        scaler_dense.scale(loss).backward()
        scaler_dense.unscale_(opt_dense)
        torch.nn.utils.clip_grad_norm_(dense_gpt2.parameters(), 1.0)
        scaler_dense.step(opt_dense)
        scaler_dense.update()
        opt_dense.zero_grad()

        if step % 200 == 0:
            print(f"Dense GPT-2 | Step {step:>4}/{total_steps} | Train Loss: {loss.item():.4f} | LR: {lr:.6f} | Elapsed: {time.time()-t0:.1f}s")

    # Evaluate Dense GPT-2
    dense_in_loss, dense_in_ppl, dense_in_top1, dense_in_top5 = evaluate_model(dense_gpt2, val_tokens, num_batches=50)
    dense_out_loss, dense_out_ppl, dense_out_top1, dense_out_top5 = evaluate_model(dense_gpt2, ptb_tokens, num_batches=50)

    print(f"\nDense GPT-2 Results:")
    print(f"  In-Domain WikiText-2:  Loss: {dense_in_loss:.4f} | PPL: {dense_in_ppl:.2f} | Top-1: {dense_in_top1:.2f}% | Top-5: {dense_in_top5:.2f}%")
    print(f"  Out-of-Domain PennTB:  Loss: {dense_out_loss:.4f} | PPL: {dense_out_ppl:.2f} | Top-1: {dense_out_top1:.2f}% | Top-5: {dense_out_top5:.2f}%")

    # 5. Train SubQ-GPT2 (Experiment)
    print("\n" + "=" * 125)
    print("  [3/5] TRAINING MODEL B: 12-LAYER SUBQ-GPT2 (124M PARAMS, O(L*K) CAUSAL WAVE ROUTING)")
    print(f"  Exact Identical Batches, Seeds, and Budget: 1,000 Steps on WikiText-2 Train | LR=1e-4 -> 1e-5 Cosine")
    print("=" * 125)

    base_for_subq = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    subq_gpt2 = Full12LayerSubQGPT2(base_for_subq).to(device)
    subq_gpt2.wte.requires_grad_(False)
    subq_gpt2.wpe.requires_grad_(False)

    opt_subq = torch.optim.AdamW([p for p in subq_gpt2.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    scaler_subq = torch.amp.GradScaler('cuda')

    t0 = time.time()
    for step in range(1, total_steps + 1):
        subq_gpt2.train()
        progress = step / total_steps
        lr = 1e-5 + 0.5 * (1e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))
        for g in opt_subq.param_groups: g['lr'] = lr

        x, y = cached_batches[step - 1] # identical batch!
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = subq_gpt2(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        scaler_subq.scale(loss).backward()
        scaler_subq.unscale_(opt_subq)
        torch.nn.utils.clip_grad_norm_(subq_gpt2.parameters(), 1.0)
        scaler_subq.step(opt_subq)
        scaler_subq.update()
        opt_subq.zero_grad()

        if step % 200 == 0:
            print(f"SubQ GPT-2  | Step {step:>4}/{total_steps} | Train Loss: {loss.item():.4f} | LR: {lr:.6f} | Elapsed: {time.time()-t0:.1f}s")

    # Evaluate SubQ GPT-2
    subq_in_loss, subq_in_ppl, subq_in_top1, subq_in_top5 = evaluate_model(subq_gpt2, val_tokens, num_batches=50)
    subq_out_loss, subq_out_ppl, subq_out_top1, subq_out_top5 = evaluate_model(subq_gpt2, ptb_tokens, num_batches=50)

    print(f"\nSubQ GPT-2 Results:")
    print(f"  In-Domain WikiText-2:  Loss: {subq_in_loss:.4f} | PPL: {subq_in_ppl:.2f} | Top-1: {subq_in_top1:.2f}% | Top-5: {subq_in_top5:.2f}%")
    print(f"  Out-of-Domain PennTB:  Loss: {subq_out_loss:.4f} | PPL: {subq_out_ppl:.2f} | Top-1: {subq_out_top1:.2f}% | Top-5: {subq_out_top5:.2f}%")

    # 6. Qualitative Comparison on Test Prompts
    print("\n" + "=" * 125)
    print("  [4/5] QUALITATIVE AUTOREGRESSIVE GENERATION COMPARISON (BOTH FINE-TUNED)")
    print("=" * 125)

    def generate(model_gen, prompt, max_new_tokens=40, temp=0.7, top_k=40):
        model_gen.eval()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                cur_ctx = generated[:, -seq_len:]
                out = model_gen(cur_ctx)
                logits = out.logits if hasattr(out, "logits") else out
                next_logits = logits[:, -1, :] / temp
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=-1)
        return tokenizer.decode(generated[0].tolist())

    test_prompts = [
        "In artificial intelligence, neural networks are designed to",
        "The discovery of gravitational waves revealed that",
        "The history of ancient civilizations shows that",
        "She opened the dusty old book in the library and discovered"
    ]

    for p in test_prompts:
        print(f"\nPrompt: \"{p}\"")
        print(f"  [Fine-Tuned Dense GPT-2]: {generate(dense_gpt2, p, max_new_tokens=35)}")
        print(f"  [Fine-Tuned SubQ GPT-2]:  {generate(subq_gpt2, p, max_new_tokens=35)}")

    # 7. Final Controlled Scorecard
    print("\n" + "=" * 125)
    print("  [5/5] FINAL CONTROLLED APPLES-TO-APPLES BENCHMARK SCORECARD")
    print("=" * 125)
    print(f"{'Evaluation Dataset':<25} | {'Model Architecture':<28} | {'Complexity':<14} | {'Val Loss':<10} | {'PPL':<10} | {'Top-1 Acc':<12} | {'Top-5 Acc':<12}")
    print("-" * 125)
    print(f"{'WikiText-2 (In-Domain)':<25} | {'12L Dense GPT-2 (Fine-Tuned)':<28} | {'O(L^2) Dense':<14} | {dense_in_loss:>8.4f} | {dense_in_ppl:>8.2f} | {dense_in_top1:>10.2f}% | {dense_in_top5:>10.2f}%")
    print(f"{'WikiText-2 (In-Domain)':<25} | {'12L SubQ-GPT2 (Fine-Tuned)':<28} | {'O(L*K) Wave':<14} | {subq_in_loss:>8.4f} | {subq_in_ppl:>8.2f} | {subq_in_top1:>10.2f}% | {subq_in_top5:>10.2f}%")
    print("-" * 125)
    print(f"{'PennTreebank (Out-Domain)':<25} | {'12L Dense GPT-2 (Fine-Tuned)':<28} | {'O(L^2) Dense':<14} | {dense_out_loss:>8.4f} | {dense_out_ppl:>8.2f} | {dense_out_top1:>10.2f}% | {dense_out_top5:>10.2f}%")
    print(f"{'PennTreebank (Out-Domain)':<25} | {'12L SubQ-GPT2 (Fine-Tuned)':<28} | {'O(L*K) Wave':<14} | {subq_out_loss:>8.4f} | {subq_out_ppl:>8.2f} | {subq_out_top1:>10.2f}% | {subq_out_top5:>10.2f}%")
    print("=" * 125)

    # Save Checkpoints
    torch.save(dense_gpt2.state_dict(), "/root/checkpoints/dense_gpt2_controlled_1k.pt")
    torch.save(subq_gpt2.state_dict(), "/root/checkpoints/subq_gpt2_controlled_1k.pt")
    volume.commit()
    print("⭐ Both controlled checkpoints permanently saved to Modal volume.")

@app.local_entrypoint()
def main():
    run_controlled_gpt2_vs_subq.remote()
