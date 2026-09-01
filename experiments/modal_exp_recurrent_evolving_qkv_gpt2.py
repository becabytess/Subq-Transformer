import modal
import os

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

app = modal.App("exp-recurrent-evolving-qkv-gpt2", image=image)

@app.function(gpu="A10G", timeout=3600)
def run_recurrent_evolving_qkv_gpt2():
    import math
    import time
    import requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 130)
    print("  STUDY 66: PRE-TRAINED GPT-2 WITH RECURRENT HARMONIC SUBQ & FULL EVOLVING Q, K, V")
    print("  Evaluating Dynamic Re-projection of Q, K, V Across Recurrent Thought Iterations (T = 1, 2, 4, 8)")
    print("=" * 130)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Tokenizer & Datasets
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    val_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
    ptb_url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"

    train_text = requests.get(train_url).text
    val_text = requests.get(val_url).text
    ptb_text = requests.get(ptb_url).text

    def encode_corpus(raw_text):
        return torch.tensor(tokenizer.encode(raw_text), dtype=torch.long, device=device)

    train_tokens = encode_corpus(train_text)
    val_tokens = encode_corpus(val_text)
    ptb_tokens = encode_corpus(ptb_text)

    d_model, n_heads, seq_len = 768, 12, 128
    head_dim = d_model // n_heads
    K_peaks = 8
    num_waves = 12
    max_d = 128
    batch_size = 16
    total_steps = 1000

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_tokens if split == "train" else (val_tokens if split == "val" else ptb_tokens)
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x, y

    # -------------------------------------------------------------------------
    # Recurrent Harmonic SubQ Block with Full Evolving Q, K, V
    # -------------------------------------------------------------------------
    class RecurrentEvolvingQKVGPT2(nn.Module):
        def __init__(self, pretrained_gpt2, T=4):
            super().__init__()
            self.T = T
            self.wte = pretrained_gpt2.transformer.wte
            self.wpe = pretrained_gpt2.transformer.wpe

            # Take weights from pre-trained Layer 0
            layer0 = pretrained_gpt2.transformer.h[0]
            self.ln_1 = nn.LayerNorm(d_model)
            self.ln_2 = nn.LayerNorm(d_model)
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)
            self.mlp_c_fc = nn.Linear(d_model, 4 * d_model)
            self.mlp_c_proj = nn.Linear(4 * d_model, d_model)
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, 50257, bias=False)

            with torch.no_grad():
                self.ln_1.weight.copy_(layer0.ln_1.weight)
                self.ln_1.bias.copy_(layer0.ln_1.bias)
                self.ln_2.weight.copy_(layer0.ln_2.weight)
                self.ln_2.bias.copy_(layer0.ln_2.bias)
                self.c_attn.weight.copy_(layer0.attn.c_attn.weight.t())
                self.c_attn.bias.copy_(layer0.attn.c_attn.bias)
                self.c_proj.weight.copy_(layer0.attn.c_proj.weight.t())
                self.c_proj.bias.copy_(layer0.attn.c_proj.bias)
                self.mlp_c_fc.weight.copy_(layer0.mlp.c_fc.weight.t())
                self.mlp_c_fc.bias.copy_(layer0.mlp.c_fc.bias)
                self.mlp_c_proj.weight.copy_(layer0.mlp.c_proj.weight.t())
                self.mlp_c_proj.bias.copy_(layer0.mlp.c_proj.bias)
                self.ln_f.weight.copy_(pretrained_gpt2.transformer.ln_f.weight)
                self.ln_f.bias.copy_(pretrained_gpt2.transformer.ln_f.bias)
                self.lm_head.weight.copy_(pretrained_gpt2.lm_head.weight)

            # Wave router parameters
            init_latents = torch.zeros(n_heads, num_waves, 4)
            init_latents[..., 0] = 0.5
            init_latents[..., 3] = 0.1
            self.init_wave_latent = nn.Parameter(init_latents.view(n_heads, num_waves * 4))
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4)
            )

            log_freqs = torch.linspace(math.log10(math.pi / 1.0), math.log10(math.pi / 32.0), num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

        def forward(self, input_ids, labels=None):
            B, L = input_ids.shape
            pos = torch.arange(0, L, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            s = self.wte(input_ids) + self.wpe(pos)

            curr_wave = self.init_wave_latent
            inv_sqrt_T = 1.0 / math.sqrt(self.T)

            # Unroll T recurrent iterations with full evolving Q, K, V
            for t in range(self.T):
                # 1. LayerNorm input state
                s_norm = self.ln_1(s)

                # 2. Project FULL EVOLVING Q, K, V from evolving state s_norm
                qkv = self.c_attn(s_norm)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
                k = k.view(B, L, n_heads, head_dim).transpose(1, 2)
                v = v.view(B, L, n_heads, head_dim).transpose(1, 2)

                # 3. Compute continuous wave carrier
                params = curr_wave.view(n_heads, num_waves, 4)
                amp = torch.tanh(params[..., 0]).view(1, n_heads, 1, num_waves)
                omega = (F.softplus(params[..., 1]).view(1, n_heads, 1, num_waves) * self.base_freqs)
                phi = (params[..., 2] * math.pi).view(1, n_heads, 1, num_waves)
                decay = (F.softplus(params[..., 3]) * 0.05).view(1, n_heads, 1, num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

                # 4. Extract Top-K peaks
                topk_vals, past_peak_offsets = torch.topk(wave_1d, k=K_peaks - 1, dim=-1)
                past_peak_offsets = past_peak_offsets + 1
                zero_offset = torch.zeros((B, n_heads, 1), dtype=torch.long, device=input_ids.device)
                zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=input_ids.device)
                peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
                peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

                # 5. Gather Keys & Values for peak offsets
                q_pos = torch.arange(L, device=input_ids.device).view(1, 1, L, 1)
                target_indices = q_pos - peak_offsets.unsqueeze(2)
                valid_mask = target_indices >= 0
                target_indices_clamped = torch.clamp(target_indices, min=0)

                idx_exp = target_indices_clamped.unsqueeze(-1).expand(B, n_heads, L, K_peaks, head_dim)
                K_g = torch.gather(k.unsqueeze(3).expand(B, n_heads, L, K_peaks, head_dim), dim=2, index=idx_exp)
                V_g = torch.gather(v.unsqueeze(3).expand(B, n_heads, L, K_peaks, head_dim), dim=2, index=idx_exp)

                # 6. Attention with Harmonic Logit Bias
                scores = (q.unsqueeze(3) * K_g).sum(dim=-1) / math.sqrt(head_dim) + peak_vals.unsqueeze(2)
                scores = scores.masked_fill(~valid_mask, -1e4)
                attn = F.softmax(scores, dim=-1) * valid_mask.float()
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

                attn_out = (attn.unsqueeze(-1) * V_g).sum(dim=3)
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, d_model)
                attn_out = self.c_proj(attn_out)

                # 7. Recurrent Residual Accumulation
                s = s + inv_sqrt_T * attn_out

                # 8. MLP block
                m_norm = self.ln_2(s)
                mlp_out = self.mlp_c_proj(F.gelu(self.mlp_c_fc(m_norm)))
                s = s + inv_sqrt_T * mlp_out

                # 9. Dynamic wave transition for next hop
                if t < self.T - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            # Final LayerNorm & Output Head
            s_final = self.ln_f(s)
            logits = self.lm_head(s_final)

            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            class Output:
                pass
            res = Output()
            res.logits = logits
            res.loss = loss
            return res

    def evaluate_model(model, split="val", n_batches=30):
        model.eval()
        total_loss, total_tokens = 0.0, 0
        correct_top1, correct_top5 = 0, 0

        with torch.no_grad():
            for s_idx in range(n_batches):
                bx, by = get_batch(split, step_seed=50000 + s_idx)
                outputs = model(bx, labels=by)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item() * (bx.shape[0] * bx.shape[1])
                total_tokens += bx.shape[0] * bx.shape[1]

                preds = logits.view(-1, logits.shape[-1])
                targets = by.view(-1)
                _, top1_idx = preds.topk(1, dim=-1)
                _, top5_idx = preds.topk(5, dim=-1)

                correct_top1 += (top1_idx.squeeze(-1) == targets).sum().item()
                correct_top5 += (top5_idx == targets.unsqueeze(-1)).any(dim=-1).sum().item()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        top1_acc = (correct_top1 / total_tokens) * 100.0
        top5_acc = (correct_top5 / total_tokens) * 100.0
        return avg_loss, ppl, top1_acc, top5_acc

    # -------------------------------------------------------------------------
    # Train / Benchmark across T = 1, 2, 4, 8 Hops
    # -------------------------------------------------------------------------
    base_pretrained_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

    t_depths = [1, 2, 4, 8]
    results = []

    for T_val in t_depths:
        torch.manual_seed(42)
        model = RecurrentEvolvingQKVGPT2(base_pretrained_gpt2, T=T_val).to(device)
        param_count = sum(p.numel() for p in model.parameters())

        print(f"\n" + "=" * 110)
        print(f"  Benchmarking Recurrent Evolving Q,K,V GPT-2 at Thought Depth T = {T_val} (Params: {param_count:,})")
        print("=" * 110)

        # Zero-shot evaluation
        val_loss_zero, ppl_zero, top1_zero, top5_zero = evaluate_model(model, split="val", n_batches=15)
        print(f"  Zero-Shot Val Loss: {val_loss_zero:.4f} | Zero-Shot PPL: {ppl_zero:.2f} | Top-1: {top1_zero:.2f}%")

        # Training / Adaptation (1,000 steps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

        t0 = time.time()
        for step in range(total_steps):
            model.train()
            bx, by = get_batch("train", step_seed=1000 + step)
            optimizer.zero_grad()
            outputs = model(bx, labels=by)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            if (step + 1) % 250 == 0 or step == total_steps - 1:
                print(f"    Step {step+1:4d}/{total_steps} | Loss: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")

        elapsed = time.time() - t0

        # In-Domain & Out-of-Domain Evaluation
        val_loss, ppl, top1_acc, top5_acc = evaluate_model(model, split="val", n_batches=30)
        ptb_loss, ptb_ppl, ptb_top1, ptb_top5 = evaluate_model(model, split="ptb", n_batches=30)

        print(f"  --> Final T={T_val} Results:")
        print(f"      WikiText-2 (In-Domain)  : Loss = {val_loss:.4f} | PPL = {ppl:>6.2f} | Top-1 = {top1_acc:>5.2f}% | Top-5 = {top5_acc:>5.2f}%")
        print(f"      Penn Treebank (Out-Domain): Loss = {ptb_loss:.4f} | PPL = {ptb_ppl:>6.2f} | Top-1 = {ptb_top1:>5.2f}% | Top-5 = {ptb_top5:>5.2f}%")

        results.append({
            "T": T_val,
            "params": param_count,
            "val_loss": val_loss,
            "val_ppl": ppl,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "ptb_loss": ptb_loss,
            "ptb_ppl": ptb_ppl,
            "ptb_top1": ptb_top1,
            "elapsed": elapsed
        })

    # Summary
    print("\n" + "=" * 135)
    print("  STUDY 66 FINAL SUMMARY: RECURRENT EVOLVING Q, K, V GPT-2 ACROSS THOUGHT DEPTHS (T = 1, 2, 4, 8)")
    print("=" * 135)
    print(f"{'Thought Depth T':<18} | {'Physical Params':<18} | {'WikiText-2 PPL':<18} | {'Top-1 Acc':<14} | {'Top-5 Acc':<14} | {'PTB PPL (OOD)':<16} | {'Adapt Time':<12}")
    print("-" * 135)
    for r in results:
        print(f"T = {r['T']:<14} | {r['params']:<18,d} | {r['val_ppl']:<18.2f} | {r['top1_acc']:<12.2f}% | {r['top5_acc']:<12.2f}% | {r['ptb_ppl']:<16.2f} | {r['elapsed']:>6.1f}s")
    print("=" * 135)

    return results

@app.local_entrypoint()
def main():
    run_recurrent_evolving_qkv_gpt2.remote()
