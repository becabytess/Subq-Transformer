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

app = modal.App("exp-harmonic-subq-gpt2-transplant", image=image)

@app.function(gpu="A10G", timeout=3600)
def run_harmonic_subq_gpt2_transplant():
    import math
    import time
    import requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 130)
    print("  STUDY 65: FULL 12-LAYER HARMONIC SUBQ-GPT2 (124M PARAMETERS) TRANSPLANT & BENCHMARK")
    print("  Evaluating Pre-Trained GPT-2 Adapted with Continuous Harmonic Waves vs Fixed Dyadic SubQ vs Dense Oracle")
    print("=" * 130)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 2. Download Datasets
    print("\n[1/4] Downloading Datasets (WikiText-2 In-Domain + Penn Treebank Out-of-Domain)...")
    train_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    val_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
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
    # 3. Model Architectures
    # -------------------------------------------------------------------------

    # A. New Harmonic SubQ Self-Attention Layer for GPT-2
    class HarmonicSubQAttention(nn.Module):
        def __init__(self, orig_attn_layer):
            super().__init__()
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)
            self.init_wave_latent = nn.Parameter(torch.randn(n_heads, num_waves * 4) * 0.1)

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

            with torch.no_grad():
                self.c_attn.weight.copy_(orig_attn_layer.c_attn.weight.t())
                self.c_attn.bias.copy_(orig_attn_layer.c_attn.bias)
                self.c_proj.weight.copy_(orig_attn_layer.c_proj.weight.t())
                self.c_proj.bias.copy_(orig_attn_layer.c_proj.bias)

        def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, **kwargs):
            x = hidden_states
            B, L, D = x.shape
            qkv = self.c_attn(x)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
            k = k.view(B, L, n_heads, head_dim).transpose(1, 2)
            v = v.view(B, L, n_heads, head_dim).transpose(1, 2)

            curr_params = self.init_wave_latent.view(n_heads, num_waves, 4)
            amp = torch.tanh(curr_params[..., 0]).view(1, n_heads, 1, num_waves)
            omega = (F.softplus(curr_params[..., 1]).view(1, n_heads, 1, num_waves) * self.base_freqs)
            phi = (curr_params[..., 2] * math.pi).view(1, n_heads, 1, num_waves)
            decay = (F.softplus(curr_params[..., 3]) * 0.05).view(1, n_heads, 1, num_waves)

            wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
            wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

            topk_vals, past_peak_offsets = torch.topk(wave_1d, k=K_peaks - 1, dim=-1)
            past_peak_offsets = past_peak_offsets + 1

            zero_offset = torch.zeros((B, n_heads, 1), dtype=torch.long, device=x.device)
            zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=x.device)
            peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
            peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

            q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
            target_indices = q_pos - peak_offsets.unsqueeze(2)
            valid_mask = target_indices >= 0
            target_indices_clamped = torch.clamp(target_indices, min=0)

            idx_expanded = target_indices_clamped.unsqueeze(-1).expand(B, n_heads, L, K_peaks, head_dim)
            K_gathered = torch.gather(k.unsqueeze(3).expand(B, n_heads, L, K_peaks, head_dim), dim=2, index=idx_expanded)
            V_gathered = torch.gather(v.unsqueeze(3).expand(B, n_heads, L, K_peaks, head_dim), dim=2, index=idx_expanded)

            scores = (q.unsqueeze(3) * K_gathered).sum(dim=-1) / math.sqrt(head_dim) + peak_vals.unsqueeze(2)
            scores = scores.masked_fill(~valid_mask, -1e4)
            attn_weights = F.softmax(scores, dim=-1) * valid_mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            out = (attn_weights.unsqueeze(-1) * V_gathered).sum(dim=3)
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            out = self.c_proj(out)
            outputs = (out, layer_past)
            if output_attentions:
                outputs += (attn_weights,)
            return outputs

    # B. Old Fixed Dyadic SubQ Self-Attention Layer (Study 38 Baseline)
    class FixedDyadicSubQAttention(nn.Module):
        def __init__(self, orig_attn_layer):
            super().__init__()
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)
            self.register_buffer("offsets", torch.tensor([0, 1, 2, 4, 8, 16, 32, 64], dtype=torch.long))

            with torch.no_grad():
                self.c_attn.weight.copy_(orig_attn_layer.c_attn.weight.t())
                self.c_attn.bias.copy_(orig_attn_layer.c_attn.bias)
                self.c_proj.weight.copy_(orig_attn_layer.c_proj.weight.t())
                self.c_proj.bias.copy_(orig_attn_layer.c_proj.bias)

        def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, **kwargs):
            x = hidden_states
            B, L, D = x.shape
            qkv = self.c_attn(x)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
            k = k.view(B, L, n_heads, head_dim).transpose(1, 2)
            v = v.view(B, L, n_heads, head_dim).transpose(1, 2)

            q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
            target_indices = q_pos - self.offsets.view(1, 1, 1, K_peaks)
            valid_mask = target_indices >= 0
            target_indices_clamped = torch.clamp(target_indices, min=0)

            idx_expanded = target_indices_clamped.expand(B, n_heads, L, K_peaks).unsqueeze(-1).expand(B, n_heads, L, K_peaks, head_dim)
            K_gathered = torch.gather(k.unsqueeze(3).expand(B, n_heads, L, K_peaks, head_dim), dim=2, index=idx_expanded)
            V_gathered = torch.gather(v.unsqueeze(3).expand(B, n_heads, L, K_peaks, head_dim), dim=2, index=idx_expanded)

            scores = (q.unsqueeze(3) * K_gathered).sum(dim=-1) / math.sqrt(head_dim)
            scores = scores.masked_fill(~valid_mask, -1e4)
            attn_weights = F.softmax(scores, dim=-1) * valid_mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            out = (attn_weights.unsqueeze(-1) * V_gathered).sum(dim=3)
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            out = self.c_proj(out)
            outputs = (out, layer_past)
            if output_attentions:
                outputs += (attn_weights,)
            return outputs

    # Transplant Function
    def build_transplanted_gpt2(attn_type="harmonic"):
        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        if attn_type == "dense":
            return base_model.to(device)

        for idx, block in enumerate(base_model.transformer.h):
            if attn_type == "harmonic":
                new_attn = HarmonicSubQAttention(block.attn)
            else:
                new_attn = FixedDyadicSubQAttention(block.attn)
            block.attn = new_attn
        return base_model.to(device)

    # -------------------------------------------------------------------------
    # 4. Evaluation Function
    # -------------------------------------------------------------------------
    def evaluate_model(model, split="val", n_batches=30):
        model.eval()
        total_loss, total_tokens = 0.0, 0
        correct_top1, correct_top5 = 0, 0

        with torch.no_grad():
            for s_idx in range(n_batches):
                bx, by = get_batch(split, step_seed=50000 + s_idx)
                outputs = model(bx, labels=by)
                loss = outputs.loss
                logits = outputs.logits # [B, L, V]

                total_loss += loss.item() * (bx.shape[0] * bx.shape[1])
                total_tokens += bx.shape[0] * bx.shape[1]

                # Accuracy metrics
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
    # 5. Training / Adaptation Loop
    # -------------------------------------------------------------------------
    def train_model(model, name):
        print(f"\n[Training & Adapting {name}] (1,000 Steps on WikiText-2)...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
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
                print(f"  Step {step+1:4d}/{total_steps} | Loss: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")

        elapsed = time.time() - t0
        return elapsed

    # -------------------------------------------------------------------------
    # 6. Benchmark Shootout
    # -------------------------------------------------------------------------
    benchmark_results = [
        {
            "name": "1. Standard 12L Dense GPT-2 (Oracle)",
            "params": 124439808,
            "zero_ppl": 7234.25,
            "val_loss": 4.6461,
            "val_ppl": 104.17,
            "top1_acc": 6.28,
            "top5_acc": 24.91,
            "ptb_loss": 6.0460,
            "ptb_ppl": 422.42,
            "ptb_top1": 6.82,
            "ptb_top5": 20.80,
            "elapsed": 193.7
        }
    ]

    models_to_test = [
        ("2. Old Fixed Dyadic SubQ-GPT2 (Study 38 Baseline, K=8)", "fixed_dyadic"),
        ("3. NEW 12L Harmonic SubQ-GPT2 (Learned Wave Peaks, K=8)", "harmonic"),
    ]

    benchmark_results = []

    for name, m_type in models_to_test:
        torch.manual_seed(42)
        model = build_transplanted_gpt2(attn_type=m_type)
        param_count = sum(p.numel() for p in model.parameters())

        # Zero-shot evaluation before adaptation
        val_loss_zero, ppl_zero, top1_zero, top5_zero = evaluate_model(model, split="val", n_batches=15)
        print(f"\n{name} -> Zero-Shot Val Loss: {val_loss_zero:.4f} | Zero-Shot PPL: {ppl_zero:.2f} | Top-1: {top1_zero:.2f}%")

        # Train / Adapt
        elapsed = train_model(model, name)

        # In-Domain WikiText-2 Evaluation
        val_loss, ppl, top1_acc, top5_acc = evaluate_model(model, split="val", n_batches=30)
        # Out-of-Domain Penn Treebank Evaluation
        ptb_loss, ptb_ppl, ptb_top1, ptb_top5 = evaluate_model(model, split="ptb", n_batches=30)

        print(f"\n---> {name} Results:")
        print(f"     WikiText-2 (In-Domain)  : Loss = {val_loss:.4f} | PPL = {ppl:>6.2f} | Top-1 = {top1_acc:>5.2f}% | Top-5 = {top5_acc:>5.2f}%")
        print(f"     Penn Treebank (Out-Domain): Loss = {ptb_loss:.4f} | PPL = {ptb_ppl:>6.2f} | Top-1 = {ptb_top1:>5.2f}% | Top-5 = {ptb_top5:>5.2f}%")

        # Sample generation
        model.eval()
        prompt_text = "The discovery of gravitational waves showed that"
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = input_ids.clone()
            for _ in range(25):
                out = model(gen_ids)
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
        completion = tokenizer.decode(gen_ids[0].tolist())
        print(f"     Sample Generation: \"{completion.replace(chr(10), ' ')}\"")

        benchmark_results.append({
            "name": name,
            "params": param_count,
            "zero_ppl": ppl_zero,
            "val_loss": val_loss,
            "val_ppl": ppl,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "ptb_loss": ptb_loss,
            "ptb_ppl": ptb_ppl,
            "ptb_top1": ptb_top1,
            "ptb_top5": ptb_top5,
            "elapsed": elapsed
        })

    # Summary Table
    print("\n" + "=" * 140)
    print("  STUDY 65 FINAL SUMMARY: 12-LAYER HARMONIC SUBQ-GPT2 (124M) VS FIXED DYADIC VS DENSE ORACLE")
    print("=" * 140)
    print(f"{'Architecture':<58} | {'Complexity':<12} | {'WikiText PPL':<14} | {'Top-1 Acc':<10} | {'Top-5 Acc':<10} | {'PTB PPL (OOD)':<14} | {'PTB Top-1':<10}")
    print("-" * 140)
    for r in benchmark_results:
        print(f"{r['name']:<58} | {'O(L*K)' if 'SubQ' in r['name'] else 'O(L^2)':<12} | {r['val_ppl']:<14.2f} | {r['top1_acc']:<10.2f}% | {r['top5_acc']:<10.2f}% | {r['ptb_ppl']:<14.2f} | {r['ptb_top1']:<10.2f}%")
    print("=" * 140)

    return benchmark_results

@app.local_entrypoint()
def main():
    run_harmonic_subq_gpt2_transplant.remote()
