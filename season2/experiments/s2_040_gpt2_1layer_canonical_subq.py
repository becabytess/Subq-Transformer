import json
import math
import os
import pathlib
import sys
import time
import modal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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

app = modal.App("exp-s2-040-gpt2-1layer-canonical", image=image)

@app.function(image=image, gpu="A10G", timeout=2400)
def run_gpt2_1layer_canonical(k_peaks: int = 8, T_hops: int = 8):
    import math
    import time
    import requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, AutoTokenizer

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 120, flush=True)
    print(f"  STUDY S2-040: 1-LAYER RECURRENT GPT-2 TRANSPLANT WITH CANONICAL SEASON 2 SUBQ", flush=True)
    print(f"  Configuration: 1 Physical Block (Layer 0, 85M Params) | T = {T_hops} Pure Linear Attention Hops", flush=True)
    print(f"  K = {k_peaks} Peaks | 1 Shared Wave Across Heads | MLP Executed Strictly Once at Block End", flush=True)
    print(f"  Device: {torch.cuda.get_device_name(0)} (24GB VRAM)", flush=True)
    print("=" * 120, flush=True)

    # -------------------------------------------------------------------------
    # 1. Tokenizer & Datasets (WikiText-2 In-Domain + PTB Out-of-Domain)
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    val_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
    ptb_url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"

    print("[1/4] Downloading WikiText-2 & Penn Treebank...", flush=True)
    train_text = requests.get(train_url).text
    val_text = requests.get(val_url).text
    ptb_text = requests.get(ptb_url).text

    def encode_corpus(raw_text):
        tokens = tokenizer.encode(raw_text)
        return torch.tensor(tokens, dtype=torch.long, device=device)

    train_tokens = encode_corpus(train_text)
    val_tokens = encode_corpus(val_text)
    ptb_tokens = encode_corpus(ptb_text)
    print(f"  WikiText-2: Train {len(train_tokens):,} | Val {len(val_tokens):,} tokens", flush=True)
    print(f"  Penn Treebank: {len(ptb_tokens):,} tokens", flush=True)

    d_model = 768
    n_heads = 12
    head_dim = d_model // n_heads  # 64
    seq_len = 128
    batch_size = 16
    total_steps = 1000
    num_waves = 12
    max_d = 128

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_tokens if split == "train" else (val_tokens if split == "val" else ptb_tokens)
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x, y

    # -------------------------------------------------------------------------
    # 2. 1-Layer Recurrent SubQ Model with End-of-Block MLP
    # -------------------------------------------------------------------------
    class Canonical1LRecurrentGPT2(nn.Module):
        """
        1-Layer Recurrent GPT-2 with Season 2 Canonical Architecture:
        - Takes Layer 0 weights from pre-trained GPT-2 (wte, wpe, ln_1, c_attn, c_proj, ln_2, mlp, ln_f, lm_head)
        - 1 Shared wave generator broadcast across all 12 heads (Standard MHA subspace splitting)
        - Strictly K=8 candidate peaks
        - T_hops of pure linear attention accumulation: s <- s + (1 / sqrt(T)) * attn_out
        - 1 Single MLP execution strictly at the end of the recurrent block: s <- s + mlp(ln_2(s))
        - Parameters: ~85M (32% reduction vs 124M dense GPT-2)
        """
        def __init__(self, pretrained_gpt2, T_hops=8, k_peaks=8):
            super().__init__()
            self.T = T_hops
            self.k_peaks = k_peaks
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = head_dim
            self.scale = 1.0 / math.sqrt(head_dim)
            self.num_waves = num_waves
            self.max_d = max_d

            # Embeddings
            self.wte = pretrained_gpt2.transformer.wte
            self.wpe = pretrained_gpt2.transformer.wpe

            # Take weights from pre-trained Layer 0
            layer0 = pretrained_gpt2.transformer.h[0]
            self.ln_1 = layer0.ln_1
            self.c_attn = layer0.attn.c_attn
            self.c_proj = layer0.attn.c_proj
            self.ln_2 = layer0.ln_2
            self.mlp = layer0.mlp

            # Final LayerNorm & LM Head
            self.ln_f = pretrained_gpt2.transformer.ln_f
            self.lm_head = pretrained_gpt2.lm_head

            # 1 Shared continuous harmonic wave generator for the block
            init_latents = torch.zeros(1, num_waves, 4)
            init_latents[..., 0] = 0.5   # initial amplitude
            init_latents[..., 3] = 0.1   # initial decay
            self.init_wave_latent = nn.Parameter(init_latents.view(1, num_waves * 4))
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4)
            )

            # Buffers
            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))
            self.register_buffer("q_pos", torch.arange(seq_len).unsqueeze(1))  # (L, 1)

        def precompute_all_wave_offsets(self, device):
            waves = [self.init_wave_latent]
            for _ in range(self.T - 1):
                waves.append(waves[-1] + 0.1 * self.wave_transition(waves[-1]))
            all_waves = torch.cat(waves, dim=0)  # (T, num_waves * 4)

            curr_params = all_waves.view(self.T, self.num_waves, 4)
            amp = torch.tanh(curr_params[..., 0]).view(self.T, 1, 1, self.num_waves)
            omega = (F.softplus(curr_params[..., 1]).view(self.T, 1, 1, self.num_waves) * self.base_freqs)
            phi = (curr_params[..., 2] * math.pi).view(self.T, 1, 1, self.num_waves)
            decay = (F.softplus(curr_params[..., 3]) * 0.05).view(self.T, 1, 1, self.num_waves)

            wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
            wave_1d = wave_comps.sum(dim=-1)  # (T, 1, max_d - 1)

            topk_vals, past_offsets = torch.topk(wave_1d, k=self.k_peaks - 1, dim=-1)
            past_offsets = past_offsets + 1  # 1-indexed relative offsets

            zero_off = torch.zeros((self.T, 1, 1), dtype=torch.long, device=device)
            zero_val = torch.zeros((self.T, 1, 1), dtype=torch.float, device=device)

            all_offsets = torch.cat([zero_off, past_offsets], dim=-1).squeeze(1)  # (T, K)
            all_vals = torch.cat([zero_val, topk_vals], dim=-1).squeeze(1)        # (T, K)
            return all_offsets, all_vals

        def forward(self, input_ids, labels=None):
            B, L = input_ids.shape
            device = input_ids.device
            pos = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0)
            s = self.wte(input_ids) + self.wpe(pos)

            inv_sqrt_T = 1.0 / math.sqrt(self.T)
            all_offsets, all_vals = self.precompute_all_wave_offsets(device)
            q_pos = self.q_pos[:L]

            # 1. Pure Linear Attention Thought Loop (T Hops, NO in-loop MLP)
            for t in range(self.T):
                z = self.ln_1(s)

                # Project Q, K, V
                qkv = self.c_attn(z)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

                # Candidate indexing
                offsets = all_offsets[t].unsqueeze(0)
                target_indices = q_pos - offsets
                valid_mask = (target_indices >= 0)
                clamped_indices = torch.clamp(target_indices, min=0)

                # Direct Tensor Slicing
                K_g = k[:, :, clamped_indices, :]
                V_g = v[:, :, clamped_indices, :]

                # Attention dot product with harmonic bias
                peak_val = all_vals[t].view(1, 1, 1, self.k_peaks)
                scores = (q.unsqueeze(3) * K_g).sum(dim=-1) * self.scale + peak_val

                mask_4d = valid_mask.view(1, 1, L, self.k_peaks)
                scores = scores.masked_fill(~mask_4d, -1e4)
                attn_weights = F.softmax(scores, dim=-1) * mask_4d.float()
                attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

                context = (attn_weights.unsqueeze(-1) * V_g).sum(dim=3)
                context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
                attn_out = self.c_proj(context)

                # Contractive residual accumulation
                s = s + inv_sqrt_T * attn_out

            # 2. MLP Executed STRICTLY ONCE at Block End (Canonical Season 2 Schedule)
            s = s + self.mlp(self.ln_2(s))

            # 3. Final Output
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

    # -------------------------------------------------------------------------
    # 3. Model Construction
    # -------------------------------------------------------------------------
    print("\n[2/4] Collapsing Pre-Trained GPT-2 into 1-Layer Canonical SubQ Block...", flush=True)
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    orig_params = sum(p.numel() for p in base_model.parameters())

    model = Canonical1LRecurrentGPT2(base_model, T_hops=T_hops, k_peaks=k_peaks).to(device)
    new_params = sum(p.numel() for p in model.parameters())
    print(f"  Successfully built 1-Layer Canonical Recurrent SubQ-GPT2 (T={T_hops}, K={k_peaks})", flush=True)
    print(f"  Parameters: Original 12L GPT-2 = {orig_params:,} | 1L Canonical SubQ = {new_params:,} ({(new_params/orig_params)*100:.1f}% of original size)", flush=True)

    # -------------------------------------------------------------------------
    # 4. Evaluation Helper (Bfloat16 AMP)
    # -------------------------------------------------------------------------
    def evaluate_model(eval_model, split="val", n_batches=30):
        eval_model.eval()
        total_loss, total_tokens = 0.0, 0
        correct_top1, correct_top5 = 0, 0

        with torch.no_grad():
            for s_idx in range(n_batches):
                bx, by = get_batch(split, step_seed=50000 + s_idx)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = eval_model(bx, labels=by)
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

    # Zero-shot evaluation
    print("\n[3/4] Evaluating Zero-Shot Performance (0 Adaptation Steps)...", flush=True)
    zero_val_loss, zero_val_ppl, zero_top1, zero_top5 = evaluate_model(model, split="val", n_batches=15)
    print(f"  Zero-Shot WikiText-2 Val Loss : {zero_val_loss:.4f} | Val PPL : {zero_val_ppl:.2f}", flush=True)
    print(f"  Zero-Shot Top-1 Accuracy       : {zero_top1:.2f}% | Top-5 : {zero_top5:.2f}%", flush=True)

    # -------------------------------------------------------------------------
    # 5. Fast Adaptation Training with Bfloat16 AMP (1,000 steps on WikiText-2)
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Fast Adaptation on WikiText-2 ({total_steps} steps, lr=1e-4, batch=16, L=128, Bfloat16 AMP)...", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    t0 = time.time()
    for step in range(total_steps):
        model.train()
        bx, by = get_batch("train", step_seed=1000 + step)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(bx, labels=by)
            loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        if (step + 1) % 200 == 0 or step == total_steps - 1:
            tok_per_sec = ((step + 1) * batch_size * seq_len) / (time.time() - t0)
            print(f"  Step {step+1:4d}/{total_steps} | Loss: {loss.item():.4f} | Tok/s: {tok_per_sec:.0f} | Elapsed: {time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t0

    # Final In-Domain & Out-of-Domain Evaluation
    print("\nEvaluating Adapted Model on WikiText-2 & Penn Treebank...", flush=True)
    val_loss, ppl, top1_acc, top5_acc = evaluate_model(model, split="val", n_batches=30)
    ptb_loss, ptb_ppl, ptb_top1, ptb_top5 = evaluate_model(model, split="ptb", n_batches=30)

    # Sample generation
    model.eval()
    prompt_text = "The discovery of gravitational waves showed that"
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = input_ids.clone()
        for _ in range(25):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(gen_ids)
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
    completion = tokenizer.decode(gen_ids[0].tolist())

    res = {
        "architecture": f"1L Canonical Recurrent SubQ-GPT2 (K={k_peaks}, T={T_hops}, End-MLP)",
        "k_peaks": k_peaks,
        "T_hops": T_hops,
        "parameters": new_params,
        "zero_shot_loss": round(zero_val_loss, 4),
        "zero_shot_ppl": round(zero_val_ppl, 2),
        "zero_shot_top1": round(zero_top1, 2),
        "zero_shot_top5": round(zero_top5, 2),
        "val_loss": round(val_loss, 4),
        "val_ppl": round(ppl, 2),
        "top1_acc": round(top1_acc, 2),
        "top5_acc": round(top5_acc, 2),
        "ptb_loss": round(ptb_loss, 4),
        "ptb_ppl": round(ptb_ppl, 2),
        "ptb_top1": round(ptb_top1, 2),
        "ptb_top5": round(ptb_top5, 2),
        "training_time_s": round(elapsed, 1),
        "throughput_tok_s": round((total_steps * batch_size * seq_len) / elapsed, 1),
        "sample_generation": completion.replace("\n", " ")
    }

    print("\n" + "=" * 120, flush=True)
    print(f"  S2-040 RESULTS: 1L CANONICAL SUBQ-GPT2 (K={k_peaks}, T={T_hops}, END-MLP)", flush=True)
    print("=" * 120, flush=True)
    print(f"  WikiText-2 In-Domain  : Loss = {val_loss:.4f} | PPL = {ppl:>6.2f} | Top-1 = {top1_acc:>5.2f}% | Top-5 = {top5_acc:>5.2f}%", flush=True)
    print(f"  Penn Treebank (OOD)   : Loss = {ptb_loss:.4f} | PPL = {ptb_ppl:>6.2f} | Top-1 = {ptb_top1:>5.2f}% | Top-5 = {ptb_top5:>5.2f}%", flush=True)
    print(f"  Sample Generation     : \"{completion.replace(chr(10), ' ')}\"", flush=True)
    print(f"  Throughput            : {(total_steps * batch_size * seq_len) / elapsed:.0f} tokens/sec ({elapsed:.1f}s)", flush=True)
    print("=" * 120, flush=True)

    return res

@app.local_entrypoint()
def main():
    print("Launching Study S2-040 on Modal A10G...", flush=True)
    t0 = time.time()

    res = run_gpt2_1layer_canonical.remote(k_peaks=8, T_hops=8)

    total_time = time.time() - t0

    print("\n" + "=" * 135)
    print("  STUDY S2-040 FINAL SCORECARD: 1-LAYER CANONICAL SUBQ-GPT2 VS PAST BENCHMARKS")
    print("=" * 135)
    print(f"{'Model Architecture':<48} | {'Complexity':<12} | {'Params':<12} | {'WikiText PPL':<14} | {'Top-1 Acc':<12} | {'PTB PPL (OOD)':<14}")
    print("-" * 135)
    print(f"{'Study 65: 12L Dense GPT-2 Baseline':<48} | {'O(L^2) Dense':<12} | {'124,439,808':<12} | {'104.17':<14} | {'6.28%':<12} | {'422.42':<14}")
    print(f"{'Study 65: 12L Stacked SubQ (T=1, K=8)':<48} | {'O(L*K) Sparse':<12} | {'124,513,536':<12} | {'141.45':<14} | {'-':<12} | {'-':<14}")
    print(f"{'Study 66: 1L In-Loop MLP SubQ (T=8, K=8)':<48} | {'O(L*K) Sparse':<12} | {'85,074,320':<12} | {'88.62':<14} | {'31.46%':<12} | {'467.82':<14}")
    print(f"{'Study S2-039: 12L Stacked SubQ (T=8, K=8, 96H)':<48} | {'O(L*K) Sparse':<12} | {'124,496,640':<12} | {'275.15':<14} | {'4.71%':<12} | {'1079.54':<14}")
    print(f"{res['architecture']:<48} | {'O(L*K) Sparse':<12} | {res['parameters']:<12,d} | {res['val_ppl']:<14.2f} | {res['top1_acc']:<10.2f}% | {res['ptb_ppl']:<14.2f}")
    print("=" * 135)

    os.makedirs("season2/results", exist_ok=True)
    out_json = "season2/results/s2_040_gpt2_1layer_canonical_subq.json"
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[Saved S2-040 results to {out_json}] (Total runtime: {total_time:.1f}s)")
