import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "requests",
        "accelerate"
    )
)

app = modal.App("subq-bidirectional-wave-lattice", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=3000, volumes={"/root/checkpoints": volume})
def run_bidirectional_wave_experiment():
    import math
    import time
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import requests
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 30: BIDIRECTIONAL SUBQ WAVE LATTICE VS. MULTI-LAYER BERT (FROM SCRATCH)")
    print("  Question: Does an omnidirectional SubQ Wave Lattice (1 Layer, K=17, T=4) outperform causal models and")
    print("            match/beat a 4-Layer Dense BERT on Masked Reconstruction from scratch?")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Preparation (TinyShakespeare Char-Level Tokenization)
    print("\n[1/5] Downloading & Preparing TinyShakespeare...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    chars = sorted(list(set(text)))
    vocab_size = len(chars) + 2 # +1 for [MASK], +1 for [PAD]
    mask_token_id = len(chars)
    pad_token_id = len(chars) + 1

    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    idx2char[mask_token_id] = "[M]"
    idx2char[pad_token_id] = "[P]"

    data_ids = [char2idx[c] for c in text]
    data_tensor = torch.tensor(data_ids, dtype=torch.long, device=device)

    split = int(len(data_tensor) * 0.9)
    train_data = data_tensor[:split]
    val_data = data_tensor[split:]
    print(f"Dataset: Train {len(train_data):,} | Val {len(val_data):,} chars | Vocab: {vocab_size} (Mask ID: {mask_token_id})")

    # Hyperparameters
    d_model = 256
    n_heads = 8
    seq_len = 256
    mask_ratio = 0.20 # 20% BERT-style masking
    batch_size = 16
    total_steps = 1500

    # -------------------------------------------------------------
    # 2. Architectures Definition
    # -------------------------------------------------------------

    # A. Standard Dense BERT Block
    class DenseBertBlock(nn.Module):
        def __init__(self, d_model=256, n_heads=8):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )

        def forward(self, x):
            norm_x = self.ln1(x)
            attn_out, _ = self.attn(norm_x, norm_x, norm_x) # Dense all-to-all bidirectional
            x = x + attn_out
            norm_x2 = self.ln2(x)
            mlp_out = self.mlp(norm_x2)
            x = x + mlp_out
            return x

    class DenseBertLM(nn.Module):
        def __init__(self, num_layers=1, d_model=256, n_heads=8):
            super().__init__()
            self.wte = nn.Embedding(vocab_size, d_model)
            self.wpe = nn.Embedding(seq_len, d_model)
            self.layers = nn.ModuleList([DenseBertBlock(d_model, n_heads) for _ in range(num_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            B, L = x.shape
            pos = torch.arange(0, L, dtype=torch.long, device=x.device).unsqueeze(0)
            h = self.wte(x) + self.wpe(pos)
            for layer in self.layers:
                h = layer(h)
            h = self.ln_f(h)
            logits = self.lm_head(h)
            return logits, 1, [[0.0]]

    # B. SubQ Omnidirectional Wave Lattice Attention
    class SubQWaveAttention(nn.Module):
        def __init__(self, d_model=256, n_heads=8, is_bidirectional=True, T_max=4):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.is_bidirectional = is_bidirectional
            self.T_max = T_max

            if is_bidirectional:
                # Symmetrical Wave Menu (Past & Future)
                offsets = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]
            else:
                # Causal Backward Jumps Only
                offsets = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]

            self.K = len(offsets)
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long, device=device))
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)
            self.temp_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1))

            # Recurrent Dynamical Relaxation Processor (GRU)
            self.w_ih = nn.Linear(d_model, 3 * d_model)
            self.w_gate_h = nn.Linear(d_model, 2 * d_model)
            self.w_cand_h = nn.Linear(d_model, d_model)

        def forward(self, x, T_max=None):
            B, L, D = x.shape
            T_max = T_max or self.T_max
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

                k_s_list = []
                v_s_list = []
                m_list = []

                for delta_val in self.offsets:
                    d = delta_val.item()
                    if d >= 0:
                        # Backward / past jump
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
                        # Forward / future jump (Bidirectional wave)
                        d_abs = abs(d)
                        if d_abs >= L:
                            k_s = torch.zeros_like(k)
                            v_s = torch.zeros_like(v)
                            m = torch.zeros(B, 1, L, device=x.device, dtype=torch.bool)
                        else:
                            k_s = F.pad(k[:, :, d_abs:, :], (0, 0, 0, d_abs))
                            v_s = F.pad(v[:, :, d_abs:, :], (0, 0, 0, d_abs))
                            m = torch.cat([
                                torch.ones(B, 1, L - d_abs, device=x.device, dtype=torch.bool),
                                torch.zeros(B, 1, d_abs, device=x.device, dtype=torch.bool)
                            ], dim=-1)

                    k_s_list.append(k_s)
                    v_s_list.append(v_s)
                    m_list.append(m)

                K_cand = torch.stack(k_s_list, dim=3)
                V_cand = torch.stack(v_s_list, dim=3)
                valid_mask = torch.stack(m_list, dim=3)

                q_exp = q.unsqueeze(3)
                scores = (q_exp * K_cand).sum(dim=-1) / math.sqrt(self.head_dim)
                scores = scores * self.temp_scale
                scores = scores.masked_fill(~valid_mask, float("-inf"))
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

                out_h = (attn_weights.unsqueeze(-1) * V_cand).sum(dim=3)
                out = out_h.transpose(1, 2).contiguous().view(B, L, D)
                context = self.c_proj(out)

                # Recurrent message passing relaxation step
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

    class SubQWaveLM(nn.Module):
        def __init__(self, is_bidirectional=True, d_model=256, n_heads=8, T_max=4):
            super().__init__()
            self.wte = nn.Embedding(vocab_size, d_model)
            self.wpe = nn.Embedding(seq_len, d_model)
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = SubQWaveAttention(d_model=d_model, n_heads=n_heads, is_bidirectional=is_bidirectional, T_max=T_max)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size)

        def forward(self, x, T_max=None):
            B, L = x.shape
            pos = torch.arange(0, L, dtype=torch.long, device=x.device).unsqueeze(0)
            h = self.wte(x) + self.wpe(pos)

            norm_1 = self.ln1(h)
            attn_out, hops, e_trace = self.attn(norm_1, T_max=T_max)
            h = h + attn_out

            norm_2 = self.ln2(h)
            mlp_out = self.mlp(norm_2)
            h = h + mlp_out

            h = self.ln_f(h)
            logits = self.lm_head(h)
            return logits, hops, [e_trace]

    # -------------------------------------------------------------
    # 3. Batch Sampling & Masking Function
    # -------------------------------------------------------------
    def get_batch(data, B=batch_size, L=seq_len, mask_p=mask_ratio):
        starts = torch.randint(0, len(data) - L - 1, (B,))
        raw_seqs = torch.stack([data[s : s + L] for s in starts]) # [B, L]
        
        # 20% BERT-style masking
        mask_matrix = torch.rand(B, L, device=device) < mask_p
        masked_inputs = raw_seqs.clone()
        masked_inputs[mask_matrix] = mask_token_id

        return masked_inputs, raw_seqs, mask_matrix

    def evaluate_model(model_eval, val_source, num_batches=30, T_eval=None):
        model_eval.eval()
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_tokens = 0

        with torch.no_grad():
            for b in range(num_batches):
                inp, targets, mask = get_batch(val_source, B=batch_size, L=seq_len, mask_p=mask_ratio)
                if isinstance(model_eval, SubQWaveLM):
                    logits, _, _ = model_eval(inp, T_max=T_eval)
                else:
                    logits, _, _ = model_eval(inp)

                masked_logits = logits[mask]
                masked_targets = targets[mask]

                loss = F.cross_entropy(masked_logits, masked_targets)
                total_loss += loss.item() * masked_targets.numel()

                preds = masked_logits.argmax(dim=-1)
                top1_acc = (preds == masked_targets).float().sum().item()
                total_top1 += top1_acc

                _, top5 = torch.topk(masked_logits, 5, dim=-1)
                top5_acc = (top5 == masked_targets.unsqueeze(-1)).any(dim=-1).float().sum().item()
                total_top5 += top5_acc

                total_tokens += masked_targets.numel()

        avg_loss = total_loss / max(1, total_tokens)
        avg_top1 = (total_top1 / max(1, total_tokens)) * 100.0
        avg_top5 = (total_top5 / max(1, total_tokens)) * 100.0
        avg_ppl = math.exp(min(avg_loss, 20.0))
        return avg_loss, avg_ppl, avg_top1, avg_top5

    # -------------------------------------------------------------
    # 4. Train All 4 Models Head-to-Head
    # -------------------------------------------------------------
    models_to_test = {
        "1. Standard BERT (1L)": DenseBertLM(num_layers=1, d_model=d_model, n_heads=n_heads).to(device),
        "2. Standard BERT (4L)": DenseBertLM(num_layers=4, d_model=d_model, n_heads=n_heads).to(device),
        "3. Causal SubQ (1L, T=4)": SubQWaveLM(is_bidirectional=False, d_model=d_model, n_heads=n_heads, T_max=4).to(device),
        "4. Bidirectional SubQ Wave (1L, T=4)": SubQWaveLM(is_bidirectional=True, d_model=d_model, n_heads=n_heads, T_max=4).to(device)
    }

    results_table = []
    print("\n" + "=" * 125)
    print("  [2/5] TRAINING 4 ARCHITECTURES HEAD-TO-HEAD ON 20% MASKED RECONSTRUCTION (1,500 STEPS EACH)")
    print("=" * 125)

    for name, model in models_to_test.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n---> Training {name} (Params: {n_params:,})...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        scaler = torch.amp.GradScaler('cuda')
        best_top1 = 0.0

        t0 = time.time()
        for step in range(1, total_steps + 1):
            model.train()
            progress = step / total_steps
            lr_now = 1e-5 + 0.5 * (5e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))
            for g in optimizer.param_groups:
                g['lr'] = lr_now

            inp, targets, mask = get_batch(train_data, B=batch_size, L=seq_len, mask_p=mask_ratio)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, _, _ = model(inp)
                masked_logits = logits[mask]
                masked_targets = targets[mask]
                loss = F.cross_entropy(masked_logits, masked_targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        elapsed = time.time() - t0
        val_loss, val_ppl, val_top1, val_top5 = evaluate_model(model, val_data, num_batches=40)
        print(f"     Finished in {elapsed:.1f}s | Val Loss: {val_loss:.4f} | Mask PPL: {val_ppl:.2f} | Top-1 Acc: {val_top1:.2f}% | Top-5 Acc: {val_top5:.2f}%")

        # Save Bidirectional SubQ Wave Checkpoint Permanently
        if "Bidirectional SubQ" in name:
            ckpt_path = "/root/checkpoints/subq_bidirectional_wave_best.pt"
            torch.save({
                "step": total_steps,
                "state_dict": model.state_dict(),
                "val_top1": val_top1,
                "val_top5": val_top5,
                "val_ppl": val_ppl,
                "d_model": d_model,
                "n_heads": n_heads,
            }, ckpt_path)
            volume.commit()
            print(f"     ⭐ Model permanently saved to Modal Volume: {ckpt_path}!")

        results_table.append({
            "name": name,
            "params": n_params,
            "val_loss": val_loss,
            "ppl": val_ppl,
            "top1": val_top1,
            "top5": val_top5,
            "time": elapsed
        })

    # -------------------------------------------------------------
    # 5. Omnidirectional Wave Reverberation Depth Sweep (T = 1..6)
    # -------------------------------------------------------------
    print("\n" + "=" * 125)
    print("  [3/5] WAVE REVERBERATION DEPTH SWEEP ON BIDIRECTIONAL SUBQ (T = 1 -> 6)")
    print("=" * 125)
    wave_model = models_to_test["4. Bidirectional SubQ Wave (1L, T=4)"]
    
    print(f"{'Wave Depth (T)':<16} | {'Mask PPL':<14} | {'Top-1 Acc (%)':<16} | {'Top-5 Acc (%)':<16} | {'Physical Velocity Trace':<35}")
    print("-" * 105)

    for t_hop in [1, 2, 3, 4, 6]:
        v_loss, v_ppl, v_top1, v_top5 = evaluate_model(wave_model, val_data, num_batches=30, T_eval=t_hop)
        
        # Sample one batch to inspect velocity trace
        inp, targets, mask = get_batch(val_data, B=4, L=seq_len)
        with torch.no_grad():
            _, _, e_traces = wave_model(inp, T_max=t_hop)
        trace_str = " -> ".join([f"{v:.3f}" for v in e_traces[0]])

        print(f"T = {t_hop:<12} | {v_ppl:>10.2f}     | {v_top1:>12.2f}%    | {v_top5:>12.2f}%    | [{trace_str}]")

    # -------------------------------------------------------------
    # 6. Qualitative Mask Infilling Comparison
    # -------------------------------------------------------------
    print("\n" + "=" * 125)
    print("  [4/5] QUALITATIVE BIDIRECTIONAL MASK RECONSTRUCTION DEMO")
    print("=" * 125)

    test_sentences = [
        "To be, or not to be, that is the question:",
        "Friends, Romans, countrymen, lend me your ears;",
        "Now is the winter of our discontent",
        "Romeo, Romeo! wherefore art thou Romeo?"
    ]

    for sent in test_sentences:
        clean_ids = [char2idx.get(c, 0) for c in sent]
        # Pad to seq_len
        if len(clean_ids) < seq_len:
            clean_ids = clean_ids + [pad_token_id] * (seq_len - len(clean_ids))
        clean_tensor = torch.tensor(clean_ids[:seq_len], dtype=torch.long, device=device).unsqueeze(0)

        # Corrupt 25% of the characters with [MASK]
        corrupted = clean_tensor.clone()
        mask_idx = torch.rand(corrupted.shape, device=device) < 0.25
        # Don't mask pad tokens
        mask_idx[:, len(sent):] = False
        corrupted[mask_idx] = mask_token_id

        # Reconstruct with 4L BERT vs 1L Bidirectional SubQ Wave
        with torch.no_grad():
            bert_logits, _, _ = models_to_test["2. Standard BERT (4L)"](corrupted)
            subq_logits, _, _ = models_to_test["4. Bidirectional SubQ Wave (1L, T=4)"](corrupted, T_max=4)

        bert_pred = bert_logits.argmax(dim=-1)[0][:len(sent)].tolist()
        subq_pred = subq_logits.argmax(dim=-1)[0][:len(sent)].tolist()

        corrupt_str = "".join([idx2char[t] for t in corrupted[0][:len(sent)].tolist()])
        bert_str = "".join([idx2char[t] for t in bert_pred])
        subq_str = "".join([idx2char[t] for t in subq_pred])

        print(f"\nOriginal:   \"{sent}\"")
        print(f"Corrupted:  \"{corrupt_str}\"")
        print(f"4L BERT:    \"{bert_str}\"")
        print(f"SubQ Wave:  \"{subq_str}\"")

    # -------------------------------------------------------------
    # 7. Final Summary Scorecard
    # -------------------------------------------------------------
    print("\n" + "=" * 125)
    print("  [5/5] FINAL BENCHMARK SCORECARD: BIDIRECTIONAL WAVE LATTICE VS. BERT")
    print("=" * 125)
    print(f"{'Architecture':<38} | {'Parameters':<12} | {'Val Loss':<10} | {'Mask PPL':<10} | {'Top-1 Acc':<12} | {'Top-5 Acc':<12}")
    print("-" * 110)
    for r in results_table:
        print(f"{r['name']:<38} | {r['params']:>10,} | {r['val_loss']:>8.4f} | {r['ppl']:>8.2f} | {r['top1']:>10.2f}% | {r['top5']:>10.2f}%")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_bidirectional_wave_experiment.remote()
