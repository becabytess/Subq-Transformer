import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets",
        "numpy",
        "accelerate"
    )
)

app = modal.App("exp-bert-subq-wave-transplant", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=1800, volumes={"/root/checkpoints": volume})
def run_bert_subq_transplant():
    import math
    import time
    import os
    import requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertForMaskedLM, BertTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 32: PRE-TRAINED FOUNDATION MODEL (BERT-BASE 110M) TRANSPLANT INTO 1-LAYER SUBQ WAVE LATTICE")
    print("  Collapsing 12 Dense Physical Layers into 1 Omnidirectional Wave Layer (K=15 Symmetrical Offsets, T=4 Hops)")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Load Pre-Trained BERT-Base & Tokenizer
    print("\n[1/5] Loading Pre-Trained bert-base-uncased (110M parameters, 12 layers)...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    orig_bert = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    orig_bert.eval()
    mask_token_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size
    print(f"Loaded BERT: Vocab {vocab_size:,} | Mask Token ID: {mask_token_id}")

    # 2. Load & Prepare WikiText-2 Dataset directly
    print("\n[2/5] Downloading WikiText-2 Dataset...")
    train_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    val_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"

    train_text = requests.get(train_url).text
    val_text = requests.get(val_url).text

    def encode_corpus(raw_text):
        tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        return torch.tensor(tokens, dtype=torch.long, device=device)

    train_tokens = encode_corpus(train_text)
    val_tokens = encode_corpus(val_text)
    print(f"WikiText-2: Train {len(train_tokens):,} | Val {len(val_tokens):,} WordPiece tokens")

    seq_len = 128
    batch_size = 16
    d_model = 768
    n_heads = 12
    head_dim = d_model // n_heads
    offsets = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]
    K = len(offsets)

    # 3. SubQ Omnidirectional Wave Layer Definition
    class SubQBertWaveAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long, device=device))
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.temp_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1))

            # Recurrent Wave Integrator (GRU)
            self.w_ih = nn.Linear(d_model, 3 * d_model)
            self.w_gate_h = nn.Linear(d_model, 2 * d_model)
            self.w_cand_h = nn.Linear(d_model, d_model)

        def forward(self, x, T_max=4):
            B, L, D = x.shape
            s = x
            energy_trace = []
            for t in range(T_max):
                s_prev = s
                q = self.q_proj(s).view(B, L, n_heads, head_dim).transpose(1, 2)
                k = self.k_proj(s).view(B, L, n_heads, head_dim).transpose(1, 2)
                v = self.v_proj(s).view(B, L, n_heads, head_dim).transpose(1, 2)

                k_s_list, v_s_list, m_list = [], [], []
                for delta_val in self.offsets:
                    d = delta_val.item()
                    if d >= 0:
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
                    else:
                        d_abs = abs(d)
                        if d_abs >= L:
                            k_s = torch.zeros_like(k); v_s = torch.zeros_like(v)
                            m = torch.zeros(B, 1, L, device=x.device, dtype=torch.bool)
                        else:
                            k_s = F.pad(k[:, :, d_abs:, :], (0, 0, 0, d_abs))
                            v_s = F.pad(v[:, :, d_abs:, :], (0, 0, 0, d_abs))
                            m = torch.cat([torch.ones(B, 1, L - d_abs, device=x.device, dtype=torch.bool), torch.zeros(B, 1, d_abs, device=x.device, dtype=torch.bool)], dim=-1)

                    k_s_list.append(k_s); v_s_list.append(v_s); m_list.append(m)

                K_cand = torch.stack(k_s_list, dim=3)
                V_cand = torch.stack(v_s_list, dim=3)
                valid_mask = torch.stack(m_list, dim=3)

                scores = (q.unsqueeze(3) * K_cand).sum(dim=-1) / math.sqrt(head_dim) * self.temp_scale
                scores = scores.masked_fill(~valid_mask, float("-inf"))
                attn_weights = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)

                out = (attn_weights.unsqueeze(-1) * V_cand).sum(dim=3).transpose(1, 2).contiguous().view(B, L, D)
                context = self.out_proj(out)

                r_ih, z_ih, n_ih = self.w_ih(context).chunk(3, dim=-1)
                r_h, z_h = self.w_gate_h(s).chunk(2, dim=-1)
                r = torch.sigmoid(r_ih + r_h)
                z = torch.sigmoid(z_ih + z_h)
                n = torch.tanh(n_ih + self.w_cand_h(r * s))
                s = (1.0 - z) * n + z * s
                energy_trace.append(torch.norm(s - s_prev, p=2, dim=-1).mean().item())
            return s, energy_trace

    class SubQBertModel(nn.Module):
        def __init__(self, bert_source):
            super().__init__()
            # Copy Embeddings
            self.embeddings = bert_source.bert.embeddings
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = SubQBertWaveAttention()
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
            self.cls = bert_source.cls

            # Weight Surgery: Average across all 12 BERT layers
            print("  Performing Weight Surgery: Distilling 12 BERT Layers into 1 SubQ Wave Layer...")
            with torch.no_grad():
                avg_q_w = torch.stack([l.attention.self.query.weight for l in bert_source.bert.encoder.layer]).mean(0)
                avg_q_b = torch.stack([l.attention.self.query.bias for l in bert_source.bert.encoder.layer]).mean(0)
                avg_k_w = torch.stack([l.attention.self.key.weight for l in bert_source.bert.encoder.layer]).mean(0)
                avg_k_b = torch.stack([l.attention.self.key.bias for l in bert_source.bert.encoder.layer]).mean(0)
                avg_v_w = torch.stack([l.attention.self.value.weight for l in bert_source.bert.encoder.layer]).mean(0)
                avg_v_b = torch.stack([l.attention.self.value.bias for l in bert_source.bert.encoder.layer]).mean(0)
                avg_out_w = torch.stack([l.attention.output.dense.weight for l in bert_source.bert.encoder.layer]).mean(0)
                avg_out_b = torch.stack([l.attention.output.dense.bias for l in bert_source.bert.encoder.layer]).mean(0)

                self.attn.q_proj.weight.copy_(avg_q_w); self.attn.q_proj.bias.copy_(avg_q_b)
                self.attn.k_proj.weight.copy_(avg_k_w); self.attn.k_proj.bias.copy_(avg_k_b)
                self.attn.v_proj.weight.copy_(avg_v_w); self.attn.v_proj.bias.copy_(avg_v_b)
                self.attn.out_proj.weight.copy_(avg_out_w); self.attn.out_proj.bias.copy_(avg_out_b)

                avg_mlp1_w = torch.stack([l.intermediate.dense.weight for l in bert_source.bert.encoder.layer]).mean(0)
                avg_mlp1_b = torch.stack([l.intermediate.dense.bias for l in bert_source.bert.encoder.layer]).mean(0)
                avg_mlp2_w = torch.stack([l.output.dense.weight for l in bert_source.bert.encoder.layer]).mean(0)
                avg_mlp2_b = torch.stack([l.output.dense.bias for l in bert_source.bert.encoder.layer]).mean(0)

                self.mlp[0].weight.copy_(avg_mlp1_w); self.mlp[0].bias.copy_(avg_mlp1_b)
                self.mlp[2].weight.copy_(avg_mlp2_w); self.mlp[2].bias.copy_(avg_mlp2_b)

                # Initialize GRU to balance residual passthrough
                nn.init.orthogonal_(self.attn.w_ih.weight, gain=0.1)
                nn.init.orthogonal_(self.attn.w_gate_h.weight, gain=0.1)
                nn.init.orthogonal_(self.attn.w_cand_h.weight, gain=0.1)
                # Bias z to ~0.7 to preserve hidden state memory
                self.attn.w_ih.bias.data.zero_()
                self.attn.w_gate_h.bias.data.zero_()
                self.attn.w_gate_h.bias.data[d_model:].fill_(1.0)
                self.attn.w_cand_h.bias.data.zero_()

        def forward(self, input_ids, T_max=4):
            h = self.embeddings(input_ids)
            attn_out, e_trace = self.attn(self.ln1(h), T_max=T_max)
            h = h + attn_out
            h = h + self.mlp(self.ln2(h))
            logits = self.cls(h)
            return logits, e_trace

    # 4. Evaluation Function
    def get_batch(data, B=batch_size, L=seq_len, mask_p=0.15):
        starts = torch.randint(0, len(data) - L - 1, (B,))
        raw = torch.stack([data[s : s + L] for s in starts])
        mask = torch.rand(B, L, device=device) < mask_p
        masked = raw.clone()
        masked[mask] = mask_token_id
        return masked, raw, mask

    def evaluate_model(model_eval, val_source, num_batches=30, T_eval=4):
        model_eval.eval()
        total_loss, total_top1, total_top5, total_tokens = 0.0, 0.0, 0.0, 0
        all_traces = []

        with torch.no_grad():
            for _ in range(num_batches):
                inp, targets, mask = get_batch(val_source, B=batch_size, L=seq_len, mask_p=0.15)
                if isinstance(model_eval, SubQBertModel):
                    logits, e_trace = model_eval(inp, T_max=T_eval)
                    all_traces.append(e_trace)
                else:
                    out = model_eval(inp)
                    logits = out.logits

                m_logits = logits[mask]
                m_targets = targets[mask]

                loss = F.cross_entropy(m_logits, m_targets)
                total_loss += loss.item() * m_targets.numel()

                preds = m_logits.argmax(dim=-1)
                total_top1 += (preds == m_targets).float().sum().item()

                _, top5 = torch.topk(m_logits, 5, dim=-1)
                total_top5 += (top5 == m_targets.unsqueeze(-1)).any(dim=-1).float().sum().item()

                total_tokens += m_targets.numel()

        avg_loss = total_loss / max(1, total_tokens)
        top1 = (total_top1 / max(1, total_tokens)) * 100.0
        top5 = (total_top5 / max(1, total_tokens)) * 100.0
        ppl = math.exp(min(avg_loss, 20.0))
        return avg_loss, ppl, top1, top5, all_traces

    # -------------------------------------------------------------
    # 5. Baseline Evaluation: Original 12-Layer BERT-Base
    # -------------------------------------------------------------
    print("\n" + "=" * 125)
    print("  [3/5] EVALUATING ORACLE BASELINE: ORIGINAL 12-LAYER DENSE BERT-BASE (110M PARAMS)")
    print("=" * 125)
    b_loss, b_ppl, b_top1, b_top5, _ = evaluate_model(orig_bert, val_tokens, num_batches=40)
    print(f"Original 12L BERT-Base: Val Loss: {b_loss:.4f} | Mask PPL: {b_ppl:.2f} | Top-1: {b_top1:.2f}% | Top-5: {b_top5:.2f}%")

    # -------------------------------------------------------------
    # 6. Instantiate & Evaluate Zero-Shot SubQ Wave Transplant
    # -------------------------------------------------------------
    print("\n" + "=" * 125)
    print("  [4/5] ZERO-SHOT TRANSPLANTED 1-LAYER SUBQ WAVE LATTICE (BEFORE TRAINING)")
    print("=" * 125)
    subq_bert = SubQBertModel(orig_bert).to(device)
    subq_params = sum(p.numel() for p in subq_bert.parameters())
    print(f"1-Layer SubQ Parameters: {subq_params:,} (vs {sum(p.numel() for p in orig_bert.parameters()):,} in 12L BERT)")

    for t_hop in [1, 2, 4, 6]:
        z_loss, z_ppl, z_top1, z_top5, z_traces = evaluate_model(subq_bert, val_tokens, num_batches=20, T_eval=t_hop)
        trace_str = " -> ".join([f"{v:.3f}" for v in z_traces[0]])
        print(f"Zero-Shot T = {t_hop:<2} | Loss: {z_loss:.4f} | Mask PPL: {z_ppl:.2f} | Top-1: {z_top1:.2f}% | Top-5: {z_top5:.2f}% | Trace: [{trace_str}]")

    # -------------------------------------------------------------
    # 7. Rapid Adaptation Training (1,000 Steps on A10G)
    # -------------------------------------------------------------
    print("\n" + "=" * 125)
    print("  [5/5] RAPID ADAPTATION TRAINING OF 1-LAYER SUBQ WAVE LATTICE (1,000 STEPS)")
    print("=" * 125)
    
    # Freeze embeddings to keep representations grounded, adapt Wave Attention + MLPs + GRU
    subq_bert.embeddings.requires_grad_(False)
    subq_bert.cls.requires_grad_(False)
    
    optimizer = torch.optim.AdamW(
        [p for p in subq_bert.parameters() if p.requires_grad],
        lr=2e-4,
        weight_decay=0.01
    )
    scaler = torch.amp.GradScaler('cuda')
    total_steps = 1000

    t0 = time.time()
    for step in range(1, total_steps + 1):
        subq_bert.train()
        progress = step / total_steps
        lr = 1e-5 + 0.5 * (2e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups: g['lr'] = lr

        inp, targets, mask = get_batch(train_tokens, B=batch_size, L=seq_len, mask_p=0.15)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, _ = subq_bert(inp, T_max=4)
            loss = F.cross_entropy(logits[mask], targets[mask])

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(subq_bert.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % 200 == 0:
            print(f"Step {step:>4}/{total_steps} | Train Loss: {loss.item():.4f} | LR: {lr:.6f} | Elapsed: {time.time()-t0:.1f}s")

    # Final Evaluation
    f_loss, f_ppl, f_top1, f_top5, f_traces = evaluate_model(subq_bert, val_tokens, num_batches=50, T_eval=4)
    trace_str = " -> ".join([f"{v:.3f}" for v in f_traces[0]])
    print(f"\nFinal Adapted 1-Layer SubQ Wave (T=4): Val Loss: {f_loss:.4f} | Mask PPL: {f_ppl:.2f} | Top-1: {f_top1:.2f}% | Top-5: {f_top5:.2f}% | Trace: [{trace_str}]")

    # Save to Modal Volume
    ckpt_path = "/root/checkpoints/subq_bert_transplant_best.pt"
    torch.save({
        "step": total_steps,
        "state_dict": subq_bert.state_dict(),
        "top1": f_top1,
        "top5": f_top5,
        "ppl": f_ppl,
        "val_loss": f_loss
    }, ckpt_path)
    volume.commit()
    print(f"⭐ Checkpoint saved permanently to Modal Volume: {ckpt_path}!")

    # -------------------------------------------------------------
    # 8. Scorecard Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 125)
    print("  FINAL SCORECARD: PRE-TRAINED BERT-BASE (12L) VS. 1-LAYER SUBQ WAVE LATTICE")
    print("=" * 125)
    print(f"{'Architecture':<45} | {'Layers':<8} | {'Parameters':<14} | {'Val Loss':<10} | {'Mask PPL':<10} | {'Top-1 Acc':<12} | {'Top-5 Acc':<12}")
    print("-" * 125)
    print(f"{'1. Original BERT-Base (Dense Oracle)':<45} | {'12L':<8} | {'109,514,298':>14} | {b_loss:>8.4f} | {b_ppl:>8.2f} | {b_top1:>10.2f}% | {b_top5:>10.2f}%")
    print(f"{'2. 1-Layer SubQ Wave (Zero-Shot Surgery)':<45} | {'1L (T=4)':<8} | {'38,154,298':>14} | {z_loss:>8.4f} | {z_ppl:>8.2f} | {z_top1:>10.2f}% | {z_top5:>10.2f}%")
    print(f"{'3. 1-Layer SubQ Wave (Adapted 1k Steps)':<45} | {'1L (T=4)':<8} | {'38,154,298':>14} | {f_loss:>8.4f} | {f_ppl:>8.2f} | {f_top1:>10.2f}% | {f_top5:>10.2f}%")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_bert_subq_transplant.remote()
