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

app = modal.App("exp-bert-subq-kl-distillation", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=2400, volumes={"/root/checkpoints": volume})
def run_bert_subq_kl_distillation():
    import math
    import time
    import os
    import requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertForMaskedLM, BertTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 34: 1-LAYER SUBQ-BERT FULL KL-DIVERGENCE LOGIT DISTILLATION (TEACHER: 12-LAYER DENSE BERT)")
    print("  Continuing from saved checkpoint '/root/checkpoints/subq_bert_transplant_best.pt'")
    print("  Matching the full 30,522-way soft probability distribution of 12-layer BERT-Base with Temperature Scaling")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Load Tokenizer and 12-Layer Teacher BERT
    print("\n[1/4] Loading 12-Layer BERT Teacher Oracle...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    teacher_bert = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    teacher_bert.eval()
    for p in teacher_bert.parameters():
        p.requires_grad = False
    
    mask_token_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size

    # 2. Download Dataset
    print("\n[2/4] Loading WikiText-2 Dataset...")
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

    d_model, n_heads, seq_len = 768, 12, 128
    head_dim = d_model // n_heads
    offsets = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]
    batch_size = 16

    # 3. Model Architecture
    class SubQBertWaveAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long, device=device))
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.temp_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1))

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
        def __init__(self):
            super().__init__()
            self.embeddings = teacher_bert.bert.embeddings
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = SubQBertWaveAttention()
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
            self.cls = teacher_bert.cls

        def forward(self, input_ids, T_max=4):
            h = self.embeddings(input_ids)
            attn_out, e_trace = self.attn(self.ln1(h), T_max=T_max)
            h = h + attn_out
            h = h + self.mlp(self.ln2(h))
            logits = self.cls(h)
            return logits, e_trace

    # Load from saved checkpoint
    print("\n[3/4] Initializing SubQ-BERT from Saved Checkpoint...")
    student_bert = SubQBertModel().to(device)
    ckpt_path = "/root/checkpoints/subq_bert_transplant_best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        student_bert.load_state_dict(ckpt["state_dict"])
        print(f"Loaded starting weights from {ckpt_path} (Initial Top-1 Acc: {ckpt.get('top1', 'N/A'):.2f}%)!")
    else:
        print("Checkpoint not found, initializing fresh surgery.")

    # Freeze embeddings to retain pre-trained grounding
    student_bert.embeddings.requires_grad_(False)
    student_bert.cls.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in student_bert.parameters() if p.requires_grad],
        lr=2e-4,
        weight_decay=0.01
    )
    scaler = torch.amp.GradScaler('cuda')

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
                    logits = model_eval(inp).logits

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

    # 4. KL-Divergence Distillation Training
    total_steps = 1500
    temperature = 2.0
    alpha_kl = 0.8 # 80% soft KL logit distillation, 20% hard cross-entropy
    print("\n" + "=" * 125)
    print(f"  [4/4] DISTILLING 12L DENSE BERT -> 1L SUBQ WAVE (1,500 Steps | Temp={temperature} | Alpha_KL={alpha_kl})")
    print("=" * 125)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        student_bert.train()
        progress = step / total_steps
        lr = 1e-5 + 0.5 * (2e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups: g['lr'] = lr

        inp, targets, mask = get_batch(train_tokens, B=batch_size, L=seq_len, mask_p=0.15)

        # Teacher Forward Pass (No Grad)
        with torch.no_grad():
            teacher_logits = teacher_bert(inp).logits # [B, L, Vocab]

        # Student Forward Pass
        with torch.amp.autocast('cuda', dtype=torch.float16):
            student_logits, _ = student_bert(inp, T_max=4) # [B, L, Vocab]

            # Distill over ALL tokens (both masked and unmasked context!)
            t_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
            t_soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
            
            # KL Divergence Loss
            kl_loss = F.kl_div(t_log_soft, t_soft_targets, reduction='batchmean') * (temperature ** 2)
            
            # Hard Ground-Truth Cross-Entropy on Masked Tokens
            ce_loss = F.cross_entropy(student_logits[mask], targets[mask])

            loss = alpha_kl * kl_loss + (1.0 - alpha_kl) * ce_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_bert.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % 250 == 0:
            print(f"Step {step:>4}/{total_steps} | Total Loss: {loss.item():.4f} | KL: {kl_loss.item():.4f} | CE: {ce_loss.item():.4f} | Elapsed: {time.time()-t0:.1f}s")

    # Final Quantitative Evaluation on WikiText-2
    f_loss, f_ppl, f_top1, f_top5, f_traces = evaluate_model(student_bert, val_tokens, num_batches=50, T_eval=4)
    trace_str = " -> ".join([f"{v:.3f}" for v in f_traces[0]])
    print(f"\nFinal KL-Distilled 1-Layer SubQ Wave (T=4): Val Loss: {f_loss:.4f} | Mask PPL: {f_ppl:.2f} | Top-1: {f_top1:.2f}% | Top-5: {f_top5:.2f}% | Trace: [{trace_str}]")

    # Save to Volume
    distill_ckpt_path = "/root/checkpoints/subq_bert_kldistill_best.pt"
    torch.save({
        "step": total_steps,
        "state_dict": student_bert.state_dict(),
        "top1": f_top1,
        "top5": f_top5,
        "ppl": f_ppl,
        "val_loss": f_loss
    }, distill_ckpt_path)
    volume.commit()
    print(f"⭐ Distilled Checkpoint permanently saved to Modal Volume: {distill_ckpt_path}!")

    # Qualitative Test on Custom Prompts
    print("\n" + "=" * 125)
    print("  QUALITATIVE EVALUATION ON CUSTOM TEXTS AFTER KL-DISTILLATION")
    print("=" * 125)

    custom_prompts = [
        "Paris is the [MASK] of France.",
        "The Earth revolves around the [MASK].",
        "Python is a popular programming [MASK].",
        "The cat sat on the comfortable [MASK].",
        "Albert Einstein was a famous [MASK] who discovered relativity.",
        "Water boils at one hundred degrees [MASK].",
        "She opened the book and started to [MASK].",
        "The doctor prescribed some [MASK] for the infection."
    ]

    student_bert.eval()
    for prompt in custom_prompts:
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
        inp_tensor = torch.tensor([encoded], dtype=torch.long, device=device)
        mask_positions = (inp_tensor == mask_token_id).nonzero(as_tuple=True)[1]

        with torch.no_grad():
            logits, trace = student_bert(inp_tensor, T_max=4)

        print(f"\nPrompt:  \"{prompt}\"")
        for pos in mask_positions:
            probs = F.softmax(logits[0, pos], dim=-1)
            top5_probs, top5_ids = torch.topk(probs, 5)
            candidates = [f"'{tokenizer.decode([t.item()]).strip()}' ({p.item()*100:.1f}%)" for p, t in zip(top5_probs, top5_ids)]
            print(f"Top-5 Predictions: {', '.join(candidates)}")

    print("\n" + "=" * 125)
    print("Study 34 Completed Successfully!")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_bert_subq_kl_distillation.remote()
