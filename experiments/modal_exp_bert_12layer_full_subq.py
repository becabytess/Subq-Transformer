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

app = modal.App("exp-bert-12layer-full-subq", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=2400, volumes={"/root/checkpoints": volume})
def run_12layer_subq_bert():
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
    print("  STUDY 35: FULL 12-LAYER SUBQ-BERT (110M PARAMETERS) — ZERO-SHOT TRANSPLANT & ADAPTATION")
    print("  Replacing Dense All-to-All Attention in ALL 12 Layers with SubQ Symmetrical Wave Attention (K=15)")
    print("  No Destructive Averaging: Exact 1-to-1 Weight Transplant Across All 12 Layers")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Load Tokenizer and Base BERT-Base Model
    print("\n[1/5] Loading Pre-Trained bert-base-uncased (110M params, 12 layers)...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    orig_bert = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    orig_bert.eval()
    mask_token_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size

    # 2. Download WikiText-2
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

    d_model = 768
    n_heads = 12
    head_dim = d_model // n_heads
    offsets = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]
    K = len(offsets)
    seq_len = 128
    batch_size = 16

    # 3. SubQ Symmetrical Wave Attention Layer
    class SubQWaveSelfAttention(nn.Module):
        def __init__(self, orig_attn_layer):
            super().__init__()
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long, device=device))
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.temp_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1))

            # Exact 1-to-1 weight copy from BERT layer
            with torch.no_grad():
                self.q_proj.weight.copy_(orig_attn_layer.self.query.weight)
                self.q_proj.bias.copy_(orig_attn_layer.self.query.bias)
                self.k_proj.weight.copy_(orig_attn_layer.self.key.weight)
                self.k_proj.bias.copy_(orig_attn_layer.self.key.bias)
                self.v_proj.weight.copy_(orig_attn_layer.self.value.weight)
                self.v_proj.bias.copy_(orig_attn_layer.self.value.bias)
                self.out_proj.weight.copy_(orig_attn_layer.output.dense.weight)
                self.out_proj.bias.copy_(orig_attn_layer.output.dense.bias)

        def forward(self, x):
            B, L, D = x.shape
            q = self.q_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)

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
            return self.out_proj(out)

    class SubQBertLayer(nn.Module):
        def __init__(self, orig_layer):
            super().__init__()
            self.attn = SubQWaveSelfAttention(orig_layer.attention)
            self.ln1 = orig_layer.attention.output.LayerNorm
            self.intermediate = orig_layer.intermediate
            self.output = orig_layer.output

        def forward(self, h):
            attn_out = self.attn(h)
            h = self.ln1(h + attn_out)
            inter_out = self.intermediate(h)
            h = self.output(inter_out, h)
            return h

    class Full12LayerSubQBert(nn.Module):
        def __init__(self, orig_bert_model):
            super().__init__()
            self.embeddings = orig_bert_model.bert.embeddings
            print("  Transplanting all 12 BERT Layers into 12 SubQ Symmetrical Wave Layers (1-to-1 exact weights)...")
            self.layers = nn.ModuleList([
                SubQBertLayer(orig_bert_model.bert.encoder.layer[i]) for i in range(12)
            ])
            self.cls = orig_bert_model.cls

        def forward(self, input_ids):
            h = self.embeddings(input_ids)
            for layer in self.layers:
                h = layer(h)
            logits = self.cls(h)
            return logits

    # 4. Evaluation Function
    def get_batch(data, B=batch_size, L=seq_len, mask_p=0.15):
        starts = torch.randint(0, len(data) - L - 1, (B,))
        raw = torch.stack([data[s : s + L] for s in starts])
        mask = torch.rand(B, L, device=device) < mask_p
        masked = raw.clone()
        masked[mask] = mask_token_id
        return masked, raw, mask

    def evaluate_model(model_eval, val_source, num_batches=30):
        model_eval.eval()
        total_loss, total_top1, total_top5, total_tokens = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for _ in range(num_batches):
                inp, targets, mask = get_batch(val_source, B=batch_size, L=seq_len, mask_p=0.15)
                out = model_eval(inp)
                if hasattr(out, "logits"):
                    logits = out.logits
                else:
                    logits = out

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
        return avg_loss, ppl, top1, top5

    # 5. Evaluate Teacher Oracle
    print("\n" + "=" * 125)
    print("  [3/5] EVALUATING 12-LAYER DENSE BERT-BASE ORACLE")
    print("=" * 125)
    o_loss, o_ppl, o_top1, o_top5 = evaluate_model(orig_bert, val_tokens, num_batches=40)
    print(f"Original 12L Dense BERT: Val Loss: {o_loss:.4f} | Mask PPL: {o_ppl:.2f} | Top-1: {o_top1:.2f}% | Top-5: {o_top5:.2f}%")

    # 6. Zero-Shot Evaluation of Full 12-Layer SubQ Model
    print("\n" + "=" * 125)
    print("  [4/5] ZERO-SHOT 12-LAYER SUBQ-BERT (IMMEDIATELY AFTER TRANSPLANT, ZERO GRADIENTS)")
    print("=" * 125)
    full_subq = Full12LayerSubQBert(orig_bert).to(device)
    total_params = sum(p.numel() for p in full_subq.parameters())
    print(f"Full 12-Layer SubQ-BERT Parameters: {total_params:,}")

    z_loss, z_ppl, z_top1, z_top5 = evaluate_model(full_subq, val_tokens, num_batches=40)
    print(f"Zero-Shot 12L SubQ: Val Loss: {z_loss:.4f} | Mask PPL: {z_ppl:.2f} | Top-1: {z_top1:.2f}% | Top-5: {z_top5:.2f}%")

    # 7. Rapid Adaptation Training (1,000 Steps on A10G)
    print("\n" + "=" * 125)
    print("  [5/5] RAPID ADAPTATION TRAINING OF 12-LAYER SUBQ-BERT (1,000 STEPS)")
    print("=" * 125)
    
    full_subq.embeddings.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in full_subq.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01
    )
    scaler = torch.amp.GradScaler('cuda')
    total_steps = 1000

    t0 = time.time()
    for step in range(1, total_steps + 1):
        full_subq.train()
        progress = step / total_steps
        lr = 1e-5 + 0.5 * (1e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups: g['lr'] = lr

        inp, targets, mask = get_batch(train_tokens, B=batch_size, L=seq_len, mask_p=0.15)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = full_subq(inp)
            loss = F.cross_entropy(logits[mask], targets[mask])

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(full_subq.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % 200 == 0:
            print(f"Step {step:>4}/{total_steps} | Train Loss: {loss.item():.4f} | LR: {lr:.6f} | Elapsed: {time.time()-t0:.1f}s")

    # Final Evaluation
    f_loss, f_ppl, f_top1, f_top5 = evaluate_model(full_subq, val_tokens, num_batches=50)
    print(f"\nFinal Adapted 12-Layer SubQ-BERT: Val Loss: {f_loss:.4f} | Mask PPL: {f_ppl:.2f} | Top-1: {f_top1:.2f}% | Top-5: {f_top5:.2f}%")

    # Save to Modal Volume
    full_ckpt_path = "/root/checkpoints/subq_bert_12layer_full_best.pt"
    torch.save({
        "step": total_steps,
        "state_dict": full_subq.state_dict(),
        "top1": f_top1,
        "top5": f_top5,
        "ppl": f_ppl,
        "val_loss": f_loss
    }, full_ckpt_path)
    volume.commit()
    print(f"⭐ Full 12-Layer Checkpoint permanently saved to Modal Volume: {full_ckpt_path}!")

    # Qualitative Test on Custom Prompts
    print("\n" + "=" * 125)
    print("  QUALITATIVE EVALUATION ON CUSTOM TEXTS (12-LAYER SUBQ-BERT)")
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

    full_subq.eval()
    for prompt in custom_prompts:
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
        inp_tensor = torch.tensor([encoded], dtype=torch.long, device=device)
        mask_positions = (inp_tensor == mask_token_id).nonzero(as_tuple=True)[1]

        with torch.no_grad():
            logits = full_subq(inp_tensor)

        print(f"\nPrompt:  \"{prompt}\"")
        for pos in mask_positions:
            probs = F.softmax(logits[0, pos], dim=-1)
            top5_probs, top5_ids = torch.topk(probs, 5)
            candidates = [f"'{tokenizer.decode([t.item()]).strip()}' ({p.item()*100:.1f}%)" for p, t in zip(top5_probs, top5_ids)]
            print(f"Top-5 Predictions: {', '.join(candidates)}")

    # Scorecard
    print("\n" + "=" * 125)
    print("  FINAL SCORECARD: 12-LAYER DENSE BERT VS. 12-LAYER FULL SUBQ-BERT")
    print("=" * 125)
    print(f"{'Architecture':<45} | {'Layers':<8} | {'Complexity':<14} | {'Val Loss':<10} | {'Mask PPL':<10} | {'Top-1 Acc':<12} | {'Top-5 Acc':<12}")
    print("-" * 125)
    print(f"{'1. Original BERT-Base (Dense Oracle)':<45} | {'12L':<8} | {'O(L^2) Dense':<14} | {o_loss:>8.4f} | {o_ppl:>8.2f} | {o_top1:>10.2f}% | {o_top5:>10.2f}%")
    print(f"{'2. Full 12-Layer SubQ-BERT (Zero-Shot)':<45} | {'12L':<8} | {'O(L*K) Wave':<14} | {z_loss:>8.4f} | {z_ppl:>8.2f} | {z_top1:>10.2f}% | {z_top5:>10.2f}%")
    print(f"{'3. Full 12-Layer SubQ-BERT (Adapted 1k)':<45} | {'12L':<8} | {'O(L*K) Wave':<14} | {f_loss:>8.4f} | {f_ppl:>8.2f} | {f_top1:>10.2f}% | {f_top5:>10.2f}%")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_12layer_subq_bert.remote()
