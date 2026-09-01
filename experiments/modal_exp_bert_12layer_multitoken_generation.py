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

app = modal.App("exp-bert-12layer-multitoken-gen", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=2400, volumes={"/root/checkpoints": volume})
def run_12layer_multitoken_generation():
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
    print("  STUDY 36: FULL 12-LAYER SUBQ-BERT MULTI-TOKEN FORWARD SPAN GENERATION & INFILLING")
    print("  Training and Evaluating Parallel & Iterative Future-Span Generation (N = 4, 8, 16, 32, 48 Tokens Forward)")
    print("  Initialized from Best Full 12-Layer SubQ Checkpoint: '/root/checkpoints/subq_bert_12layer_full_best.pt'")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Load Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    mask_token_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size

    # 2. Download WikiText-2
    print("\n[1/5] Loading WikiText-2 Dataset...")
    train_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    val_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"

    train_text = requests.get(train_url).text
    val_text = requests.get(val_url).text

    def encode_corpus(raw_text):
        tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        return torch.tensor(tokens, dtype=torch.long, device=device)

    train_tokens = encode_corpus(train_text)
    val_tokens = encode_corpus(val_text)
    print(f"WikiText-2: Train {len(train_tokens):,} | Val {len(val_tokens):,} tokens")

    d_model, n_heads, seq_len = 768, 12, 128
    head_dim = d_model // n_heads
    offsets = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]
    batch_size = 16

    # 3. Model Architecture
    class SubQWaveSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long, device=device))
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.temp_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1))

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
            self.attn = SubQWaveSelfAttention()
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

    # 4. Load Pretrained Checkpoint
    print("\n[2/5] Initializing 12-Layer SubQ-BERT from Saved Checkpoint...")
    base_bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model = Full12LayerSubQBert(base_bert).to(device)

    ckpt_path = "/root/checkpoints/subq_bert_12layer_full_best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded existing weights from {ckpt_path} (Top-1: {ckpt.get('top1', 'N/A'):.2f}%)!")
    else:
        print("Warning: Checkpoint not found, initializing fresh structure.")

    # 5. Span Masking Batch Generator (Contiguous Future Spans)
    def get_span_batch(data, B=batch_size, L=seq_len, min_span=4, max_span=32):
        starts = torch.randint(0, len(data) - L - 1, (B,))
        raw = torch.stack([data[s : s + L] for s in starts])
        masked = raw.clone()
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            span_len = torch.randint(min_span, max_span + 1, (1,)).item()
            # Choose a future span start position
            pos = torch.randint(10, L - span_len - 1, (1,)).item()
            mask[b, pos : pos + span_len] = True
            masked[b, pos : pos + span_len] = mask_token_id

        return masked, raw, mask

    # 6. Fine-Tuning for Long Contiguous Span Infilling (1,000 Steps)
    print("\n" + "=" * 125)
    print("  [3/5] TRAINING 12-LAYER SUBQ-BERT ON CONTIGUOUS FUTURE SPAN INFILLING (Spans 4 to 32 tokens)")
    print("=" * 125)

    model.embeddings.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
        weight_decay=0.01
    )
    scaler = torch.amp.GradScaler('cuda')
    total_steps = 1000

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        progress = step / total_steps
        lr = 1e-5 + 0.5 * (5e-5 - 1e-5) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups: g['lr'] = lr

        inp, targets, mask = get_span_batch(train_tokens, B=batch_size, L=seq_len, min_span=4, max_span=32)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(inp)
            loss = F.cross_entropy(logits[mask], targets[mask])

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % 200 == 0:
            print(f"Step {step:>4}/{total_steps} | Span Loss: {loss.item():.4f} | LR: {lr:.6f} | Elapsed: {time.time()-t0:.1f}s")

    # 7. Evaluate Performance Across Different Fixed Span Lengths (N = 4, 8, 16, 24, 32, 48)
    print("\n" + "=" * 125)
    print("  [4/5] SYSTEMATIC BENCHMARK ACROSS FUTURE SPAN LENGTHS (N = 4, 8, 16, 24, 32, 48 TOKENS)")
    print("=" * 125)

    def evaluate_fixed_span(N_span, num_batches=30):
        model.eval()
        total_loss, total_top1, total_top5, total_tokens = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for _ in range(num_batches):
                starts = torch.randint(0, len(val_tokens) - seq_len - 1, (batch_size,))
                raw = torch.stack([val_tokens[s : s + seq_len] for s in starts])
                masked = raw.clone()
                mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
                
                pos = 20 # fixed prefix of 20 tokens
                mask[:, pos : pos + N_span] = True
                masked[:, pos : pos + N_span] = mask_token_id

                logits = model(masked)
                m_logits = logits[mask]
                m_targets = raw[mask]

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
        return ppl, top1, top5

    span_lengths = [4, 8, 12, 16, 24, 32, 48]
    span_results = {}
    print(f"{'Future Span Length (N)':<25} | {'Masked PPL':<12} | {'Top-1 Accuracy':<15} | {'Top-5 Accuracy':<15}")
    print("-" * 75)
    for N in span_lengths:
        ppl, top1, top5 = evaluate_fixed_span(N, num_batches=30)
        span_results[N] = (ppl, top1, top5)
        print(f"N = {N:>2} Tokens Forward       | {ppl:>10.2f} | {top1:>13.2f}% | {top5:>13.2f}%")

    # 8. Multi-Pass Iterative Future Generation Demo
    print("\n" + "=" * 125)
    print("  [5/5] QUALITATIVE DEMO: GENERATING FORWARD SPANS VIA PARALLEL & ITERATIVE RELAXATION")
    print("=" * 125)

    def generate_span(prompt_prefix, num_future_tokens=16, num_refine_steps=3):
        prefix_ids = tokenizer.encode(prompt_prefix, add_special_tokens=True)[:-1] # drop [SEP]
        input_ids = prefix_ids + [mask_token_id] * num_future_tokens + [tokenizer.sep_token_id]
        cur_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        span_start = len(prefix_ids)
        span_end = span_start + num_future_tokens

        # Pass 1: 1-Shot Direct Parallel Generation
        with torch.no_grad():
            logits = model(cur_tensor)
            pred_tokens_p1 = logits[0, span_start:span_end].argmax(dim=-1).tolist()
            text_p1 = tokenizer.decode(pred_tokens_p1)

        # Iterative Refinement Passes (Confidence-guided Relaxation)
        refined_tensor = cur_tensor.clone()
        for p in range(num_refine_steps):
            with torch.no_grad():
                logits = model(refined_tensor)
                preds = logits[0, span_start:span_end].argmax(dim=-1)
                refined_tensor[0, span_start:span_end] = preds

        final_tokens = refined_tensor[0, span_start:span_end].tolist()
        text_refined = tokenizer.decode(final_tokens)
        return text_p1, text_refined

    test_prompts = [
        ("The discovery of gravity revolutionized our understanding of", 12),
        ("In computer science, neural networks are designed to", 14),
        ("The president of the United States announced that", 12),
        ("Albert Einstein published his theory of relativity which", 14),
        ("She walked into the quiet library and opened a", 10),
        ("The doctor prescribed an effective medicine to treat the severe", 10)
    ]

    for p_text, n_tokens in test_prompts:
        p1_out, ref_out = generate_span(p_text, num_future_tokens=n_tokens, num_refine_steps=3)
        print(f"\nPrompt: \"{p_text} ... [{n_tokens} TOKENS FUTURE]\"")
        print(f"  [Pass 1 Parallel Gen]:   {p_text} -> {p1_out}")
        print(f"  [Iterative Refined Gen]: {p_text} -> {ref_out}")

    # Permanent Checkpoint
    span_ckpt_path = "/root/checkpoints/subq_bert_12layer_multitoken_best.pt"
    torch.save({
        "step": total_steps,
        "state_dict": model.state_dict(),
        "span_results": span_results
    }, span_ckpt_path)
    volume.commit()
    print(f"\n⭐ Multi-Token Infilling Checkpoint permanently saved to Modal Volume: {span_ckpt_path}!")

    print("\n" + "=" * 125)
    print("Study 36 Completed Successfully!")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_12layer_multitoken_generation.remote()
