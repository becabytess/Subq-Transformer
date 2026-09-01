import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "numpy"
    )
)

app = modal.App("exp-long-seq-accuracy-eval", image=image)
vol = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=1800, volumes={"/root/checkpoints": vol})
def evaluate_long_context_accuracy_retention():
    import math
    import os
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 135)
    print("  STUDY 42: ZERO-SHOT LONG-CONTEXT ACCURACY & PERPLEXITY RETENTION BENCHMARK")
    print("  Testing Trained 12L Dense GPT-2 vs. 12L SubQ-GPT2 (Fixed K=8) Across Sequence Lengths L = 128 -> 4,096 Tokens")
    print("=" * 135)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    import urllib.request

    # 1. Load Tokenizer & Validation Data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading WikiText-2 validation dataset...")
    val_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
    val_text = urllib.request.urlopen(val_url).read().decode('utf-8')
    val_token_ids = tokenizer.encode(val_text)
    print(f"Total Validation Tokens: {len(val_token_ids):,} tokens")

    # 2. Exact Model Definitions from Study 38
    d_model, n_heads = 768, 12
    head_dim = d_model // n_heads
    causal_offsets = [0, 1, 2, 4, 8, 16, 32, 64]
    K = len(causal_offsets)

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
            h = self.wte(input_ids) + self.wpe(pos % 1024)
            for block in self.blocks:
                h = block(h)
            h = self.ln_f(h)
            return self.lm_head(h)

    class ExtDenseGPT2(nn.Module):
        def __init__(self, base_hf):
            super().__init__()
            self.model = base_hf

        def forward(self, input_ids):
            B, L = input_ids.shape
            pos = torch.arange(0, L, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            pos_emb = self.model.transformer.wpe(pos % 1024)
            h = self.model.transformer.wte(input_ids) + pos_emb
            for block in self.model.transformer.h:
                attn_out = block.attn(block.ln_1(h))[0]
                h = h + attn_out
                h = h + block.mlp(block.ln_2(h))
            h = self.model.transformer.ln_f(h)
            return self.model.lm_head(h)

    # 3. Load Trained Checkpoints
    dense_base = AutoModelForCausalLM.from_pretrained("gpt2")
    dense_ckpt_path = "/root/checkpoints/dense_gpt2_controlled_1k.pt"
    if os.path.exists(dense_ckpt_path):
        print(f"Loading trained Dense GPT-2 checkpoint from {dense_ckpt_path}...")
        dense_base.load_state_dict(torch.load(dense_ckpt_path, map_location=device))
    dense_model = ExtDenseGPT2(dense_base).to(device)

    subq_raw = AutoModelForCausalLM.from_pretrained("gpt2")
    subq_model = Full12LayerSubQGPT2(subq_raw).to(device)
    subq_ckpt_path = "/root/checkpoints/subq_gpt2_controlled_1k.pt"
    if os.path.exists(subq_ckpt_path):
        print(f"Loading trained SubQ-GPT2 checkpoint from {subq_ckpt_path}...")
        subq_model.load_state_dict(torch.load(subq_ckpt_path, map_location=device))
    else:
        alt_ckpt = "/root/checkpoints/subq_gpt2_12layer_full_best.pt"
        if os.path.exists(alt_ckpt):
            print(f"Loading SubQ-GPT2 from {alt_ckpt}...")
            subq_model.load_state_dict(torch.load(alt_ckpt, map_location=device))

    dense_model.eval()
    subq_model.eval()

    # 4. Systematic Multi-Length Evaluation
    test_sequence_lengths = [128, 256, 512, 1024, 2048, 4096]
    scorecard = []

    def evaluate_model_on_chunks(model, token_ids, chunk_len, max_eval_tokens=65536):
        total_loss = 0.0
        total_tokens = 0
        correct_top1 = 0
        correct_top5 = 0

        # Create contiguous chunks of length chunk_len
        num_chunks = min(len(token_ids) // chunk_len, max_eval_tokens // chunk_len)
        if num_chunks == 0:
            num_chunks = 1

        with torch.no_grad():
            for i in range(num_chunks):
                chunk = token_ids[i * chunk_len : (i + 1) * chunk_len]
                input_tensor = torch.tensor([chunk], dtype=torch.long, device=device)
                inputs = input_tensor[:, :-1]
                targets = input_tensor[:, 1:]

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(inputs)
                    loss = F.cross_entropy(logits.view(-1, 50257), targets.view(-1))

                B_curr, L_curr = targets.shape
                total_loss += loss.item() * (B_curr * L_curr)
                total_tokens += (B_curr * L_curr)

                # Top-1 & Top-5
                top5_preds = logits.topk(5, dim=-1).indices
                top1_preds = top5_preds[:, :, 0]
                correct_top1 += (top1_preds == targets).sum().item()
                correct_top5 += (top5_preds == targets.unsqueeze(-1)).sum().item()

        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(min(20.0, avg_loss))
        top1_acc = (correct_top1 / max(1, total_tokens)) * 100.0
        top5_acc = (correct_top5 / max(1, total_tokens)) * 100.0
        return avg_loss, ppl, top1_acc, top5_acc

    print("\n" + "=" * 135)
    print("  RUNNING SYSTEMATIC ZERO-SHOT SEQUENCE LENGTH EVALUATION (WikiText-2 Validation Split)")
    print("=" * 135)

    for L in test_sequence_lengths:
        print(f"\n>>> EVALUATING CONTEXT LENGTH L = {L:,} TOKENS (SubQ evaluates only K={K} offsets = {K/L*100:.2f}% of sequence):")

        # Dense Evaluation
        t0 = time.time()
        d_loss, d_ppl, d_top1, d_top5 = evaluate_model_on_chunks(dense_model, val_token_ids, L)
        d_time = time.time() - t0
        print(f"  [12L Dense GPT-2] (All {L} tokens) -> Val Loss: {d_loss:.4f} | PPL: {d_ppl:>6.2f} | Top-1: {d_top1:.2f}% | Top-5: {d_top5:.2f}% ({d_time:.1f}s)")

        # SubQ Evaluation
        t0 = time.time()
        s_loss, s_ppl, s_top1, s_top5 = evaluate_model_on_chunks(subq_model, val_token_ids, L)
        s_time = time.time() - t0
        print(f"  [12L SubQ-GPT2]   (Only {K} offsets) -> Val Loss: {s_loss:.4f} | PPL: {s_ppl:>6.2f} | Top-1: {s_top1:.2f}% | Top-5: {s_top5:.2f}% ({s_time:.1f}s)")

        # Relative power retention
        accuracy_retention = (s_top1 / max(1e-5, d_top1)) * 100.0
        print(f"  ⚡ SubQ Accuracy Retention vs Dense GPT-2: {accuracy_retention:.1f}% (Evaluating {K/L*100:.2f}% of context)")

        scorecard.append({
            "L": L,
            "pct_tokens": f"{K/L*100:.2f}%",
            "d_loss": f"{d_loss:.4f}",
            "d_ppl": f"{d_ppl:.2f}",
            "d_top1": f"{d_top1:.2f}%",
            "d_top5": f"{d_top5:.2f}%",
            "s_loss": f"{s_loss:.4f}",
            "s_ppl": f"{s_ppl:.2f}",
            "s_top1": f"{s_top1:.2f}%",
            "s_top5": f"{s_top5:.2f}%",
            "retention": f"{accuracy_retention:.1f}%"
        })

    # Master Scorecard
    print("\n" + "=" * 135)
    print("  MASTER LONG-CONTEXT ACCURACY RETENTION SCORECARD: DENSE GPT-2 VS. SUBQ-GPT2")
    print("=" * 135)
    print(f"{'Sequence Len (L)':<18} | {'SubQ Token %':<14} | {'Dense PPL':<12} | {'SubQ PPL':<12} | {'Dense Top-1':<14} | {'SubQ Top-1':<14} | {'Accuracy Retention':<18}")
    print("-" * 135)
    for row in scorecard:
        print(f"{row['L']:<18,} | {row['pct_tokens']:<14} | {row['d_ppl']:<12} | {row['s_ppl']:<12} | {row['d_top1']:<14} | {row['s_top1']:<14} | {row['retention']:<18}")
    print("=" * 135)

@app.local_entrypoint()
def main():
    evaluate_long_context_accuracy_retention.remote()
