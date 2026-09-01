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

app = modal.App("exp-harmonic-gpt2-refined-transplant", image=image)

@app.function(gpu="A10G", timeout=3600)
def run_harmonic_gpt2_refined():
    import math
    import time
    import requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 130)
    print("  STUDY 65B: REFINED 12-LAYER HARMONIC SUBQ-GPT2 (124M PARAMETERS) TRANSPLANT")
    print("  Harmonic Carrier Waves with Log-Scale Initial Resonance & Continuous Adaptation on WikiText-2")
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
    # Model: Refined Harmonic SubQ Attention for Pre-Trained GPT-2
    # -------------------------------------------------------------------------
    class RefinedHarmonicSubQAttention(nn.Module):
        def __init__(self, orig_attn_layer, layer_idx=0):
            super().__init__()
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)

            # Initialize wave latents with structured frequency spectrum
            init_latents = torch.zeros(n_heads, num_waves, 4)
            # Amplitude: positive
            init_latents[..., 0] = 0.5
            # Decay: small positive
            init_latents[..., 3] = 0.1
            self.init_wave_latent = nn.Parameter(init_latents.view(n_heads, num_waves * 4))

            # Base logarithmic frequencies: covering periods from 2 to 64 tokens
            log_freqs = torch.linspace(math.log10(math.pi / 1.0), math.log10(math.pi / 32.0), num_waves)
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
            wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1) # [B, n_heads, max_d - 1]

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

    # Build model
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    for idx, block in enumerate(base_model.transformer.h):
        block.attn = RefinedHarmonicSubQAttention(block.attn, layer_idx=idx)
    model = base_model.to(device)

    # Zero-shot evaluation
    val_loss_zero, ppl_zero, top1_zero, top5_zero = evaluate_model(model, split="val", n_batches=15)
    print(f"\nRefined 12L Harmonic SubQ-GPT2 -> Zero-Shot Val Loss: {val_loss_zero:.4f} | Zero-Shot PPL: {ppl_zero:.2f} | Top-1: {top1_zero:.2f}%")

    # Training with learning rate warmup & cosine schedule
    print(f"\n[Training & Adapting Refined 12L Harmonic SubQ-GPT2] (1,000 Steps on WikiText-2)...")
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

        if (step + 1) % 200 == 0 or step == total_steps - 1:
            print(f"  Step {step+1:4d}/{total_steps} | Loss: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")

    elapsed = time.time() - t0

    # In-Domain WikiText-2 Evaluation
    val_loss, ppl, top1_acc, top5_acc = evaluate_model(model, split="val", n_batches=30)
    # Out-of-Domain Penn Treebank Evaluation
    ptb_loss, ptb_ppl, ptb_top1, ptb_top5 = evaluate_model(model, split="ptb", n_batches=30)

    print(f"\n---> Refined 12L Harmonic SubQ-GPT2 Final Results:")
    print(f"     WikiText-2 (In-Domain)  : Loss = {val_loss:.4f} | PPL = {ppl:>6.2f} | Top-1 = {top1_acc:>5.2f}% | Top-5 = {top5_acc:>5.2f}%")
    print(f"     Penn Treebank (Out-Domain): Loss = {ptb_loss:.4f} | PPL = {ptb_ppl:>6.2f} | Top-1 = {ptb_top1:>5.2f}% | Top-5 = {ptb_top5:>5.2f}%")

    # Sample generation
    model.eval()
    prompt_text = "The discovery of gravitational waves showed that"
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = input_ids.clone()
        for _ in range(30):
            out = model(gen_ids)
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
    completion = tokenizer.decode(gen_ids[0].tolist())
    print(f"\n     Sample Generation:\n     \"{completion}\"\n")

    return {
        "val_ppl": ppl,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "ptb_ppl": ptb_ppl,
        "ptb_top1": ptb_top1,
        "elapsed": elapsed
    }

@app.local_entrypoint()
def main():
    run_harmonic_gpt2_refined.remote()
