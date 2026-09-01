import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2.0", "numpy", "requests")
)

app = modal.App("exp-subq-bidirectional-multitoken", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=1200, volumes={"/root/checkpoints": volume})
def run_multitoken_wave_infilling():
    import math
    import time
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import requests

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 31: BIDIRECTIONAL SUBQ MULTI-TOKEN SPAN INFILLING & WAVE DIFFUSION")
    print("  Testing the trained 1-Layer SubQ Wave Lattice (saved in Modal Volume) on contiguous multi-token blocks")
    print("  (N = 4, 8, 12, 16 tokens) bounded by bidirectional past & future context!")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset & Tokenizer Setup
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    chars = sorted(list(set(text)))
    vocab_size = len(chars) + 2
    mask_token_id = len(chars)
    pad_token_id = len(chars) + 1

    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    idx2char[mask_token_id] = "[M]"
    idx2char[pad_token_id] = "[P]"

    data_ids = [char2idx[c] for c in text]
    data_tensor = torch.tensor(data_ids, dtype=torch.long, device=device)
    split = int(len(data_tensor) * 0.9)
    val_data = data_tensor[split:]

    d_model, n_heads, seq_len = 256, 8, 256
    head_dim = d_model // n_heads
    offsets = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]

    # 2. SubQ Wave Lattice Architecture
    class SubQWaveAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long, device=device))
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)
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
                qkv = self.c_attn(s)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
                k = k.view(B, L, n_heads, head_dim).transpose(1, 2)
                v = v.view(B, L, n_heads, head_dim).transpose(1, 2)

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
                context = self.c_proj(out)

                r_ih, z_ih, n_ih = self.w_ih(context).chunk(3, dim=-1)
                r_h, z_h = self.w_gate_h(s).chunk(2, dim=-1)
                r = torch.sigmoid(r_ih + r_h)
                z = torch.sigmoid(z_ih + z_h)
                n = torch.tanh(n_ih + self.w_cand_h(r * s))
                s = (1.0 - z) * n + z * s
                energy_trace.append(torch.norm(s - s_prev, p=2, dim=-1).mean().item())
            return s, energy_trace

    class SubQWaveModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(vocab_size, d_model)
            self.wpe = nn.Embedding(seq_len, d_model)
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = SubQWaveAttention()
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size)

        def forward(self, x, T_max=4):
            B, L = x.shape
            pos = torch.arange(0, L, dtype=torch.long, device=x.device).unsqueeze(0)
            h = self.wte(x) + self.wpe(pos)
            attn_out, e_trace = self.attn(self.ln1(h), T_max=T_max)
            h = h + attn_out
            h = h + self.mlp(self.ln2(h))
            logits = self.lm_head(self.ln_f(h))
            return logits, e_trace

    # 3. Load Checkpoint from Modal Volume
    print("\n[1/3] Loading Trained Bidirectional Wave Lattice Checkpoint from Volume...")
    model = SubQWaveModel().to(device)
    ckpt_path = "/root/checkpoints/model4_subq_wave.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Successfully loaded checkpoint from {ckpt_path} (Trained Val PPL: {ckpt.get('ppl', 'N/A'):.2f})!")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}!")

    model.eval()

    # 4. Multi-Token Contiguous Span Infilling Evaluation
    print("\n" + "=" * 125)
    print("  [2/3] EVALUATING MULTI-TOKEN CONTIGUOUS SPAN RECONSTRUCTION (N = 4, 8, 12, 16 TOKENS)")
    print("=" * 125)

    span_sizes = [4, 8, 12, 16]
    print(f"{'Span Size (N)':<16} | {'Span Top-1 Acc':<18} | {'Span Top-5 Acc':<18} | {'Span Perplexity':<18} | {'Wave Settling Trace (||Δs||)':<35}")
    print("-" * 115)

    for n_span in span_sizes:
        total_top1 = 0.0
        total_top5 = 0.0
        total_loss = 0.0
        total_evaluated = 0
        all_traces = []

        with torch.no_grad():
            for i in range(100):
                idx = i * seq_len
                if idx + seq_len >= len(val_data):
                    break
                
                raw = val_data[idx : idx + seq_len].clone().unsqueeze(0) # [1, L]
                
                # Pick a random contiguous span of length n_span in the middle
                start_span = 64
                end_span = start_span + n_span
                
                masked_input = raw.clone()
                masked_input[0, start_span:end_span] = mask_token_id
                target_span = raw[0, start_span:end_span]

                logits, e_trace = model(masked_input, T_max=4)
                span_logits = logits[0, start_span:end_span, :] # [n_span, vocab_size]

                loss = F.cross_entropy(span_logits, target_span)
                total_loss += loss.item()

                preds = span_logits.argmax(dim=-1)
                total_top1 += (preds == target_span).float().mean().item()

                _, top5 = torch.topk(span_logits, 5, dim=-1)
                total_top5 += (top5 == target_span.unsqueeze(-1)).any(dim=-1).float().mean().item()

                total_evaluated += 1
                all_traces.append(e_trace)

        avg_loss = total_loss / total_evaluated
        top1 = (total_top1 / total_evaluated) * 100.0
        top5 = (total_top5 / total_evaluated) * 100.0
        ppl = math.exp(min(avg_loss, 20.0))

        # Compute average trace
        num_hops = len(all_traces[0])
        mean_trace = [sum(t[h] for t in all_traces) / len(all_traces) for h in range(num_hops)]
        trace_str = " -> ".join([f"{v:.3f}" for v in mean_trace])

        print(f"N = {n_span:<12} | {top1:>14.2f}%    | {top5:>14.2f}%    | {ppl:>14.2f}     | [{trace_str}]")

    # 5. Qualitative Multi-Token Span Infilling Demos
    print("\n" + "=" * 125)
    print("  [3/3] QUALITATIVE MULTI-TOKEN BOUNDARY SPAN INFILLING DEMOS")
    print("=" * 125)

    test_passages = [
        ("To be, or not to be, that is the question:", 20, 10), # mask 10 chars in middle
        ("Friends, Romans, countrymen, lend me your ears;", 17, 10),
        ("Now is the winter of our discontent", 11, 8),
        ("Romeo, Romeo! wherefore art thou Romeo?", 14, 10)
    ]

    for orig, start, length in test_passages:
        clean_ids = [char2idx.get(c, 0) for c in orig]
        # Pad to seq_len
        padded_ids = clean_ids + [pad_token_id] * (seq_len - len(clean_ids))
        inp = torch.tensor(padded_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Corrupt contiguous span
        corrupted = inp.clone()
        corrupted[0, start : start + length] = mask_token_id

        with torch.no_grad():
            logits, _ = model(corrupted, T_max=4)
        
        preds = logits.argmax(dim=-1)[0][:len(orig)].tolist()
        corrupt_str = "".join([idx2char[t] for t in corrupted[0][:len(orig)].tolist()])
        infilled_str = "".join([idx2char[t] for t in preds])

        print(f"\nOriginal:  \"{orig}\"")
        print(f"Corrupted: \"{corrupt_str}\"")
        print(f"Infilled:  \"{infilled_str}\"")

    print("\n" + "=" * 125)
    print("Study 31 Completed Successfully!")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_multitoken_wave_infilling.remote()
