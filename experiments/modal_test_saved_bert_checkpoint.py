import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "requests",
        "numpy"
    )
)

app = modal.App("test-saved-bert-checkpoint", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=600, volumes={"/root/checkpoints": volume})
def test_saved_bert_transplant():
    import math
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 110)
    print("  TESTING SAVED 1-LAYER SUBQ-BERT CHECKPOINT ON CUSTOM OUT-OF-DOMAIN PROMPTS")
    print("=" * 110)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    d_model, n_heads, seq_len = 768, 12, 128
    head_dim = d_model // n_heads
    offsets = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]

    # Model Definition
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
            from transformers import BertForMaskedLM
            bert_base = BertForMaskedLM.from_pretrained("bert-base-uncased")
            self.embeddings = bert_base.bert.embeddings
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = SubQBertWaveAttention()
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
            self.cls = bert_base.cls

        def forward(self, input_ids, T_max=4):
            h = self.embeddings(input_ids)
            attn_out, e_trace = self.attn(self.ln1(h), T_max=T_max)
            h = h + attn_out
            h = h + self.mlp(self.ln2(h))
            logits = self.cls(h)
            return logits, e_trace

    # Load Saved Checkpoint from Modal Volume
    print("\nLoading checkpoint: /root/checkpoints/subq_bert_transplant_best.pt...")
    model = SubQBertModel().to(device)
    ckpt_path = "/root/checkpoints/subq_bert_transplant_best.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"⭐ Successfully loaded checkpoint! Saved Val PPL: {ckpt.get('ppl', 'N/A'):.2f}, Top-1 Acc: {ckpt.get('top1', 'N/A'):.2f}%")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}!")

    model.eval()

    # Diverse Custom Test Prompts across Science, Geography, Daily Life, and Syntax
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

    print("\n" + "=" * 110)
    print("  QUALITATIVE EVALUATION ON CUSTOM TEXTS")
    print("=" * 110)

    for prompt in custom_prompts:
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
        inp_tensor = torch.tensor([encoded], dtype=torch.long, device=device)
        
        # Find [MASK] token position
        mask_positions = (inp_tensor == mask_token_id).nonzero(as_tuple=True)[1]

        with torch.no_grad():
            logits, trace = model(inp_tensor, T_max=4)

        print(f"\nPrompt:  \"{prompt}\"")
        trace_str = " -> ".join([f"{v:.3f}" for v in trace])
        print(f"Relaxation Trace: [{trace_str}]")

        for pos in mask_positions:
            mask_logits = logits[0, pos]
            probs = F.softmax(mask_logits, dim=-1)
            top5_probs, top5_ids = torch.topk(probs, 5)

            candidates = []
            for prob, token_id in zip(top5_probs, top5_ids):
                token_str = tokenizer.decode([token_id.item()]).strip()
                candidates.append(f"'{token_str}' ({prob.item()*100:.1f}%)")

            print(f"Top-5 Predictions for [MASK]: {', '.join(candidates)}")

    print("\n" + "=" * 110)
    print("Evaluation Complete!")
    print("=" * 110)

@app.local_entrypoint()
def main():
    test_saved_bert_transplant.remote()
