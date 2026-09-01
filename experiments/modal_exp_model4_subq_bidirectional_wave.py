import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2.0", "numpy", "requests")
)

app = modal.App("exp-model4-subq-wave", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=1200, volumes={"/root/checkpoints": volume})
def train_subq_wave():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import requests

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90)
    print("  MODEL 4: BIDIRECTIONAL SUBQ WAVE LATTICE (1 LAYER, OMNIDIRECTIONAL JUMPS, T=4 HOPS)")
    print("=" * 90)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    chars = sorted(list(set(text)))
    vocab_size = len(chars) + 2
    mask_token_id = len(chars)

    char2idx = {ch: i for i, ch in enumerate(chars)}
    data_ids = [char2idx[c] for c in text]
    data_tensor = torch.tensor(data_ids, dtype=torch.long, device=device)

    split = int(len(data_tensor) * 0.9)
    train_data = data_tensor[:split]
    val_data = data_tensor[split:]

    d_model, n_heads, seq_len = 256, 8, 256
    batch_size, total_steps = 16, 2000
    head_dim = d_model // n_heads

    # Symmetrical Omnidirectional Wave Menu (Past & Future)
    offsets = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]
    K = len(offsets)

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
            for t in range(T_max):
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
            return s

    class Model(nn.Module):
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
            h = h + self.attn(self.ln1(h), T_max=T_max)
            h = h + self.mlp(self.ln2(h))
            return self.lm_head(self.ln_f(h))

    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    def get_batch(data):
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        raw = torch.stack([data[s : s + seq_len] for s in starts])
        mask = torch.rand(batch_size, seq_len, device=device) < 0.20
        masked = raw.clone()
        masked[mask] = mask_token_id
        return masked, raw, mask

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        progress = step / total_steps
        lr = 1e-5 + 0.5 * (5e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups: g['lr'] = lr

        inp, target, mask = get_batch(train_data)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(inp, T_max=4)
            loss = F.cross_entropy(logits[mask], target[mask])

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Evaluation
    model.eval()
    total_loss, total_top1, total_top5, total_tokens = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for _ in range(40):
            inp, target, mask = get_batch(val_data)
            logits = model(inp, T_max=4)
            m_logits = logits[mask]
            m_target = target[mask]
            total_loss += F.cross_entropy(m_logits, m_target).item() * m_target.numel()
            total_top1 += (m_logits.argmax(-1) == m_target).float().sum().item()
            _, top5 = torch.topk(m_logits, 5, dim=-1)
            total_top5 += (top5 == m_target.unsqueeze(-1)).any(-1).float().sum().item()
            total_tokens += m_target.numel()

    avg_loss = total_loss / total_tokens
    top1 = (total_top1 / total_tokens) * 100.0
    top5 = (total_top5 / total_tokens) * 100.0
    ppl = math.exp(min(avg_loss, 20.0))

    ckpt_path = "/root/checkpoints/model4_subq_wave.pt"
    torch.save({"state_dict": model.state_dict(), "top1": top1, "top5": top5, "ppl": ppl}, ckpt_path)
    volume.commit()

    print(f"\n[MODEL 4 FINISHED] Params: {sum(p.numel() for p in model.parameters()):,} | Time: {time.time()-t0:.1f}s")
    print(f"Val Loss: {avg_loss:.4f} | Mask PPL: {ppl:.2f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")
    print(f"Saved to: {ckpt_path}\n")

@app.local_entrypoint()
def main():
    train_subq_wave.remote()
