import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2.0", "numpy", "requests")
)

app = modal.App("exp-model1-dense-bert-1l", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=1200, volumes={"/root/checkpoints": volume})
def train_dense_bert_1layer():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import requests

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90)
    print("  MODEL 1: DENSE BERT (1 LAYER, DENSE ATTENTION)")
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

    class DenseBertBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

        def forward(self, x):
            norm_x = self.ln1(x)
            attn_out, _ = self.attn(norm_x, norm_x, norm_x)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(vocab_size, d_model)
            self.wpe = nn.Embedding(seq_len, d_model)
            self.layer = DenseBertBlock()
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            B, L = x.shape
            pos = torch.arange(0, L, dtype=torch.long, device=x.device).unsqueeze(0)
            h = self.wte(x) + self.wpe(pos)
            h = self.layer(h)
            h = self.ln_f(h)
            return self.lm_head(h)

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
            logits = model(inp)
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
            logits = model(inp)
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

    ckpt_path = "/root/checkpoints/model1_bert1l.pt"
    torch.save({"state_dict": model.state_dict(), "top1": top1, "top5": top5, "ppl": ppl}, ckpt_path)
    volume.commit()

    print(f"\n[MODEL 1 FINISHED] Params: {sum(p.numel() for p in model.parameters()):,} | Time: {time.time()-t0:.1f}s")
    print(f"Val Loss: {avg_loss:.4f} | Mask PPL: {ppl:.2f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")
    print(f"Saved to: {ckpt_path}\n")

@app.local_entrypoint()
def main():
    train_dense_bert_1layer.remote()
