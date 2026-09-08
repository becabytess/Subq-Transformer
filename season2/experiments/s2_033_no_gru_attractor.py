"""S2-033: Quick attractor diagnostic for hard-coded-offset SubQ without GRU.

This mirrors the old attractor probe's task and offset graph, but replaces the
GRU state update with direct attention-state replacement. The MLP is applied
once after all transport hops, matching the transport-first Season 2 variant.
"""

import json
import math
import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-033-no-gru-attractor", image=image)


@app.function(image=image, gpu="T4", timeout=1800)
def run(seed: int = 42):
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len, batch_size = 256, 32
    d_model, d_mlp, n_heads = 128, 512, 4
    max_T, train_steps = 6, 1500
    offsets = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 255]
    K = len(offsets)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "/tmp/input.txt")
    text = open("/tmp/input.txt", "r", encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    vocab_size = len(chars)

    def get_batch(source):
        starts = torch.randint(len(source) - seq_len - 1, (batch_size,))
        x = torch.stack([source[int(i): int(i) + seq_len] for i in starts])
        y = torch.stack([source[int(i) + 1: int(i) + seq_len + 1] for i in starts])
        return x.to(device), y.to(device)

    class NoGRUReplacement(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            self.ln_attn = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.o = nn.Linear(d_model, d_model, bias=False)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            targets = torch.zeros(seq_len, K, dtype=torch.long)
            valid = torch.zeros(seq_len, K, dtype=torch.bool)
            for i in range(seq_len):
                for j, offset in enumerate(offsets):
                    source = i - offset
                    if source >= 0:
                        targets[i, j] = source
                        valid[i, j] = True
            self.register_buffer("targets", targets)
            self.register_buffer("valid", valid)

        def transport_step(self, state):
            bsz, length, _ = state.shape
            z = self.ln_attn(state)
            q = self.q(z).view(bsz, length, n_heads, -1).transpose(1, 2)
            k = self.k(z).view(bsz, length, n_heads, -1).transpose(1, 2)
            v = self.v(z).view(bsz, length, n_heads, -1).transpose(1, 2)
            indices = self.targets[:length]
            valid = self.valid[:length].view(1, 1, length, K)
            gathered_k = k[:, :, indices, :]
            gathered_v = v[:, :, indices, :]
            scores = (q.unsqueeze(3) * gathered_k).sum(-1) / math.sqrt(d_model // n_heads)
            weights = F.softmax(scores.masked_fill(~valid, -1e4), dim=-1)
            context = (weights.unsqueeze(-1) * gathered_v).sum(3)
            # This is the deliberate no-GRU/full-replacement update.
            return context.transpose(1, 2).contiguous().view(bsz, length, d_model)

        def trajectory(self, idx, T=max_T, noise_step=None, noise_std=0.0):
            bsz, length = idx.shape
            pos = torch.arange(length, device=idx.device).unsqueeze(0)
            state = self.tok(idx) + self.pos(pos)
            states = [state]
            for step in range(1, T + 1):
                state = self.transport_step(state)
                if noise_step == step and noise_std:
                    state = state + torch.randn_like(state) * noise_std
                states.append(state)
            return states

        def logits_from_state(self, state):
            state = state + self.mlp(self.ln_mlp(state))
            return self.head(self.ln_f(state))

    model = NoGRUReplacement().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    model.train()
    for step in range(train_steps):
        x, y = get_batch(train_data)
        T = int(torch.randint(1, max_T + 1, ()).item())
        states = model.trajectory(x, T=T)
        logits = model.logits_from_state(states[-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    velocity_sum = torch.zeros(max_T, device=device)
    cosine_sum = torch.zeros(max_T, device=device)
    count = 0
    recovery = {str(sigma): {str(t): [] for t in range(1, max_T + 1)} for sigma in (0.1, 0.5, 1.0)}
    ppl = {str(t): [] for t in range(1, max_T + 1)}

    with torch.no_grad():
        for _ in range(25):
            x, y = get_batch(val_data)
            clean = model.trajectory(x)
            for t in range(1, max_T + 1):
                delta = clean[t] - clean[t - 1]
                velocity_sum[t - 1] += delta.norm(dim=-1).mean()
                cosine_sum[t - 1] += F.cosine_similarity(clean[t], clean[t - 1], dim=-1).mean()
                out = model.logits_from_state(clean[t])
                ppl[str(t)].append(float(math.exp(F.cross_entropy(out.reshape(-1, vocab_size), y.reshape(-1)).item())))

            for sigma in (0.1, 0.5, 1.0):
                pert = model.trajectory(x, noise_step=1, noise_std=sigma)
                initial_error = (pert[1] - clean[1]).norm(dim=-1).mean().item() + 1e-8
                for t in range(1, max_T + 1):
                    err = (pert[t] - clean[t]).norm(dim=-1).mean().item() / initial_error
                    recovery[str(sigma)][str(t)].append(err)
            count += 1

    result = {
        "experiment": "S2-033",
        "seed": seed,
        "architecture": "hard-coded offsets + no-GRU direct replacement + final MLP",
        "offsets": offsets,
        "train_steps": train_steps,
        "velocity": [round(float(v / count), 6) for v in velocity_sum],
        "consecutive_cosine": [round(float(v / count), 6) for v in cosine_sum],
        "ppl": {t: round(sum(vals) / len(vals), 4) for t, vals in ppl.items()},
        "error_ratio_after_noise": {
            sigma: {t: round(sum(vals) / len(vals), 6) for t, vals in steps.items()}
            for sigma, steps in recovery.items()
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main():
    result = run.remote()
    print(json.dumps(result, indent=2), flush=True)
    with open("season2/results/s2_033_no_gru_attractor.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
