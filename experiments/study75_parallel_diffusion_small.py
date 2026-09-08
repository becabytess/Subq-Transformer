"""Study 75: Canonical 1-Layer Harmonic SubQ -> Parallel Diffusion (Modal-only).

What this proves (cheap, T4, ~10 min):
  A. AR baseline (causal offsets, T=4) on TinyShakespeare char-level
  B. Masked-diffusion twin (same 1-layer, bidirectional offsets, [MASK] + MaskGIT)
  C. Zero-train Jacobi parallel decode using the AR weights (no retrain)

Run (local PC only launches, never trains):
  modal run experiments/study75_parallel_diffusion_small.py

All torch code executes inside @app.function(gpu="T4").
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy", "requests", "tqdm"
)
app = modal.App("study75-subq-parallel-diffusion-small")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_study75(steps: int = 1200, seq_len: int = 128, d_model: int = 128):
    import math, time, os, requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 100)
    print(" STUDY 75: 1-LAYER SubQ AR vs PARALLEL DIFFUSION (MaskGIT + Jacobi)")
    print(f" Device={device} | steps={steps} L={seq_len} d={d_model}")
    print("=" * 100)

    # ---- 1. TinyShakespeare char data ----
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url, timeout=60).text
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    V = len(chars)
    MASK = V  # extra id for [MASK]
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]
    print(f"vocab={V} (+1 MASK={MASK}) train={len(train)} val={len(val)}")

    def batch(bs=32):
        ix = torch.randint(0, len(train) - seq_len - 1, (bs,))
        x = torch.stack([train[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([train[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        return x, y

    # ---- 2. Minimal canonical 1-layer SubQ (causal OR bidir via flag) ----
    K = 8
    n_heads, T = 4, 4
    hd = d_model // n_heads

    class MiniSubQ(nn.Module):
        def __init__(self, bidirectional=False):
            super().__init__()
            self.bidirectional = bidirectional
            # causal offsets vs symmetric offsets (same K)
            self.offsets = (
                [-6, -4, -3, -2, -1, 1, 2, 3][:K]
                if bidirectional
                else [0, 1, 2, 3, 5, 8, 13, 21][:K]
            )
            self.tok = nn.Embedding(V + 1, d_model)
            self.pos = nn.Embedding(512, d_model)
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.proj = nn.Linear(d_model, d_model, bias=False)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)
            self.w_hh = nn.Linear(d_model, 3 * d_model, bias=False)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.lnf = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, V, bias=False)

        def attn_ctx(self, h):
            B, L, _ = h.shape
            q, k, v = self.qkv(h).chunk(3, -1)
            q = q.view(B, L, n_heads, hd).transpose(1, 2)
            k = k.view(B, L, n_heads, hd).transpose(1, 2)
            v = v.view(B, L, n_heads, hd).transpose(1, 2)
            qs = q.unsqueeze(3)  # B,H,L,1,hd
            outs = []
            for d in self.offsets:
                if d >= 0:
                    if d >= L:
                        kk = torch.zeros_like(k)
                        vv = torch.zeros_like(v)
                    else:
                        kk = F.pad(k[:, :, : L - d, :], (0, 0, d, 0)) if d > 0 else k
                        vv = F.pad(v[:, :, : L - d, :], (0, 0, d, 0)) if d > 0 else v
                    m = torch.ones(L, dtype=torch.bool, device=device)
                    m[: min(d, L)] = False
                else:  # forward look (bidir only)
                    a = -d
                    if a >= L:
                        kk = torch.zeros_like(k)
                        vv = torch.zeros_like(v)
                    else:
                        kk = F.pad(k[:, :, a:, :], (0, 0, 0, a))
                        vv = F.pad(v[:, :, a:, :], (0, 0, 0, a))
                    m = torch.ones(L, dtype=torch.bool, device=device)
                    m[L - a :] = False if a < L else True
                    if a >= L:
                        m[:] = False
                s = (qs * kk.unsqueeze(3)).sum(-1) / math.sqrt(
                    hd
                )  # B,H,L,1->K? single offset
                outs.append((s.squeeze(3), vv, m))
            scores = torch.stack([o[0] for o in outs], -1)
            Vc = torch.stack([o[1] for o in outs], 3)
            mk = torch.stack([o[2] for o in outs], -1)[None, None, :, :]
            scores = scores.masked_fill(~mk, float("-inf"))
            pi = torch.softmax(scores, -1)
            ctx = (pi.unsqueeze(-1) * Vc).sum(3).transpose(1, 2).reshape(B, L, d_model)
            return self.proj(ctx)

        def forward(self, idx):
            B, L = idx.shape
            h = self.tok(idx) + self.pos(torch.arange(L, device=device)[None, :])
            for _ in range(T):
                ctx = self.attn_ctx(self.ln1(h))
                g = self.w_ih(ctx) + self.w_hh(h)
                r, z, nn_ = g.chunk(3, -1)
                r, z = torch.sigmoid(r), torch.sigmoid(z)
                h = (1 - z) * torch.tanh(
                    nn_
                ) + z * h  # contractive GRU settle (1/sqrt(T) folded into init)
                h = h + self.mlp(self.ln2(h))
            return self.head(self.lnf(h))

    @torch.no_grad()
    def eval_ar(m, n=200):
        m.eval()
        tot = 0
        for _ in range(n):
            i = torch.randint(0, len(val) - seq_len - 1, (1,)).item()
            x, y = (
                val[i : i + seq_len].unsqueeze(0).to(device),
                val[i + 1 : i + seq_len + 1].to(device),
            )
            tot += F.cross_entropy(m(x).view(-1, V), y.view(-1)).item()
        return math.exp(tot / n)

    @torch.no_grad()
    def generate_ar(m, prompt, new=80, temp=0.8):
        m.eval()
        idx = torch.tensor(
            [[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device
        )
        for _ in range(new):
            logits = m(idx[:, -seq_len:])[:, -1, :] / temp
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return "".join(itos[i] for i in idx[0].tolist() if i in itos)

    @torch.no_grad()
    def generate_jacobi(m, prompt, new=32, iters=8, temp=0.8):
        """Zero-train parallel: draft all `new` tokens at once, refine whole block jointly."""
        m.eval()
        pre = torch.tensor(
            [[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device
        )
        draft = torch.randint(0, V, (1, new), device=device)
        idx = torch.cat([pre, draft], 1)
        L0 = pre.size(1)
        for _ in range(iters):
            logits = m(idx[:, -(seq_len):]) / temp
            # map back: only update draft region
            full = idx.size(1)
            win = min(full, seq_len)
            tail = F.softmax(logits[:, -win:, :], -1).argmax(-1)
            idx[:, -win:] = torch.where(
                torch.arange(full, device=device)[None, -win:] >= L0,
                tail,
                idx[:, -win:],
            )
        return "".join(itos[i] for i in idx[0, L0:].tolist() if i in itos)

    @torch.no_grad()
    def generate_maskgit(m, prompt, new=32, rounds=4, temp=0.7):
        """MaskGIT-style: start all-MASK, unmask most-confident 1/R per round."""
        m.eval()
        pre = torch.tensor(
            [[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device
        )
        cur = torch.full((1, new), MASK, dtype=torch.long, device=device)
        per = new // rounds
        for r in range(rounds):
            idx = torch.cat([pre, cur], 1)[:, -seq_len:]
            probs = F.softmax(m(idx)[:, -new:, :] / temp, -1)
            conf, pred = probs.max(-1)
            conf[~(cur == MASK)] = 2.0  # keep committed
            k = per if r < rounds - 1 else (cur == MASK).sum().item()
            if k <= 0:
                break
            _, pick = torch.topk(
                (conf + (cur != MASK).float() * 2).masked_fill(cur != MASK, -1), k
            )
            cur[0, pick[0]] = pred[0, pick[0]]
        return "".join(itos.get(i.item(), "?") for i in cur[0])

    # ---- 3A. Train AR baseline ----
    ar = MiniSubQ(bidirectional=False).to(device)
    opt = torch.optim.AdamW(ar.parameters(), lr=3e-4)
    t0 = time.time()
    ar.train()
    for s in range(1, steps + 1):
        x, y = batch()
        opt.zero_grad()
        loss = F.cross_entropy(ar(x).view(-1, V), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
        opt.step()
        if s % 400 == 0:
            print(
                f"[AR] {s}/{steps} loss={loss.item():.3f} ppl={eval_ar(ar, 50):.1f}",
                flush=True,
            )
    ar_ppl = eval_ar(ar)
    print(
        f"[AR] done {time.time() - t0:.0f}s ppl={ar_ppl:.2f} sample: {generate_ar(ar, 'To be, or not', 40)!r}"
    )

    # ---- 3B. Train masked-diffusion twin (same budget, bidir) ----
    dm = MiniSubQ(bidirectional=True).to(device)
    dm.load_state_dict(ar.state_dict(), strict=False)  # transplant, like Study 35/65
    opt = torch.optim.AdamW(dm.parameters(), lr=3e-4)
    t0 = time.time()
    dm.train()
    for s in range(1, steps + 1):
        x, y = batch()
        m = (
            torch.rand_like(x.float()) < torch.rand(1).item() * 0.8 + 0.1
        )  # random mask ratio
        xm = torch.where(m, torch.tensor(MASK, device=device), x)
        opt.zero_grad()
        loss = F.cross_entropy(dm(xm).view(-1, V)[m.view(-1)], y.view(-1)[m.view(-1)])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dm.parameters(), 1.0)
        opt.step()
        if s % 400 == 0:
            print(f"[DIF] {s}/{steps} masked-loss={loss.item():.3f}", flush=True)
    print(f"[DIF] done {time.time() - t0:.0f}s")

    # ---- 4. Compare: wall-clock + quality ----
    p = "To be, or not"
    for name, fn in [
        ("AR-token-by-token", lambda: generate_ar(ar, p, 32)),
        ("Jacobi-parallel(zero-train)", lambda: generate_jacobi(ar, p, 32)),
        ("MaskGIT-parallel(4 rounds)", lambda: generate_maskgit(dm, p, 32)),
    ]:
        t0 = time.time()
        s = fn()
        dt = time.time() - t0
        print(f"{name:32s} {dt * 1000:6.0f}ms | {s!r}")

    torch.save(
        {"ar": ar.state_dict(), "dm": dm.state_dict(), "V": V, "ppl_ar": ar_ppl},
        "/models/study75_parallel_small.pt",
    )
    volume.commit()
    print("saved /models/study75_parallel_small.pt")
    return {"ppl_ar": ar_ppl}


@app.local_entrypoint()
def main():
    run_study75.remote()
