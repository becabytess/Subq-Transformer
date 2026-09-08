"""Study 75c: N=8 slots + per-slot embeddings (Study 28 recipe) — Modal-only A10G.

Fixes from 75b (block-PPL stuck ~30, 14% acc on N=32):
  - BLK=8 (PRE=120, L=128), not 32. 4x easier, matches Studies 27/28.
  - Per-slot embeddings: block region gets learned slot bias (8 x d), so the
    8 MASK positions are distinguishable beyond absolute pos.
  - Proper BERT-style objective: mask w, predict w (was shifted next-token).
  - Dynamic k in [2,8] masked per batch (Study 28), U mask band kept narrow.

Run: modal run experiments/study75c_parallel_diffusion_slot8.py
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy", "requests", "tqdm"
)
app = modal.App("study75c-subq-slot8")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_study75c(steps: int = 5000, seq_len: int = 128, d_model: int = 128):
    import math, time, requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda"
    print("=" * 100)
    print(" STUDY 75c: N=8 SLOT DIFFUSION (long bidir + slot emb + BERT objective)")
    print("=" * 100)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url, timeout=60).text
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    V = len(chars)
    MASK = V
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train, val = data[:n].to(device), data[n:].to(device)

    PRE, BLK = 120, 8
    assert PRE + BLK == seq_len
    K = 8
    n_heads, T = 4, 4
    hd = d_model // n_heads

    class MiniSubQ(nn.Module):
        def __init__(self, bidirectional=False, use_slot=False):
            super().__init__()
            self.use_slot = use_slot
            self.offsets = (
                [-21, -8, -3, -1, 1, 3, 8, 21]
                if bidirectional
                else [0, 1, 2, 3, 5, 8, 13, 21]
            )
            self.tok = nn.Embedding(V + 1, d_model)
            self.pos = nn.Embedding(512, d_model)
            if use_slot:
                self.slot = nn.Parameter(torch.randn(BLK, d_model) * 0.02)
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
            qs = q.unsqueeze(3)
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
                else:
                    a = -d
                    if a >= L:
                        kk = torch.zeros_like(k)
                        vv = torch.zeros_like(v)
                        m = torch.zeros(L, dtype=torch.bool, device=device)
                    else:
                        kk = F.pad(k[:, :, a:, :], (0, 0, 0, a))
                        vv = F.pad(v[:, :, a:, :], (0, 0, 0, a))
                        m = torch.ones(L, dtype=torch.bool, device=device)
                        m[L - a :] = False
                s = (qs * kk.unsqueeze(3)).sum(-1) / math.sqrt(hd)
                outs.append((s.squeeze(3), vv, m))
            scores = torch.stack([o[0] for o in outs], -1)
            Vc = torch.stack([o[1] for o in outs], 3)
            mk = torch.stack([o[2] for o in outs], -1)[None, None, :, :]
            scores = scores.masked_fill(~mk, float("-inf"))
            pi = torch.softmax(scores, -1)
            return self.proj(
                (pi.unsqueeze(-1) * Vc).sum(3).transpose(1, 2).reshape(B, L, d_model)
            )

        def forward(self, idx):
            B, L = idx.shape
            h = self.tok(idx) + self.pos(torch.arange(L, device=device)[None, :])
            if self.use_slot and L == seq_len:
                h[:, PRE:, :] = h[:, PRE:, :] + self.slot.unsqueeze(0)
            for _ in range(T):
                ctx = self.attn_ctx(self.ln1(h))
                g = self.w_ih(ctx) + self.w_hh(h)
                r, z, nn_ = g.chunk(3, -1)
                r, z = torch.sigmoid(r), torch.sigmoid(z)
                h = (1 - z) * torch.tanh(nn_) + z * h
                h = h + self.mlp(self.ln2(h))
            return self.head(self.lnf(h))

    def cosine(lr0, s):
        p = s / steps
        return 1e-5 + 0.5 * (lr0 - 1e-5) * (1 + math.cos(math.pi * p))

    def ar_batch(bs=32):
        ix = torch.randint(0, len(train) - seq_len - 1, (bs,), device=device)
        x = torch.stack([train[i : i + seq_len] for i in ix])
        y = torch.stack([train[i + 1 : i + seq_len + 1] for i in ix])
        return x, y

    def dif_batch(bs=32):
        # BERT-style: predict same token, mask only inside last-8 block, k in [2,8]
        ix = torch.randint(0, len(train) - seq_len, (bs,), device=device)
        w = torch.stack([train[i : i + seq_len] for i in ix])
        k = int(torch.randint(2, BLK + 1, (1,)).item())
        perm = torch.rand(bs, BLK, device=device).argsort(-1)[:, :k]
        m = torch.zeros(bs, seq_len, dtype=torch.bool, device=device)
        m.scatter_(1, perm + PRE, True)
        xm = torch.where(m, torch.tensor(MASK, device=device), w)
        return xm, w, m

    @torch.no_grad()
    def eval_ar(m):
        m.eval()
        tot, acc = 0, 0
        for _ in range(100):
            i = int(torch.randint(0, len(val) - seq_len - 1, (1,)).item())
            x, y = val[i : i + seq_len].unsqueeze(0), val[i + 1 : i + seq_len + 1]
            lg = m(x)
            tot += F.cross_entropy(lg.view(-1, V), y.view(-1)).item()
            acc += (lg.argmax(-1)[0] == y).float().mean().item()
        return math.exp(tot / 100), acc / 100

    @torch.no_grad()
    def eval_block(m):
        m.eval()
        tot, acc = 0, 0
        for _ in range(100):
            i = int(torch.randint(0, len(val) - seq_len, (1,)).item())
            w = val[i : i + seq_len].unsqueeze(0)
            xm = w.clone()
            xm[0, PRE:] = MASK
            lg = m(xm)[:, PRE:, :]
            tgt = w[0, PRE:]
            tot += F.cross_entropy(lg.view(-1, V), tgt.view(-1)).item()
            acc += (lg.argmax(-1)[0] == tgt).float().mean().item()
        return math.exp(tot / 100), acc / 100

    @torch.no_grad()
    def gen_ar(m, pre, new=8, temp=0.8):
        m.eval()
        idx = pre.clone()
        for _ in range(new):
            lg = m(idx[:, -seq_len:])[:, -1, :] / temp
            idx = torch.cat([idx, torch.multinomial(F.softmax(lg, -1), 1)], 1)
        return idx[0, -new:]

    @torch.no_grad()
    def gen_maskgit(m, pre, rounds=4, temp=0.7):
        m.eval()
        cur = torch.full((1, BLK), MASK, dtype=torch.long, device=device)
        per = BLK // rounds
        for r in range(rounds):
            lg = m(torch.cat([pre, cur], 1))[:, PRE:, :] / temp
            pr = F.softmax(lg, -1)
            cf, pd = pr.max(-1)
            k = per if r < rounds - 1 else int((cur == MASK).sum())
            if k <= 0:
                break
            _, pk = torch.topk(cf.masked_fill(cur != MASK, -1), k)
            cur[0, pk[0]] = pd[0, pk[0]]
        return cur[0]

    # Phase A: AR
    ar = MiniSubQ(False, False).to(device)
    opt = torch.optim.AdamW(ar.parameters(), lr=3e-4)
    ar.train()
    t0 = time.time()
    for s in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine(3e-4, s)
        x, y = ar_batch()
        opt.zero_grad()
        loss = F.cross_entropy(ar(x).view(-1, V), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
        opt.step()
        if s % 1000 == 0:
            ppl, acc = eval_ar(ar)
            print(
                f"[AR] {s}/{steps} loss={loss.item():.3f} ppl={ppl:.2f} acc={acc * 100:.1f}%",
                flush=True,
            )
    print(f"[AR] done {time.time() - t0:.0f}s ppl={eval_ar(ar)[0]:.2f}", flush=True)

    # Phase B: slot diffusion (transplant non-slot weights)
    dm = MiniSubQ(True, True).to(device)
    dm.load_state_dict(ar.state_dict(), strict=False)
    opt = torch.optim.AdamW(dm.parameters(), lr=2e-4)
    dm.train()
    t0 = time.time()
    for s in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine(2e-4, s)
        xm, w, m = dif_batch()
        opt.zero_grad()
        loss = F.cross_entropy(dm(xm).view(-1, V)[m.view(-1)], w.view(-1)[m.view(-1)])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dm.parameters(), 1.0)
        opt.step()
        if s % 1000 == 0:
            bp, ba = eval_block(dm)
            print(
                f"[DIF] {s}/{steps} loss={loss.item():.3f} block-ppl={bp:.2f} block-acc={ba * 100:.1f}%",
                flush=True,
            )
    bp, ba = eval_block(dm)
    print(
        f"[DIF] done {time.time() - t0:.0f}s block-ppl={bp:.2f} block-acc={ba * 100:.1f}%",
        flush=True,
    )

    dec = lambda t: "".join(itos.get(i.item(), "?") for i in t)
    for j in range(3):
        i = int(torch.randint(0, len(val) - seq_len, (1,)).item())
        w = val[i : i + seq_len]
        pre = w[:PRE].unsqueeze(0)
        tgt = w[PRE:]
        t0 = time.time()
        a = gen_ar(ar, pre, BLK)
        ta = (time.time() - t0) * 1000
        t0 = time.time()
        mg = gen_maskgit(dm, pre)
        tm = (time.time() - t0) * 1000
        print(f"\n--- prefix {j} ---", flush=True)
        print(f" truth   : {dec(tgt)!r}", flush=True)
        print(
            f" AR {ta:.0f}ms acc={(a == tgt).float().mean().item() * 100:.0f}%: {dec(a)!r}",
            flush=True,
        )
        print(
            f" MaskGIT {tm:.0f}ms acc={(mg == tgt).float().mean().item() * 100:.0f}%: {dec(mg)!r}",
            flush=True,
        )

    torch.save(
        {"ar": ar.state_dict(), "dm": dm.state_dict(), "V": V},
        " /models/study75c_slot8.pt".replace(" ", ""),
    )
    volume.commit()
    print("saved /models/study75c_slot8.pt")
    return {"block_ppl": bp, "block_acc": ba}


@app.local_entrypoint()
def main():
    run_study75c.remote()
