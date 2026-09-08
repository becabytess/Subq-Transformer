"""Study 75b: Parallel diffusion with LONG bidir offsets + 5k steps (Modal-only, A10G).

Fixes from 75 (PPL 7.0 AR but MaskGIT garbage):
  - bidir offsets LONG symmetric [-21,-8,-3,-1,1,3,8,21] (was local [-6..3])
  - 5000 steps each phase + cosine LR (was 1200 const LR)
  - narrower mask ratio U[0.3,0.7] (was U[0.1,0.9], noisy loss 0.5->1.5)
  - full-length (96+32) eval/generation matching train L=128 (was 13-char short prompt shift)

Run (local only launches):
  modal run experiments/study75b_parallel_diffusion_long.py
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy", "requests", "tqdm"
)
app = modal.App("study75b-subq-parallel-long")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_study75b(steps: int = 5000, seq_len: int = 128, d_model: int = 128):
    import math, time, requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda"
    print("=" * 100)
    print(" STUDY 75b: LONG-RANGE BIDIR DIFFUSION (5k steps, cosine, U[0.3,0.7] masks)")
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
    print(f"vocab={V} train={len(train)} val={len(val)}", flush=True)

    PRE, BLK = 96, 32  # 96 prefix + 32 block = 128 = train L

    def ar_batch(bs=32):
        ix = torch.randint(0, len(train) - seq_len - 1, (bs,), device=device)
        x = torch.stack([train[i : i + seq_len] for i in ix])
        y = torch.stack([train[i + 1 : i + seq_len + 1] for i in ix])
        return x, y

    K = 8
    n_heads, T = 4, 4
    hd = d_model // n_heads

    class MiniSubQ(nn.Module):
        def __init__(self, bidirectional=False):
            super().__init__()
            self.offsets = (
                [-21, -8, -3, -1, 1, 3, 8, 21]
                if bidirectional
                else [0, 1, 2, 3, 5, 8, 13, 21]
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

    @torch.no_grad()
    def eval_ar(m):
        m.eval()
        tot, acc, tot_t = 0, 0, 0
        for _ in range(100):
            i = torch.randint(0, len(val) - seq_len - 1, (1,)).item()
            x, y = val[i : i + seq_len].unsqueeze(0), val[i + 1 : i + seq_len + 1]
            lg = m(x)
            tot += F.cross_entropy(lg.view(-1, V), y.view(-1)).item()
            acc += (lg.argmax(-1)[0] == y).float().mean().item()
        return math.exp(tot / 100), acc / 100

    @torch.no_grad()
    def eval_block(m):
        # full-mask 32-block prediction from 96 prefix (matches train length)
        m.eval()
        tot, acc = 0, 0
        for _ in range(100):
            i = torch.randint(0, len(val) - PRE - BLK - 1, (1,)).item()
            pre = val[i : i + PRE].unsqueeze(0)
            tgt = val[i + PRE : i + PRE + BLK]
            cur = torch.full((1, BLK), MASK, dtype=torch.long, device=device)
            lg = m(torch.cat([pre, cur], 1))[:, -BLK:, :]
            tot += F.cross_entropy(lg.view(-1, V), tgt.view(-1)).item()
            acc += (lg.argmax(-1)[0] == tgt).float().mean().item()
        return math.exp(tot / 100), acc / 100

    @torch.no_grad()
    def gen_ar(m, pre, new=32, temp=0.8):
        m.eval()
        idx = pre.clone()
        for _ in range(new):
            lg = m(idx[:, -seq_len:])[:, -1, :] / temp
            idx = torch.cat([idx, torch.multinomial(F.softmax(lg, -1), 1)], 1)
        return idx[0, -new:]

    @torch.no_grad()
    def gen_maskgit(m, pre, rounds=8, temp=0.7):
        m.eval()
        cur = torch.full((1, BLK), MASK, dtype=torch.long, device=device)
        per = BLK // rounds
        for r in range(rounds):
            lg = m(torch.cat([pre, cur], 1))[:, -BLK:, :] / temp
            pr = F.softmax(lg, -1)
            cf, pd = pr.max(-1)
            k = per if r < rounds - 1 else int((cur == MASK).sum())
            if k <= 0:
                break
            _, pk = torch.topk(cf.masked_fill(cur != MASK, -1), k)
            cur[0, pk[0]] = pd[0, pk[0]]
        return cur[0]

    # ---- Phase A: AR ----
    ar = MiniSubQ(False).to(device)
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
    ppl_ar, acc_ar = eval_ar(ar)
    print(
        f"[AR] done {time.time() - t0:.0f}s ppl={ppl_ar:.2f} acc={acc_ar * 100:.1f}%",
        flush=True,
    )

    # ---- Phase B: diffusion twin ----
    dm = MiniSubQ(True).to(device)
    dm.load_state_dict(ar.state_dict(), strict=False)
    opt = torch.optim.AdamW(dm.parameters(), lr=2e-4)
    dm.train()
    t0 = time.time()
    for s in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine(2e-4, s)
        x, y = ar_batch()
        ratio = 0.3 + 0.4 * torch.rand(1).item()  # U[0.3,0.7] fixed narrow band
        m = torch.rand_like(x.float()) < ratio
        xm = torch.where(m, MASK, x)
        opt.zero_grad()
        loss = F.cross_entropy(dm(xm).view(-1, V)[m.view(-1)], y.view(-1)[m.view(-1)])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dm.parameters(), 1.0)
        opt.step()
        if s % 1000 == 0:
            bp, ba = eval_block(dm)
            print(
                f"[DIF] {s}/{steps} ratio={ratio:.2f} loss={loss.item():.3f} block-ppl={bp:.2f} block-acc={ba * 100:.1f}%",
                flush=True,
            )
    bp, ba = eval_block(dm)
    print(
        f"[DIF] done {time.time() - t0:.0f}s block-ppl={bp:.2f} block-acc={ba * 100:.1f}%",
        flush=True,
    )

    # ---- Compare on same val prefixes ----
    dec = lambda t: "".join(itos.get(i.item(), "?") for i in t)
    for j in range(3):
        i = torch.randint(0, len(val) - PRE - BLK - 1, (1,)).item()
        pre = val[i : i + PRE].unsqueeze(0)
        tgt = val[i + PRE : i + PRE + BLK]
        t0 = time.time()
        a = gen_ar(ar, torch.cat([pre, tgt[:0].unsqueeze(0)], 1)[:, :PRE], BLK)
        ta = (time.time() - t0) * 1000
        t0 = time.time()
        mg = gen_maskgit(dm, pre)
        tm = (time.time() - t0) * 1000
        ma = (mg == tgt).float().mean().item() * 100
        aa = (a == tgt).float().mean().item() * 100
        print(f"\n--- prefix {j} ---", flush=True)
        print(f" truth   : {dec(tgt)!r}", flush=True)
        print(f" AR {ta:.0f}ms acc={aa:.0f}%: {dec(a)!r}", flush=True)
        print(f" MaskGIT {tm:.0f}ms acc={ma:.0f}%: {dec(mg)!r}", flush=True)

    torch.save(
        {"ar": ar.state_dict(), "dm": dm.state_dict(), "V": V},
        " /models/study75b_long.pt".replace(" ", ""),
    )
    volume.commit()
    print("saved /models/study75b_long.pt")
    return {"ppl_ar": ppl_ar, "block_ppl": bp, "block_acc": ba}


@app.local_entrypoint()
def main():
    run_study75b.remote()
