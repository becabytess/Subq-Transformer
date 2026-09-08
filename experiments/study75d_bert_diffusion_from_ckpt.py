"""Study 75d v1: MaskGIT parallel diffusion FROM the saved 12L SubQ-BERT ckpt (Modal-only, A10G).

Starts from free knowledge — no from-scratch training:
  ckpt = /root/checkpoints/subq_bert_12layer_full_best.pt  (Study 35: 58.94%, PPL 8.49)
  arch = Full12LayerSubQBert, K=15 symmetric [-64..64], 110M params (same as Study 35 file)

v1 does:
  1. Rebuild arch around bert-base-uncased, load ckpt, verify mask-PPL ~8.5
  2. Sanity: Paris / Python single-mask (should be ~96%/91%)
  3. Parallel: N=8 forward-span MaskGIT (4 rounds x 2) vs one-shot, on WikiText-2 val
     prefixes — block acc + wall time + samples (Study 36 setup, MaskGIT decoding)
  4. Light 500-step contiguous-span tune (prefix+8MASK, BERT-objective on span only)

Run: modal run experiments/study75d_bert_diffusion_from_ckpt.py
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "transformers>=4.40.0", "numpy", "requests", "accelerate"
)
app = modal.App("study75d-bert-diffusion-ckpt")
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)


@app.function(
    image=image, gpu="A10G", timeout=3600, volumes={"/root/checkpoints": volume}
)
def run_75d(steps: int = 500, seq_pre: int = 96, blk: int = 8):
    import math, time, os, requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertForMaskedLM, BertTokenizer

    device = "cuda"
    torch.manual_seed(0)
    print("=" * 110)
    print(" STUDY 75d v1: PARALLEL DIFFUSION FROM SAVED 12L SubQ-BERT (free knowledge)")
    print("=" * 110)

    # ---- 1. Rebuild arch (identical to Study 35) + load ckpt ----
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    MASK = tok.mask_token_id
    V = tok.vocab_size
    base = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    d_model, n_heads, hd = 768, 12, 64
    offsets = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]

    class SubQWaveSelfAttention(nn.Module):
        def __init__(self, o):
            super().__init__()
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
            self.q_proj = nn.Linear(768, 768)
            self.k_proj = nn.Linear(768, 768)
            self.v_proj = nn.Linear(768, 768)
            self.out_proj = nn.Linear(768, 768)
            self.temp_scale = nn.Parameter(torch.ones(1, 12, 1, 1))
            with torch.no_grad():
                self.q_proj.weight.copy_(o.self.query.weight)
                self.q_proj.bias.copy_(o.self.query.bias)
                self.k_proj.weight.copy_(o.self.key.weight)
                self.k_proj.bias.copy_(o.self.key.bias)
                self.v_proj.weight.copy_(o.self.value.weight)
                self.v_proj.bias.copy_(o.self.value.bias)
                self.out_proj.weight.copy_(o.output.dense.weight)
                self.out_proj.bias.copy_(o.output.dense.bias)

        def forward(self, x):
            B, L, _ = x.shape
            q = self.q_proj(x).view(B, L, 12, 64).transpose(1, 2)
            k = self.k_proj(x).view(B, L, 12, 64).transpose(1, 2)
            v = self.v_proj(x).view(B, L, 12, 64).transpose(1, 2)
            KL = []
            VL = []
            ML = []
            for dv in self.offsets:
                d = int(dv)
                if d >= 0:
                    if d >= L:
                        ks = torch.zeros_like(k)
                        vs = torch.zeros_like(v)
                        m = torch.zeros(B, 1, L, dtype=torch.bool, device=x.device)
                    elif d == 0:
                        ks, vs = k, v
                        m = torch.ones(B, 1, L, dtype=torch.bool, device=x.device)
                    else:
                        ks = F.pad(
                            k[:, :, :-d, :] if d < L else k[:, :, :0, :], (0, 0, d, 0)
                        )
                        vs = F.pad(v[:, :, :-d, :], (0, 0, d, 0))
                        m = torch.cat(
                            [
                                torch.zeros(B, 1, d, dtype=torch.bool, device=x.device),
                                torch.ones(
                                    B, 1, L - d, dtype=torch.bool, device=x.device
                                ),
                            ],
                            -1,
                        )
                else:
                    a = -d
                    if a >= L:
                        ks = torch.zeros_like(k)
                        vs = torch.zeros_like(v)
                        m = torch.zeros(B, 1, L, dtype=torch.bool, device=x.device)
                    else:
                        ks = F.pad(k[:, :, a:, :], (0, 0, 0, a))
                        vs = F.pad(v[:, :, a:, :], (0, 0, 0, a))
                        m = torch.cat(
                            [
                                torch.ones(
                                    B, 1, L - a, dtype=torch.bool, device=x.device
                                ),
                                torch.zeros(B, 1, a, dtype=torch.bool, device=x.device),
                            ],
                            -1,
                        )
                KL.append(ks)
                VL.append(vs)
                ML.append(m)
            Kc = torch.stack(KL, 3)
            Vc = torch.stack(VL, 3)
            Mk = torch.stack(ML, 3)
            s = (q.unsqueeze(3) * Kc).sum(-1) / math.sqrt(64) * self.temp_scale
            s = s.masked_fill(~Mk, float("-inf"))
            p = torch.nan_to_num(F.softmax(s, -1), nan=0.0)
            return self.out_proj(
                (p.unsqueeze(-1) * Vc).sum(3).transpose(1, 2).reshape(B, L, 768)
            )

    class SubQBertLayer(nn.Module):
        def __init__(self, o):
            super().__init__()
            self.attn = SubQWaveSelfAttention(o.attention)
            self.ln1 = o.attention.output.LayerNorm
            self.intermediate = o.intermediate
            self.output = o.output

        def forward(self, h):
            h = self.ln1(h + self.attn(h))
            return self.output(self.intermediate(h), h)

    class Full12LayerSubQBert(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.embeddings = b.bert.embeddings
            self.layers = nn.ModuleList(
                [SubQBertLayer(b.bert.encoder.layer[i]) for i in range(12)]
            )
            self.cls = b.cls

        def forward(self, idx):
            h = self.embeddings(idx)
            for l in self.layers:
                h = l(h)
            return self.cls(h)

    m = Full12LayerSubQBert(base).to(device)
    ckpt_path = "/root/checkpoints/subq_bert_12layer_full_best.pt"
    assert os.path.exists(ckpt_path), f"missing {ckpt_path}"
    ck = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ck["state_dict"], strict=True)
    print(
        f"loaded {ckpt_path} (Top-1={ck.get('top1', 0):.2f}% PPL={ck.get('ppl', 0):.2f})",
        flush=True,
    )
    print(f"params: {sum(p.numel() for p in m.parameters()):,}", flush=True)

    # ---- 2. Data (WikiText-2, same as Study 35) ----
    tr = requests.get(
        "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
        timeout=60,
    ).text
    va = requests.get(
        "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt",
        timeout=60,
    ).text
    train = torch.tensor(
        tok.encode(tr, add_special_tokens=False), dtype=torch.long, device=device
    )
    val = torch.tensor(
        tok.encode(va, add_special_tokens=False), dtype=torch.long, device=device
    )
    print(f"WikiText-2 WordPiece: train={len(train)} val={len(val)}", flush=True)
    L = seq_pre + blk

    @torch.no_grad()
    def eval_mask(m_, n=40, p=0.15):
        m_.eval()
        tl = tc = t5 = tt = 0
        for _ in range(n):
            s = torch.randint(0, len(val) - L - 1, (16,))
            raw = torch.stack([val[i : i + L] for i in s])
            mk = torch.rand(16, L, device=device) < p
            inp = raw.clone()
            inp[mk] = MASK
            lg = m_(inp)
            loss = F.cross_entropy(lg[mk], raw[mk])
            tl += loss.item() * raw[mk].numel()
            pr = lg[mk].argmax(-1)
            tc += (pr == raw[mk]).float().sum().item()
            _, t5_ = torch.topk(lg[mk], 5, -1)
            t5 += (t5_ == raw[mk].unsqueeze(-1)).any(-1).float().sum().item()
            tt += raw[mk].numel()
        return math.exp(tl / tt), tc / tt * 100

    @torch.no_grad()
    def eval_span(m_, n=60):
        # N=8 contiguous forward span, full-mask (hardest) — Study 36 setup
        m_.eval()
        tl = tc = tt = 0
        for _ in range(n):
            i = int(torch.randint(0, len(val) - L, (1,)).item())
            w = val[i : i + L].unsqueeze(0)
            xm = w.clone()
            xm[0, seq_pre:] = MASK
            lg = m_(xm)[:, seq_pre:, :]
            tgt = w[0, seq_pre:]
            tl += F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1)).item() * blk
            tc += (lg.argmax(-1)[0] == tgt).float().sum().item()
            tt += blk
        return math.exp(tl / tt), tc / tt * 100

    @torch.no_grad()
    def gen_oneshot(m_, pre):
        return m_(
            torch.cat(
                [pre, torch.full((1, blk), MASK, dtype=torch.long, device=device)], 1
            )
        )[:, seq_pre:, :].argmax(-1)[0]

    @torch.no_grad()
    def gen_maskgit(m_, pre, rounds=4):
        cur = torch.full((1, blk), MASK, dtype=torch.long, device=device)
        per = blk // rounds
        for r in range(rounds):
            lg = m_(torch.cat([pre, cur], 1))[:, seq_pre:, :]
            pr = F.softmax(lg, -1)
            cf, pd = pr.max(-1)
            k = per if r < rounds - 1 else int((cur == MASK).sum())
            _, pk = torch.topk(cf.masked_fill(cur != MASK, -1), k)
            cur[0, pk[0]] = pd[0, pk[0]]
        return cur[0]

    # ---- 3. Verify + sanity ----
    ppl, acc = eval_mask(m)
    print(
        f"[verify] mask-PPL={ppl:.2f} Top-1={acc:.2f}% (expect ~8.5 / ~59%)", flush=True
    )
    m.eval()
    for p in [
        "Paris is the [MASK] of France.",
        "Python is a popular programming [MASK].",
    ]:
        ids = torch.tensor(
            [tok.encode(p, add_special_tokens=True)], dtype=torch.long, device=device
        )
        pos = (ids == MASK).nonzero(as_tuple=True)[1]
        with torch.no_grad():
            pr = F.softmax(m(ids)[0, pos[0]], -1).topk(3)
        print(
            f" {p} -> {[(tok.decode([i]).strip(), float(v) * 100) for v, i in zip(*pr)]}",
            flush=True,
        )

    sp, sa = eval_span(m)
    print(
        f"[span zero-shot] N=8 block-PPL={sp:.1f} acc={sa:.1f}% (Study36 one-shot ref: PPL~133/21%)",
        flush=True,
    )

    # ---- 4. Light contiguous-span tune (500 steps) ----
    m.train()
    opt = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], lr=5e-5, weight_decay=0.01
    )
    sc = torch.amp.GradScaler("cuda")
    t0 = time.time()
    for s in range(1, steps + 1):
        m.train()
        prog = s / steps
        lr = 1e-6 + 0.5 * (5e-5 - 1e-6) * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = lr
        ix = torch.randint(0, len(train) - L, (16,), device=device)
        w = torch.stack([train[i : i + L] for i in ix])
        k = int(torch.randint(2, blk + 1, (1,)).item())  # dynamic k like Study 28
        pm = torch.rand(16, blk, device=device).argsort(-1)[:, :k]
        mk = torch.zeros(16, L, dtype=torch.bool, device=device)
        mk.scatter_(1, pm + seq_pre, True)
        xm = w.clone()
        xm[mk] = MASK
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = F.cross_entropy(m(xm)[mk], w[mk])
        sc.scale(loss).backward()
        sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        sc.step(opt)
        sc.update()
        if s % 100 == 0:
            print(f" [tune] {s}/{steps} loss={loss.item():.3f} lr={lr:.2e}", flush=True)
    sp2, sa2 = eval_span(m)
    print(
        f"[span tuned {time.time() - t0:.0f}s] N=8 block-PPL={sp2:.1f} acc={sa2:.1f}%",
        flush=True,
    )
    torch.save(
        {"state_dict": m.state_dict(), "sp_ppl": sp2, "sp_acc": sa2},
        "/root/checkpoints/subq_bert_12layer_maskgit_span8.pt",
    )
    volume.commit()
    print("saved /root/checkpoints/subq_bert_12layer_maskgit_span8.pt", flush=True)

    # ---- 5. One-shot vs MaskGIT on same prefixes ----
    dec = lambda t: tok.decode(t.tolist()).replace(" ##", "")
    for j in range(3):
        i = int(torch.randint(0, len(val) - L, (1,)).item())
        w = val[i : i + L]
        pre = w[:seq_pre].unsqueeze(0)
        tgt = w[seq_pre:]
        t0 = time.time()
        a = gen_oneshot(m, pre)
        ta = (time.time() - t0) * 1000
        t0 = time.time()
        g = gen_maskgit(m, pre)
        tm = (time.time() - t0) * 1000
        print(f"\n--- span {j} truth: {dec(tgt)!r}", flush=True)
        print(
            f" one-shot {ta:.0f}ms acc={(a == tgt).float().mean().item() * 100:.0f}%: {dec(a)!r}",
            flush=True,
        )
        print(
            f" maskgit  {tm:.0f}ms acc={(g == tgt).float().mean().item() * 100:.0f}%: {dec(g)!r}",
            flush=True,
        )
    return {"verify_ppl": ppl, "span_before": [sp, sa], "span_after": [sp2, sa2]}


@app.local_entrypoint()
def main():
    run_75d.remote()
