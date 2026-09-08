"""Study 75g: needle tuning with curriculum + long jumps (Modal-only, A10G).

75f failed (0->1/16): query->plant needs ~7 chained 64-hops; signal dies.
Fix: (1) K=15 -> K=19 by adding +-128/+-256 (400tok in ~2 hops),
(2) curriculum short->long (96 -> 192 -> 384) so the copy chain builds,
(3) 70% needle / 30% wiki retention. Start from base ckpt (not 75f).

Run: modal run experiments/study75g_needle_curriculum_longjump.py
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "transformers>=4.40.0", "numpy", "requests", "accelerate"
)
app = modal.App("study75g-needle-curriculum")
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)


@app.function(
    image=image, gpu="A10G", timeout=3600, volumes={"/root/checkpoints": volume}
)
def run_75g():
    import math, time, random, requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertForMaskedLM, BertTokenizer

    device = "cuda"
    torch.manual_seed(0)
    random.seed(0)
    print("=" * 110)
    print(" STUDY 75g: CURRICULUM (96->192->384) + LONG JUMPS +-128/+-256 (K=19)")
    print("=" * 110)

    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    MASK, PAD = tok.mask_token_id, tok.pad_token_id
    base = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    dense = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device).eval()

    offsets = [
        -256,
        -128,
        -64,
        -32,
        -16,
        -8,
        -4,
        -2,
        -1,
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    ]
    K = len(offsets)
    print(f"K={K} offsets={offsets}", flush=True)

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
                        ks = F.pad(k[:, :, :-d, :], (0, 0, d, 0))
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
    ck = torch.load(
        "/root/checkpoints/subq_bert_12layer_full_best.pt", map_location=device
    )
    sd = {k: v for k, v in ck["state_dict"].items() if not k.endswith("attn.offsets")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(
        f"loaded base ckpt (strict=False): missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    # re-pin new long offsets (load may have overwritten with K=15 buffer)
    with torch.no_grad():
        for l in m.layers:
            l.attn.offsets.copy_(
                torch.tensor(offsets, dtype=torch.long, device=l.attn.offsets.device)
            )
    print("offsets pinned to K=19", flush=True)

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
    sents = [s.strip() for s in va.split("\n") if 6 < len(s.strip().split()) <= 20]
    fill_ids = [tok.encode(s, add_special_tokens=False) for s in sents]
    fill_ids = [f for f in fill_ids if 8 <= len(f) <= 30]

    words = [
        "crimson",
        "amber",
        "wexford",
        "buxton",
        "marlowe",
        "falcon",
        "harbor",
        "lantern",
        "meadow",
        "orchard",
        "raven",
        "willow",
        "copper",
        "tesla",
        "cobalt",
        "onyx",
    ]
    needles = [(w, tok.encode(w, add_special_tokens=False)) for w in words]
    needles = [(w, i) for w, i in needles if len(i) == 1]
    needle_ids = [i[0] for _, i in needles]

    plant_pre = tok.encode("The explorer's surname is ", add_special_tokens=False)
    plant_end = tok.encode(".", add_special_tokens=False)
    distr_pre = tok.encode("The guide's surname is ", add_special_tokens=False)
    query_pre = tok.encode(
        "At the end of the journey, the explorer's surname is ",
        add_special_tokens=False,
    )
    query_end = tok.encode(".", add_special_tokens=False)
    CLS, SEP = tok.cls_token_id, tok.sep_token_id

    def make_needle(budget, rng):
        n = needle_ids[int(rng.integers(0, len(needle_ids)))]
        d = needle_ids[int(rng.integers(0, len(needle_ids)))]
        use_d = bool(rng.integers(0, 2))
        if d == n:
            use_d = False
        ids = [CLS] + plant_pre + [n] + plant_end
        if use_d:
            ids += distr_pre + [d] + plant_end
        qlen = len(query_pre) + 1 + len(query_end) + 1
        while len(ids) < budget - qlen:
            f = fill_ids[int(rng.integers(0, len(fill_ids)))]
            if len(ids) + len(f) > budget - qlen:
                break
            ids += f
        qpos = len(ids) + len(query_pre)
        ids += query_pre + [MASK] + query_end + [SEP]
        return torch.tensor(ids, dtype=torch.long, device=device), n, qpos

    def needle_batch(bs, budget, rng):
        outs = [make_needle(budget + rng.integers(-16, 17), rng) for _ in range(bs)]
        Lmax = max(o[0].numel() for o in outs)
        inp = torch.full((bs, Lmax), PAD, dtype=torch.long, device=device)
        tgt = torch.zeros(bs, dtype=torch.long, device=device)
        qp = torch.zeros(bs, dtype=torch.long, device=device)
        for b, (ids, n, p) in enumerate(outs):
            inp[b, : ids.numel()] = ids
            tgt[b] = n
            qp[b] = p
        return inp, tgt, qp

    def wiki_batch(bs=8, L=128):
        ix = torch.randint(0, len(train) - L - 1, (bs,), device=device)
        raw = torch.stack([train[i : i + L] for i in ix])
        mk = torch.rand(bs, L, device=device) < 0.15
        inp = raw.clone()
        inp[mk] = MASK
        return inp, raw, mk

    @torch.no_grad()
    def eval_mask_ppl(n=20):
        m.eval()
        tl = tt = 0
        for _ in range(n):
            inp, raw, mk = wiki_batch(8, 128)
            tl += F.cross_entropy(m(inp)[mk], raw[mk]).item() * raw[mk].numel()
            tt += raw[mk].numel()
        return math.exp(tl / tt)

    @torch.no_grad()
    def eval_needles(cases, model):
        hits = 0
        rows = []
        for ids, n, qp in cases:
            lg = model(ids.unsqueeze(0))
            lg = lg.logits if hasattr(lg, "logits") else lg
            pr = F.softmax(lg[0, qp], -1)
            v, i = pr.max(-1)
            ok = int(i) == n
            hits += ok
            rows.append(
                (
                    ok,
                    tok.decode([n]).strip(),
                    tok.decode([int(i)]).strip(),
                    float(v) * 100,
                    ids.numel(),
                )
            )
        return hits, rows

    rng = __import__("numpy").random.default_rng(123)
    eval_cases = []
    for budget in [128, 256, 384, 500]:
        for _ in range(4):
            eval_cases.append(make_needle(min(budget, 500), rng))

    print("[before]:", flush=True)
    for name, model in [("dense", dense), ("subq0", m)]:
        h, _ = eval_needles(eval_cases, model)
        print(f"  {name}: {h}/{len(eval_cases)}", flush=True)
    ppl0 = eval_mask_ppl()
    print(f"[before] mask-PPL={ppl0:.2f}", flush=True)

    # curriculum: (budget, nsteps) short -> long, 70% needle / 30% wiki
    plan = [(96, 300), (192, 300), (320, 400)]
    rng = __import__("numpy").random.default_rng(0)
    opt = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], lr=5e-5, weight_decay=0.01
    )
    sc = torch.amp.GradScaler("cuda")
    t0 = time.time()
    total = sum(n for _, n in plan)
    done = 0
    m.train()
    for budget, nsteps in plan:
        for s in range(1, nsteps + 1):
            done += 1
            prog = done / total
            lr = 1e-6 + 0.5 * (5e-5 - 1e-6) * (1 + math.cos(math.pi * prog))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            if rng.random() < 0.7:
                inp, tgt, qp = needle_batch(8, budget, rng)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = F.cross_entropy(
                        m(inp)[torch.arange(8, device=device), qp], tgt
                    )
                kind = "ndl"
            else:
                inp, raw, mk = wiki_batch(8, 128)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = F.cross_entropy(m(inp)[mk], raw[mk])
                kind = "wiki"
            sc.scale(loss).backward()
            sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            sc.step(opt)
            sc.update()
            if s % 100 == 0 or s == nsteps:
                print(
                    f" [L~{budget}] {s}/{nsteps} ({kind}) loss={loss.item():.3f} lr={lr:.1e}",
                    flush=True,
                )
    print(f"tuned {time.time() - t0:.0f}s", flush=True)

    h, rows = eval_needles(eval_cases, m)
    print(f"[after] subq needles: {h}/{len(rows)}", flush=True)
    for ok, tru, pred, conf, ln in rows:
        print(
            f"  [{'HIT ' if ok else 'miss'}] len={ln} truth={tru!r} pred={pred!r} ({conf:.1f}%)",
            flush=True,
        )
    ppl1 = eval_mask_ppl()
    print(f"[after] mask-PPL={ppl1:.2f} (was {ppl0:.2f})", flush=True)
    torch.save(
        {"state_dict": m.state_dict(), "ppl": ppl1, "hits": h, "offsets": offsets},
        "/root/checkpoints/subq_bert_12layer_needle_k19_curr.pt",
    )
    volume.commit()
    print("saved subq_bert_12layer_needle_k19_curr.pt", flush=True)
    return {"hits": h, "ppl0": ppl0, "ppl1": ppl1}


@app.local_entrypoint()
def main():
    run_75g.remote()
