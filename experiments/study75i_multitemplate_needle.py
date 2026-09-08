"""Study 75i: multi-template needle tuning (Modal-only, A10G).

75h proved single-template = template-fit (2/18 unseen). Fix: 10 diverse
relations, paraphrased query (not string-match), train words != held words,
leave-one-relation-out eval. K=19 long jumps + curriculum retained.

Train: relations 0..8 x train-words(16). Eval: (a) sanity train-rel/train-word,
(b) TRUE zero-shot: held relation #9 x held-out words (never seen).

Run: modal run experiments/study75i_multitemplate_needle.py
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "transformers>=4.40.0", "numpy", "requests", "accelerate"
)
app = modal.App("study75i-multitemplate")
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)


@app.function(
    image=image, gpu="A10G", timeout=3600, volumes={"/root/checkpoints": volume}
)
def run_75i():
    import math, time, random, requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertForMaskedLM, BertTokenizer

    device = "cuda"
    torch.manual_seed(0)
    random.seed(0)
    print("=" * 110)
    print(" STUDY 75i: MULTI-TEMPLATE COPY (10 rels, held words, LORO relation)")
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

    class A(nn.Module):
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
            return self.out_proj(
                (torch.nan_to_num(F.softmax(s, -1), nan=0.0).unsqueeze(-1) * Vc)
                .sum(3)
                .transpose(1, 2)
                .reshape(B, L, 768)
            )

    class Ly(nn.Module):
        def __init__(self, o):
            super().__init__()
            self.attn = A(o.attention)
            self.ln1 = o.attention.output.LayerNorm
            self.intermediate = o.intermediate
            self.output = o.output

        def forward(self, h):
            h = self.ln1(h + self.attn(h))
            return self.output(self.intermediate(h), h)

    class Net(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.embeddings = b.bert.embeddings
            self.layers = nn.ModuleList(
                [Ly(b.bert.encoder.layer[i]) for i in range(12)]
            )
            self.cls = b.cls

        def forward(self, idx):
            h = self.embeddings(idx)
            for l in self.layers:
                h = l(h)
            return self.cls(h)

    m = Net(base).to(device)
    ck = torch.load(
        "/root/checkpoints/subq_bert_12layer_full_best.pt", map_location=device
    )
    sd = {k: v for k, v in ck["state_dict"].items() if not k.endswith("attn.offsets")}
    m.load_state_dict(sd, strict=False)
    with torch.no_grad():
        for l in m.layers:
            l.attn.offsets.copy_(torch.tensor(offsets, dtype=torch.long, device="cuda"))
    print("loaded base + K19 pinned", flush=True)

    # 10 relations (plant, plant_end, query, query_end) — paraphrased query
    RELS = [
        (
            "The explorer's surname is ",
            ".",
            "At the end of the journey, the explorer's surname is ",
            ".",
        ),
        (
            "The vault keeper's favorite animal is ",
            ".",
            "When asked at dusk, the keeper named the animal ",
            ".",
        ),
        (
            "The chef's secret spice is ",
            ".",
            "Critics finally guessed the secret spice was ",
            ".",
        ),
        (
            "The pilot's callsign was ",
            ".",
            "Tower logs show the pilot's callsign was ",
            ".",
        ),
        (
            "The gardener's prized flower is ",
            ".",
            "In the autumn catalog, the prized flower is ",
            ".",
        ),
        (
            "The lighthouse keeper's favorite bird is ",
            ".",
            "At the lonely cliff, the keeper's favorite bird is ",
            ".",
        ),
        (
            "The museum's hidden gallery holds ",
            ".",
            "Years later, the hidden gallery holds ",
            ".",
        ),
        (
            "Her childhood nickname was ",
            ".",
            "Nobody remembers, but her nickname was ",
            ".",
        ),
        (
            "The captain's flag was ",
            ".",
            "Through the storm, the captain's flag was ",
            ".",
        ),
        (
            "The clockmaker's prized metal is ",
            ".",
            "The appraisal lists the prized metal as ",
            ".",
        ),  # HELD-OUT #9
    ]
    TRAIN_W = [
        "crimson",
        "amber",
        "wexford",
        "buxton",
        "marlowe",
        "falcon",
        "lantern",
        "meadow",
        "orchard",
        "raven",
        "willow",
        "copper",
        "tesla",
        "cobalt",
        "onyx",
        "harbor",
    ]
    HELD_C = [
        "quartz",
        "bison",
        "topaz",
        "garnet",
        "walrus",
        "cedar",
        "beacon",
        "ember",
        "flint",
        "grove",
        "heron",
        "ivory",
    ]

    def single(w):
        e = tok.encode(w, add_special_tokens=False)
        return e[0] if len(e) == 1 else None

    train_ids = [single(w) for w in TRAIN_W]
    train_ids = [i for i in train_ids if i is not None]
    held_ids = [single(w) for w in HELD_C]
    held_ids = [i for i in held_ids if i is not None and i not in train_ids]
    print(f"train words={len(train_ids)} held words={len(held_ids)}", flush=True)
    assert len(held_ids) >= 4, "need >=4 held single-token words"

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
    sents = [s.strip() for s in va.split("\n") if 6 < len(s.strip().split()) <= 20]
    fill = [tok.encode(s, add_special_tokens=False) for s in sents]
    fill = [f for f in fill if 8 <= len(f) <= 30]
    CLS, SEP = tok.cls_token_id, tok.sep_token_id
    EREL = {
        r: (
            tok.encode(a, add_special_tokens=False),
            tok.encode(b, add_special_tokens=False),
            tok.encode(c, add_special_tokens=False),
            tok.encode(d, add_special_tokens=False),
        )
        for r, (a, b, c, d) in enumerate(RELS)
    }

    def make_case(budget, rng, rels, words):
        r = int(rels[int(rng.integers(0, len(rels)))])
        a0, b0, a1, b1 = EREL[r]
        n = words[int(rng.integers(0, len(words)))]
        # distractor: different relation + different word
        r2 = int(rng.integers(0, len(RELS) - 1))
        if r2 >= 9:
            r2 = 8  # keep distractor inside train rels (never held #9)
        if r2 == r:
            r2 = (r + 1) % 9
        c0, d0, _, _ = EREL[r2]
        dd = words[int(rng.integers(0, len(words)))]
        use_d = bool(rng.integers(0, 2))
        if dd == n:
            use_d = False
        ids = [CLS] + a0 + [n] + b0
        if use_d:
            ids += c0 + [dd] + d0
        qlen = len(a1) + 1 + len(b1) + 1
        while len(ids) < budget - qlen:
            f = fill[int(rng.integers(0, len(fill)))]
            if len(ids) + len(f) > budget - qlen:
                break
            ids += f
        qp = len(ids) + len(a1)
        ids += a1 + [MASK] + b1 + [SEP]
        return torch.tensor(ids, dtype=torch.long, device=device), n, qp, (r, len(ids))

    def nbatch(bs, budget, rng, rels, words):
        outs = [
            make_case(budget + rng.integers(-16, 17), rng, rels, words)
            for _ in range(bs)
        ]
        Lm = max(o[0].numel() for o in outs)
        inp = torch.full((bs, Lm), tok.pad_token_id, dtype=torch.long, device=device)
        tgt = torch.zeros(bs, dtype=torch.long, device=device)
        qp = torch.zeros(bs, dtype=torch.long, device=device)
        for b, (ids, n, p, _) in enumerate(outs):
            inp[b, : ids.numel()] = ids
            tgt[b] = n
            qp[b] = p
        return inp, tgt, qp

    def wbatch(bs=8, L=128):
        ix = torch.randint(0, len(train) - L - 1, (bs,), device=device)
        raw = torch.stack([train[i : i + L] for i in ix])
        mk = torch.rand(bs, L, device=device) < 0.15
        inp = raw.clone()
        inp[mk] = MASK
        return inp, raw, mk

    @torch.no_grad()
    def ask(model, ids, qp):
        lg = model(ids.unsqueeze(0))
        lg = lg.logits if hasattr(lg, "logits") else lg
        pr = F.softmax(lg[0, qp], -1)
        v, i = pr.max(-1)
        return int(i), float(v) * 100

    @torch.no_grad()
    def ppl(n=16):
        m.eval()
        tl = tt = 0
        for _ in range(n):
            inp, raw, mk = wbatch()
            tl += F.cross_entropy(m(inp)[mk], raw[mk]).item() * raw[mk].numel()
            tt += raw[mk].numel()
        return math.exp(tl / tt)

    rngE = __import__("numpy").random.default_rng(99)
    sanity = [
        make_case(b, rngE, list(range(9)), train_ids)
        for b in [128, 256, 384]
        for _ in range(2)
    ]
    zero = [
        make_case(b, rngE, [9], held_ids) for b in [128, 256, 384] for _ in range(4)
    ]
    print(
        f"eval: sanity={len(sanity)} zero-shot(held rel+held words)={len(zero)}",
        flush=True,
    )
    for name, mdl in [("dense", dense), ("base", m)]:
        hs = sum(ask(mdl, i, p)[0] == n for i, n, p, _ in sanity)
        hz = sum(ask(mdl, i, p)[0] == n for i, n, p, _ in zero)
        print(
            f" [before] {name}: sanity {hs}/{len(sanity)} zero {hz}/{len(zero)}",
            flush=True,
        )
    p0 = ppl()
    print(f" [before] mask-PPL={p0:.2f}", flush=True)

    plan = [(96, 400), (192, 400), (320, 400)]
    rng = __import__("numpy").random.default_rng(0)
    opt = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], lr=5e-5, weight_decay=0.01
    )
    sc = torch.amp.GradScaler("cuda")
    t0 = time.time()
    total = sum(n for _, n in plan)
    done = 0
    m.train()
    for budget, ns in plan:
        for s in range(1, ns + 1):
            done += 1
            prog = done / total
            lr = 1e-6 + 0.5 * (5e-5 - 1e-6) * (1 + math.cos(math.pi * prog))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            if rng.random() < 0.7:
                inp, tgt, qp = nbatch(8, budget, rng, list(range(9)), train_ids)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = F.cross_entropy(
                        m(inp)[torch.arange(8, device=device), qp], tgt
                    )
                k = "ndl"
            else:
                inp, raw, mk = wbatch()
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = F.cross_entropy(m(inp)[mk], raw[mk])
                k = "wiki"
            sc.scale(loss).backward()
            sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            sc.step(opt)
            sc.update()
            if s % 100 == 0 or s == ns:
                print(
                    f" [L~{budget}] {s}/{ns} ({k}) loss={loss.item():.3f}", flush=True
                )
    print(f"tuned {time.time() - t0:.0f}s", flush=True)
    hs = sum(ask(m, i, p)[0] == n for i, n, p, _ in sanity)
    rows = []
    for ids, n, qp, (r, ln) in zero:
        i, c = ask(m, ids, qp)
        ok = i == n
        rows.append((ok, tok.decode([n]).strip(), tok.decode([i]).strip(), c, ln, r))
    hz = sum(r[0] for r in rows)
    print(
        f"[after] sanity {hs}/{len(sanity)} | ZERO-SHOT held-rel+held-words {hz}/{len(rows)}",
        flush=True,
    )
    for ok, tru, pred, c, ln, r in rows:
        print(
            f"  [{'HIT ' if ok else 'miss'}] len={ln} rel={r} truth={tru!r} pred={pred!r} ({c:.0f}%)",
            flush=True,
        )
    p1 = ppl()
    print(f"[after] mask-PPL={p1:.2f} (was {p0:.2f})", flush=True)
    torch.save(
        {"state_dict": m.state_dict(), "ppl": p1, "zero": hz},
        "/root/checkpoints/subq_bert_12layer_needle_multi.pt",
    )
    volume.commit()
    print("saved subq_bert_12layer_needle_multi.pt", flush=True)
    return {"zero": hz, "sanity": hs}


@app.local_entrypoint()
def main():
    run_75i.remote()
