"""Study 75j: pointer/copy head over frozen K19-tuned backbone (Modal-only, A10G).

75i proved the MLM head can't do open-vocab copy (0/12 zero-shot, predicts
train words). Hypothesis: backbone query-state carries *that* a needle exists
but the vocab classifier can't bind identity. A pointer head (predict POSITION
of the plant, then read its token) bypasses the vocab bottleneck entirely.

v1: freeze tuned backbone, train only a light pointer (query-proj + key-proj,
dot over input positions), multi-template train rels/words, eval zero-shot
held-rel + held-words. 300 steps, minutes.

Run: modal run experiments/study75j_pointer_copy.py
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "transformers>=4.40.0", "numpy", "requests", "accelerate"
)
app = modal.App("study75j-pointer")
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)


@app.function(
    image=image, gpu="A10G", timeout=3600, volumes={"/root/checkpoints": volume}
)
def run_75j(steps: int = 400):
    import math, time, random, requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertForMaskedLM, BertTokenizer

    device = "cuda"
    torch.manual_seed(0)
    random.seed(0)
    print("=" * 110)
    print(" STUDY 75j: POINTER HEAD (frozen backbone, position prediction)")
    print("=" * 110)

    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    MASK, PAD = tok.mask_token_id, tok.pad_token_id
    base = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
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

        def hidden(self, idx):
            h = self.embeddings(idx)
            for l in self.layers:
                h = l(h)
            return h

    m = Net(base).to(device)
    ck = torch.load(
        "/root/checkpoints/subq_bert_12layer_needle_k19_curr.pt", map_location=device
    )
    sd = {k: v for k, v in ck["state_dict"].items() if not k.endswith("attn.offsets")}
    m.load_state_dict(sd, strict=False)
    with torch.no_grad():
        for l in m.layers:
            l.attn.offsets.copy_(torch.tensor(offsets, dtype=torch.long, device="cuda"))
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    print("frozen K19-tuned backbone loaded", flush=True)

    class Pointer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(768, 256)
            self.k = nn.Linear(768, 256)

        def forward(self, h, qp):
            # h: B,L,768 states; qp: B query positions -> scores over L
            qv = self.q(h[torch.arange(h.size(0), device=h.device), qp])  # B,256
            kv = self.k(h)  # B,L,256
            return (kv * qv.unsqueeze(1)).sum(-1)  # B,L

    ptr = Pointer().to(device)
    opt = torch.optim.AdamW(ptr.parameters(), lr=3e-4)

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
        ),
    ]
    EREL = {
        r: (
            tok.encode(a, add_special_tokens=False),
            tok.encode(b, add_special_tokens=False),
            tok.encode(c, add_special_tokens=False),
            tok.encode(d, add_special_tokens=False),
        )
        for r, (a, b, c, d) in enumerate(RELS)
    }
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
    ]
    HELD_W = ["quartz", "bison", "topaz", "garnet", "cedar", "beacon", "ember", "flint"]

    def single(w):
        e = tok.encode(w, add_special_tokens=False)
        return e[0] if len(e) == 1 else None

    TR = [s for s in (single(w) for w in TRAIN_W) if s is not None]
    HE = [s for s in (single(w) for w in HELD_W) if s is not None and s not in TR]
    va = requests.get(
        "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt",
        timeout=60,
    ).text
    sents = [s.strip() for s in va.split("\n") if 6 < len(s.strip().split()) <= 20]
    fill = [tok.encode(s, add_special_tokens=False) for s in sents]
    fill = [f for f in fill if 8 <= len(f) <= 30]
    CLS, SEP = tok.cls_token_id, tok.sep_token_id
    rng = __import__("numpy").random.default_rng(0)

    def make(budget, rels, words):
        r = int(rels[int(rng.integers(0, len(rels)))])
        a0, b0, a1, b1 = EREL[r]
        n = words[int(rng.integers(0, len(words)))]
        ids = [CLS] + a0 + [n] + b0
        plant_pos = len(a0) + 1  # index of n inside (after CLS)
        qlen = len(a1) + 1 + len(b1) + 1
        while len(ids) < budget - qlen:
            f = fill[int(rng.integers(0, len(fill)))]
            if len(ids) + len(f) > budget - qlen:
                break
            ids += f
        qp = len(ids) + len(a1)
        ids += a1 + [MASK] + b1 + [SEP]
        return torch.tensor(ids, dtype=torch.long, device=device), qp, plant_pos, n

    def batch(bs, budget, rels, words):
        outs = [make(budget + rng.integers(-16, 17), rels, words) for _ in range(bs)]
        Lm = max(o[0].numel() for o in outs)
        inp = torch.full((bs, Lm), PAD, dtype=torch.long, device=device)
        qp = torch.zeros(bs, dtype=torch.long, device=device)
        pp = torch.zeros(bs, dtype=torch.long, device=device)
        for b, (ids, q, p, n) in enumerate(outs):
            inp[b, : ids.numel()] = ids
            qp[b] = q
            pp[b] = p
        return inp, qp, pp

    # train pointer on train rels/words, mixed lengths
    t0 = time.time()
    ptr.train()
    for s in range(1, steps + 1):
        budget = [96, 192, 320][int(rng.integers(0, 3))]
        inp, qp, pp = batch(8, budget, list(range(9)), TR)
        opt.zero_grad()
        with torch.no_grad():
            h = m.hidden(inp)
        scores = ptr(h, qp)
        scores = scores.masked_fill(
            torch.arange(scores.size(1), device=device)[None, :]
            >= (inp != PAD).sum(1)[:, None],
            float("-inf"),
        )
        loss = F.cross_entropy(scores, pp)
        loss.backward()
        opt.step()
        if s % 50 == 0 or s == steps:
            print(f" [ptr] {s}/{steps} loss={loss.item():.3f}", flush=True)
    print(f"pointer trained {time.time() - t0:.0f}s", flush=True)

    # eval: zero-shot held rel #9 + held words; prediction = token at pointed position
    @torch.no_grad()
    def evalset(cases):
        hits = 0
        rows = []
        for ids, qp, pp, n in cases:
            h = m.hidden(ids.unsqueeze(0))
            s = ptr(h, torch.tensor([qp], device=device))
            pred_pos = int(s[0, : ids.numel()].argmax())
            pred_tok = int(ids[pred_pos])
            ok = pred_tok == n
            hits += ok
            rows.append(
                (
                    ok,
                    tok.decode([n]).strip(),
                    tok.decode([pred_tok]).strip(),
                    pred_pos,
                    pp,
                    ids.numel(),
                )
            )
        return hits, rows

    rngE = __import__("numpy").random.default_rng(5)
    saved = rng.bit_generator.state
    zero = []
    for b in [128, 256, 384]:
        for _ in range(4):
            ids, qp, pp, n = make(b, [9], HE)
            zero.append((ids, qp, pp, n))
    h, rows = evalset(zero)
    print(f"[pointer] ZERO-SHOT held-rel+held-words: {h}/{len(rows)}", flush=True)
    for ok, tru, pred, pp, plant, ln in rows:
        print(
            f"  [{'HIT ' if ok else 'miss'}] len={ln} plant@{plant} pointed@{pp} truth={tru!r} read={pred!r}",
            flush=True,
        )
    torch.save({"pointer": ptr.state_dict()}, "/root/checkpoints/subq_pointer_k19.pt")
    volume.commit()
    print("saved subq_pointer_k19.pt", flush=True)
    return {"hits": h}


@app.local_entrypoint()
def main():
    run_75j.remote()
