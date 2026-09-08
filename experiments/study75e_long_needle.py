"""Study 75e: Long-needle probe — can 12L SubQ-BERT (K=15 sparse) recall a fact
from ~300 tokens back as well as dense BERT? (Modal-only, A10G, eval-only, no training)

Design: plant single-token needle early (e.g. surname/color), fill with real
WikiText-2 sentences, query late with [MASK]. Short prompts fit in 1 hop;
this forces multi-hop routing (offsets max +-64, so 300tok needs ~5 hops).

Compares: dense bert-base-uncased vs /root/checkpoints/subq_bert_12layer_full_best.pt
At L=128 (in-train) and L=256/512 (extrapolation; BERT pos cap = 512).

Run: modal run experiments/study75e_long_needle.py
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "transformers>=4.40.0", "numpy", "requests", "accelerate"
)
app = modal.App("study75e-long-needle")
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)


@app.function(
    image=image, gpu="A10G", timeout=1800, volumes={"/root/checkpoints": volume}
)
def run_75e():
    import math, os, requests, random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertForMaskedLM, BertTokenizer

    device = "cuda"
    torch.manual_seed(0)
    random.seed(0)
    print("=" * 110)
    print(" STUDY 75e: LONG-NEEDLE DISTANT RECALL (dense vs SubQ K=15 sparse)")
    print("=" * 110)

    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    MASK = tok.mask_token_id
    dense = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device).eval()

    # rebuild SubQ arch (same names as Study 35 ckpt)
    d_model = 768
    offsets = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]
    base = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)

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

    subq = Full12LayerSubQBert(base).to(device)
    ck = torch.load(
        "/root/checkpoints/subq_bert_12layer_full_best.pt", map_location=device
    )
    subq.load_state_dict(ck["state_dict"], strict=True)
    subq.eval()
    print(f"subq loaded (Top-1={ck.get('top1', 0):.1f}%)", flush=True)

    # ---- needle words guaranteed single-token ----
    cands = [
        "crimson",
        "amber",
        "wexford",
        "buxton",
        "marlowe",
        "quilp",
        "tavistock",
        "corvino",
        "falcon",
        "harbor",
        "lantern",
        "meadow",
        "orchard",
        "raven",
        "willow",
        "copper",
    ]
    single = [w for w in cands if len(tok.encode(w, add_special_tokens=False)) == 1]
    print(f"single-token needles: {single}", flush=True)

    wt = requests.get(
        "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt",
        timeout=60,
    ).text.split("\n")
    filler = [s.strip() for s in wt if 6 < len(s.strip().split()) <= 20]

    def build(needle, budget, distractor=None):
        # plant early, pad with short WikiText sents to ~budget tokens, query late.
        # keeps everything <=512 (BERT pos cap); distance plant->query ~= budget.
        plant = f"The explorer's surname is {needle}."
        query_s = "At the end of the journey, the explorer's surname is [MASK]."
        overhead = len(tok.encode(plant + " " + query_s, add_special_tokens=True))
        parts = [plant]
        if distractor:
            d = f"The guide's surname is {distractor}."
            parts.append(d)
            overhead += len(tok.encode(d, add_special_tokens=False))
        fi = 0
        cur = overhead
        while cur < budget - 20 and fi < len(filler):
            s = filler[fi]
            fi += 1
            if distractor and fi <= 2:
                continue  # already used first two above
            t = len(tok.encode(" " + s, add_special_tokens=False))
            if cur + t > budget - 20:
                break
            parts.append(s)
            cur += t
        parts.append(query_s)
        txt = " ".join(parts)
        ids = tok.encode(txt, add_special_tokens=True)
        return txt, ids

    @torch.no_grad()
    def query(model, ids):
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        pos = (inp == MASK).nonzero(as_tuple=True)[1]
        assert len(pos) == 1
        lg = model(inp)
        lg = lg.logits if hasattr(lg, "logits") else lg
        return F.softmax(lg[0, pos[0]], -1)

    for budget in [128, 256, 384, 500]:
        print(f"\n--- budget~{budget}tok ---", flush=True)
        for trial in range(3):
            needle = single[trial % len(single)]
            distr = single[(trial + 3) % len(single)]
            for use_distr in [False, True]:
                txt, ids = build(needle, budget, distr if use_distr else None)
                if len(ids) > 512:  # BERT pos cap
                    print(f"  skip (len={len(ids)}>512)")
                    continue
                truth = tok.encode(needle, add_special_tokens=False)[0]
                tag = f"needle={needle} distr={'-' if not use_distr else distr} len={len(ids)}"
                for name, model in [("dense", dense), ("subq", subq)]:
                    pr = query(model, ids)
                    top5 = pr.topk(5)
                    words = [
                        (tok.decode([i]).strip(), float(v) * 100) for v, i in zip(*top5)
                    ]
                    hit = "HIT" if top5.indices[0] == truth else "miss"
                    print(
                        f"  [{hit:4s}] {name:5s} {tag}: {words[0]} | top5={words}",
                        flush=True,
                    )

    # scattered-mask sanity on one long passage (mask 5 random + needle)
    print("\n--- scattered masks on long passage ---", flush=True)
    txt, ids = build(single[0], 400, single[3])
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    cands_pos = [
        i for i in range(5, len(ids) - 5) if ids[i] not in (MASK, tok.sep_token_id)
    ]
    for p in random.sample(cands_pos, 5):
        truth = tok.decode([ids[p]]).strip()
        masked = ids_t.clone()
        masked[0, p] = MASK
        for name, model in [("dense", dense), ("subq", subq)]:
            with torch.no_grad():
                lg = model(masked)
                lg = lg.logits if hasattr(lg, "logits") else lg
                pr = F.softmax(lg[0, p], -1)
                v, i = pr.max(-1)
                ok = "HIT" if i == ids[p] else "miss"
                print(
                    f"  [{ok:4s}] {name:5s} pos={p} truth={truth!r} pred={tok.decode([int(i)]).strip()!r} ({float(v) * 100:.1f}%)",
                    flush=True,
                )
    return True


@app.local_entrypoint()
def main():
    run_75e.remote()
