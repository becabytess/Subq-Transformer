"""Study 75h: generalization probe — unseen relations/words (Modal-only, A10G, eval-only).

75g hit 16/16 on the TRAIN template ("explorer's surname"). This tests whether
a real copy circuit was installed or just template-fit:
  - new relations never tuned on (lighthouse/bird, museum/gallery, nickname, flag...)
  - held-out needle words (quartz, bison, topaz, ... single-token, unseen in 75g)
  - lengths 128/256/384, with/without distractor
Models: dense baseline vs base K15 ckpt vs tuned K19 ckpt.

Run: modal run experiments/study75h_needle_generalization.py
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "transformers>=4.40.0", "numpy", "requests", "accelerate"
)
app = modal.App("study75h-needle-generalization")
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)


@app.function(
    image=image, gpu="A10G", timeout=1800, volumes={"/root/checkpoints": volume}
)
def run_75h():
    import requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertForMaskedLM, BertTokenizer
    import math

    device = "cuda"
    print("=" * 110)
    print(" STUDY 75h: UNSEEN-RELATION GENERALIZATION (dense vs base-K15 vs tuned-K19)")
    print("=" * 110)

    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    MASK = tok.mask_token_id
    dense = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device).eval()
    base_hf = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)

    def build_arch(offsets, hf):
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
                                    torch.zeros(
                                        B, 1, d, dtype=torch.bool, device=x.device
                                    ),
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
                                    torch.zeros(
                                        B, 1, a, dtype=torch.bool, device=x.device
                                    ),
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

        return Net(hf).to(device)

    OFF15 = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]
    OFF19 = [
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
    subq0 = build_arch(OFF15, base_hf).to(device)
    ck0 = torch.load(
        "/root/checkpoints/subq_bert_12layer_full_best.pt", map_location=device
    )
    subq0.load_state_dict(ck0["state_dict"], strict=True)
    subq0.eval()
    base_hf2 = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    subqT = build_arch(OFF19, base_hf2).to(device)
    ckT = torch.load(
        "/root/checkpoints/subq_bert_12layer_needle_k19_curr.pt", map_location=device
    )
    sd = {k: v for k, v in ckT["state_dict"].items() if not k.endswith("attn.offsets")}
    subqT.load_state_dict(sd, strict=False)
    with torch.no_grad():
        for l in subqT.layers:
            l.attn.offsets.copy_(torch.tensor(OFF19, dtype=torch.long, device="cuda"))
    subqT.eval()
    print("models ready: dense / base-K15 / tuned-K19", flush=True)

    # unseen relations + held-out words
    held = ["quartz", "bison", "topaz", "garnet", "walrus", "cedar", "beacon", "harbor"]
    held = [(w, tok.encode(w, add_special_tokens=False)) for w in held]
    held = [(w, i) for w, i in held if len(i) == 1]
    print(f"held-out needles: {[w for w, _ in held]}", flush=True)
    RELS = [
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
    ]
    wt = requests.get(
        "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt",
        timeout=60,
    ).text.split("\n")
    sents = [s.strip() for s in wt if 6 < len(s.strip().split()) <= 20]
    fill = [tok.encode(s, add_special_tokens=False) for s in sents]
    fill = [f for f in fill if 8 <= len(f) <= 30]
    CLS, SEP = tok.cls_token_id, tok.sep_token_id
    rng = __import__("numpy").random.default_rng(7)

    def make_case(budget):
        w, ids1 = held[int(rng.integers(0, len(held)))]
        n = ids1[0]
        r = int(rng.integers(0, len(RELS)))
        p0, e0, p1, e1 = RELS[r]
        a0 = tok.encode(p0, add_special_tokens=False)
        b0 = tok.encode(e0, add_special_tokens=False)
        a1 = tok.encode(p1, add_special_tokens=False)
        b1 = tok.encode(e1, add_special_tokens=False)
        # distractor: different relation + different word
        r2 = (r + 1) % len(RELS)
        w2, ids2 = held[int(rng.integers(0, len(held)))]
        use_d = bool(rng.integers(0, 2))
        if ids2[0] == n:
            use_d = False
        ids = [CLS] + a0 + [n] + b0
        if use_d:
            q0 = tok.encode(RELS[r2][0], add_special_tokens=False)
            q1 = tok.encode(RELS[r2][1], add_special_tokens=False)
            ids += q0 + [ids2[0]] + q1
        qlen = len(a1) + 1 + len(b1) + 1
        fi = 0
        while len(ids) < budget - qlen and fi < 2000:
            f = fill[int(rng.integers(0, len(fill)))]
            fi += 1
            if len(ids) + len(f) > budget - qlen:
                break
            ids += f
        qp = len(ids) + len(a1)
        ids += a1 + [MASK] + b1 + [SEP]
        return (
            torch.tensor(ids, dtype=torch.long, device=device),
            n,
            qp,
            (w, r, use_d, len(ids)),
        )

    @torch.no_grad()
    def ask(model, ids, qp):
        lg = model(ids.unsqueeze(0))
        lg = lg.logits if hasattr(lg, "logits") else lg
        pr = F.softmax(lg[0, qp], -1)
        v, i = pr.max(-1)
        return int(i), float(v) * 100

    for budget in [128, 256, 384]:
        print(f"\n--- unseen relations, budget~{budget} ---", flush=True)
        score = {"dense": 0, "base": 0, "tuned": 0}
        tot = 0
        for t in range(6):
            ids, n, qp, (w, r, ud, ln) = make_case(budget)
            tru = tok.decode([n]).strip()
            line = f" rel={r} w={w} distr={ud} len={ln} truth={tru!r}: "
            outs = []
            for name, mdl in [("dense", dense), ("base", subq0), ("tuned", subqT)]:
                i, c = ask(mdl, ids, qp)
                ok = i == n
                score[name] += ok
                outs.append(
                    f"{name}={'HIT' if ok else 'miss'}({tok.decode([i]).strip()},{c:.0f}%)"
                )
            tot += 1
            print("  " + line + " | ".join(outs), flush=True)
        print(
            f"  => dense {score['dense']}/{tot} | base {score['base']}/{tot} | tuned {score['tuned']}/{tot}",
            flush=True,
        )
    return True


@app.local_entrypoint()
def main():
    run_75h.remote()
