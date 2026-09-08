"""S2-027: test a bounded-hop-complete offset basis on corrected MQAR.

The baseline dyadic basis has poor exact distance coverage at T=8. This run
keeps the attention-only transport unchanged and compares it with a basis
that covers every distance from 0 through 512 in at most eight jumps.
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-027-complete-offset-basis")


@app.function(image=image, gpu="A10G", timeout=3600)
def train_one(
    model_name: str,
    mode: str,
    offset_name: str,
    offsets: list[int],
    seed: int,
    total_steps: int = 6000,
):
    import math
    import time

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len, batch_size, n_heads = 512, 32, 4
    vocab_size, num_pairs, num_queries = 256, 16, 8
    query_marker = 1
    thought_hops, d_model, d_mlp = 8, 128, 512
    start_q_pos = seq_len - (num_queries * 2) - 2
    query_positions = [start_q_pos + 2 * i + 1 for i in range(num_queries)]

    def make_batch(generator):
        x = torch.randint(100, 255, (batch_size, seq_len), generator=generator, device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        distances = torch.empty((batch_size, num_queries), dtype=torch.long, device=device)
        for row in range(batch_size):
            keys = torch.randperm(40, generator=generator, device=device)[:num_pairs] + 10
            values = torch.randperm(40, generator=generator, device=device)[:num_pairs] + 50
            kv = torch.randperm(350, generator=generator, device=device)[: num_pairs * 2].sort().values
            for pair in range(num_pairs):
                kp = int(kv[2 * pair].item())
                x[row, kp] = keys[pair]
                x[row, kp + 1] = values[pair]
            chosen = torch.randperm(num_pairs, generator=generator, device=device)[:num_queries]
            for qi, kt in enumerate(chosen):
                qpos = start_q_pos + 2 * qi
                kp = int(kv[2 * int(kt.item())].item())
                x[row, qpos] = query_marker
                x[row, qpos + 1] = keys[kt]
                y[row, qpos + 1] = values[kt]
                distances[row, qi] = qpos + 1 - kp
        return x, y, distances

    class CompleteBasisSubQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            self.ln = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            if mode == "attention_only_final_mlp":
                self.ln_mlp = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_mlp),
                    nn.GELU(),
                    nn.Linear(d_mlp, d_model),
                )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            targets = torch.zeros((seq_len, len(offsets)), dtype=torch.long)
            valid = torch.zeros((seq_len, len(offsets)), dtype=torch.bool)
            for pos in range(seq_len):
                for idx, offset in enumerate(offsets):
                    source = pos - offset
                    if source >= 0:
                        targets[pos, idx] = source
                        valid[pos, idx] = True
            self.register_buffer("targets", targets)
            self.register_buffer("valid", valid)

        def forward(self, idx):
            bsz, length = idx.shape
            state = self.tok(idx) + self.pos(torch.arange(length, device=idx.device).unsqueeze(0))
            indices = self.targets[:length]
            valid = self.valid[:length].view(1, 1, length, len(offsets))
            scale = 1.0 / math.sqrt(d_model // n_heads)
            for _ in range(thought_hops):
                z = self.ln(state)
                q = self.q(z).view(bsz, length, n_heads, -1).transpose(1, 2)
                k = self.k(z).view(bsz, length, n_heads, -1).transpose(1, 2)
                v = self.v(z).view(bsz, length, n_heads, -1).transpose(1, 2)
                scores = (q.unsqueeze(3) * k[:, :, indices, :]).sum(-1) * scale
                weights = F.softmax(scores.masked_fill(~valid, -1e4), dim=-1)
                context = (weights.unsqueeze(-1) * v[:, :, indices, :]).sum(3)
                state = context.transpose(1, 2).contiguous().view(bsz, length, d_model)
            if mode == "attention_only_final_mlp":
                state = state + self.mlp(self.ln_mlp(state))
            return self.head(self.ln_f(state))

    model = CompleteBasisSubQ().to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    start = time.time()
    for _ in range(total_steps):
        model.train()
        x, y, _ = make_batch(generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            loss = F.cross_entropy(
                model(x).reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    model.eval()
    correct = total = 0
    bins = {"128-255": [128, 256], "256-383": [256, 384], "384-511": [384, 512]}
    stats = {name: [0, 0] for name in bins}
    eval_generator = torch.Generator(device=device).manual_seed(seed + 900000)
    with torch.no_grad():
        for _ in range(50):
            x, y, distances = make_batch(eval_generator)
            logits = model(x)
            hits = logits[:, query_positions, :].argmax(-1).eq(y[:, query_positions])
            correct += hits.sum().item()
            total += hits.numel()
            for name, (lo, hi) in bins.items():
                mask = (distances >= lo) & (distances < hi)
                stats[name][0] += (hits & mask).sum().item()
                stats[name][1] += mask.sum().item()
    return {
        "experiment": "S2-027",
        "model": model_name,
        "mode": mode,
        "offset_name": offset_name,
        "offsets": offsets,
        "seed": seed,
        "d_model": d_model,
        "d_mlp": d_mlp,
        "thought_hops": thought_hops,
        "parameters": parameters,
        "steps": total_steps,
        "accuracy": round(100.0 * correct / total, 4),
        "distance_accuracy": {
            name: round(100.0 * hits / count, 4) for name, (hits, count) in stats.items()
        },
        "queries": total,
        "elapsed_s": round(time.time() - start, 3),
    }


@app.local_entrypoint()
def main():
    import pathlib
    import time

    seeds = [42, 1337, 2026]
    offset_sets = {
        "dyadic_k8": [0, 1, 2, 4, 8, 16, 32, 64],
        "complete_k9": [0, 1, 2, 4, 8, 16, 63, 127, 128],
    }
    modes = ["attention_only", "attention_only_final_mlp"]
    configs = [
        (f"{offset_name}_{mode}_s{seed}", mode, offset_name, offsets, seed)
        for offset_name, offsets in offset_sets.items()
        for mode in modes
        for seed in seeds
    ]
    calls = [
        train_one.spawn(name, mode, offset_name, offsets, seed)
        for name, mode, offset_name, offsets, seed in configs
    ]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    payload = {
        "experiment": "S2-027",
        "description": "Attention-only SubQ with old dyadic versus bounded-hop-complete offsets on corrected MQAR",
        "task": "16 independently randomized key/value pairs, 8 late queries",
        "random_exact_baseline_percent": 2.5,
        "thought_hops": 8,
        "offset_sets": offset_sets,
        "seeds": seeds,
        "training_steps": 6000,
        "conditions": configs,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    path = pathlib.Path("season2/results/s2_027_complete_offset_basis.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}", flush=True)
