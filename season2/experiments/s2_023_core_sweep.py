"""S2-023: core SubQ sweep without oracle or custom routing controls.

Compares residual, accumulator-GRU, transformed replacement, and attention-only
updates on corrected full MQAR. The second pair of conditions increases model
width at fixed hop count to test parameter scaling without adding depth.
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install("torch>=2.2.0", "numpy")
app = modal.App("season2-s2-023-core-sweep")


@app.function(image=image, gpu="A10G", timeout=3600)
def train_one(model_name: str, mode: str, seed: int, d_model: int, d_mlp: int, thought_hops: int, total_steps: int = 6000):
    import math
    import os
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

    seq_len, batch_size, n_heads = 512, 16 if d_model >= 256 else 32, 4
    vocab_size, num_pairs, num_queries = 256, 16, 8
    query_marker = 1
    offsets = [0, 1, 2, 4, 8, 16, 32, 64]
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

    class CoreSubQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            self.ln = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            if mode not in {"attention_only", "attention_only_final_mlp"}:
                self.o = nn.Linear(d_model, d_model, bias=False)
            if mode == "gru":
                self.gru = nn.GRUCell(d_model, d_model)
            if mode != "attention_only":
                self.ln_mlp = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            targets = torch.zeros((seq_len, len(offsets)), dtype=torch.long)
            valid = torch.zeros((seq_len, len(offsets)), dtype=torch.bool)
            for pos in range(seq_len):
                for idx, off in enumerate(offsets):
                    src = pos - off
                    if src >= 0:
                        targets[pos, idx] = src
                        valid[pos, idx] = True
            self.register_buffer("targets", targets)
            self.register_buffer("valid", valid)

        def forward(self, idx):
            bsz, length = idx.shape
            state = self.tok(idx) + self.pos(torch.arange(length, device=idx.device).unsqueeze(0))
            indices = self.targets[:length]
            valid = self.valid[:length].view(1, 1, length, len(offsets))
            scale = 1.0 / math.sqrt(d_model // n_heads)
            residual_scale = 1.0 / math.sqrt(thought_hops)
            for _ in range(thought_hops):
                z = self.ln(state)
                q = self.q(z).view(bsz, length, n_heads, -1).transpose(1, 2)
                k = self.k(z).view(bsz, length, n_heads, -1).transpose(1, 2)
                v = self.v(z).view(bsz, length, n_heads, -1).transpose(1, 2)
                scores = (q.unsqueeze(3) * k[:, :, indices, :]).sum(-1) * scale
                weights = F.softmax(scores.masked_fill(~valid, -1e4), dim=-1)
                context = (weights.unsqueeze(-1) * v[:, :, indices, :]).sum(3)
                message = context.transpose(1, 2).contiguous().view(bsz, length, d_model)
                if mode not in {"attention_only", "attention_only_final_mlp"}:
                    message = self.o(message)
                    if mode == "replacement":
                        state = message
                    elif mode == "gru":
                        updated = self.gru(message.reshape(-1, d_model), state.reshape(-1, d_model)).view(bsz, length, d_model)
                        state = state + residual_scale * (updated - state)
                    else:
                        state = state + residual_scale * message
                    state = state + residual_scale * self.mlp(self.ln_mlp(state))
                else:
                    state = message
            if mode == "attention_only_final_mlp":
                state = state + self.mlp(self.ln_mlp(state))
            return self.head(self.ln_f(state))

    model = CoreSubQ().to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + d_model * 31 + thought_hops * 1000 + len(mode) * 101)
    start = time.time()
    for _ in range(total_steps):
        model.train()
        x, y, _ = make_batch(generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
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
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(seed + 900000 + d_model)
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
    experiment = "S2-024" if mode == "attention_only_final_mlp" else "S2-023"
    return {"experiment": experiment, "model": model_name, "mode": mode, "seed": seed, "d_model": d_model, "d_mlp": d_mlp, "thought_hops": thought_hops, "parameters": parameters, "steps": total_steps, "accuracy": round(100.0 * correct / total, 4), "distance_accuracy": {name: round(100.0 * a / b, 4) for name, (a, b) in stats.items()}, "queries": total, "elapsed_s": round(time.time() - start, 3)}


@app.local_entrypoint()
def main():
    import os
    import pathlib
    import time

    seeds = [42, 1337, 2026]
    conditions = [
        ("residual_t8_d128", "residual", 128, 512, 8),
        ("gru_t8_d128", "gru", 128, 126, 8),
        ("replacement_t8_d128", "replacement", 128, 512, 8),
        ("attention_only_t8_d128", "attention_only", 128, 0, 8),
        ("residual_t4_d256", "residual", 256, 1024, 4),
        ("gru_t4_d256", "gru", 256, 1024, 4),
    ]
    corrected_only = os.environ.get("CORE_SWEEP_CORRECTED") == "1"
    if corrected_only:
        conditions = [("attention_only_final_mlp_t8_d128", "attention_only_final_mlp", 128, 512, 8)]
    configs = [(f"{name}_s{seed}", mode, seed, d_model, d_mlp, hops) for name, mode, d_model, d_mlp, hops in conditions for seed in seeds]
    calls = [train_one.spawn(name, mode, seed, d_model, d_mlp, hops) for name, mode, seed, d_model, d_mlp, hops in configs]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    experiment = "S2-024" if corrected_only else "S2-023"
    description = "Attention-only T=8 with one final MLP after all attention iterations" if corrected_only else "Core SubQ sweep: T=8, attention-only, and width scaling"
    payload = {"experiment": experiment, "description": description, "source_commit": "working-tree", "task": "16 independently randomized key/value pairs, 8 late queries", "offsets": [0, 1, 2, 4, 8, 16, 32, 64], "conditions": conditions, "seeds": seeds, "training_steps": 6000, "random_exact_baseline_percent": 2.5, "results": results, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = pathlib.Path(f"season2/results/{experiment.lower()}_attention_only_final_mlp.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}", flush=True)
