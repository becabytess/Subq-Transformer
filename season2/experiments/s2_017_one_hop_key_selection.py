"""S2-017: one-hop SubQ key-selection diagnostic.

The query key is guaranteed to be present among the visible candidate keys.
There is no long-range transport, value adjacency, oracle source, or self
candidate. The model must use ordinary Q/K attention to select the matching
candidate and copy its key through V to the output.

Run from the repository root:
    modal run season2/experiments/s2_017_one_hop_key_selection.py
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-017-one-hop-key-selection")


@app.function(image=image, gpu="T4", timeout=3600)
def train_one(
    model_name: str,
    offsets: list[int],
    seed: int,
    total_steps: int = 6000,
    seq_len: int = 256,
    batch_size: int = 64,
    d_model: int = 128,
    n_heads: int = 4,
    validation_batches: int = 50,
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

    vocab_size = 256
    key_low, key_high = 10, 50
    background_low, background_high = 100, 255
    target_position = 200
    candidate_count = len(offsets)

    def make_batch(generator):
        x = torch.randint(
            background_low,
            background_high,
            (batch_size, seq_len),
            generator=generator,
            device=device,
        )
        y = torch.full((batch_size,), -100, dtype=torch.long, device=device)
        source_slot = torch.randint(
            0, candidate_count, (batch_size,), generator=generator, device=device
        )
        candidate_positions = torch.tensor(
            [target_position - offset for offset in offsets],
            device=device,
            dtype=torch.long,
        )
        for row in range(batch_size):
            keys = torch.randperm(
                key_high - key_low, generator=generator, device=device
            )[:candidate_count] + key_low
            x[row, candidate_positions] = keys
            query_key = keys[source_slot[row]]
            x[row, target_position] = query_key
            y[row] = query_key
        return x, y, source_slot, candidate_positions

    class OneHopSubQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.ln = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, capture_attention=False):
            bsz, length = idx.shape
            positions = torch.arange(length, device=idx.device).unsqueeze(0)
            state = self.tok_emb(idx) + self.pos_emb(positions)
            normalized = self.ln(state)
            q = self.q_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
            k = self.k_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
            v = self.v_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
            candidate_positions = torch.tensor(
                [target_position - offset for offset in offsets],
                device=idx.device,
                dtype=torch.long,
            )
            candidate_k = k[:, :, candidate_positions, :]
            candidate_v = v[:, :, candidate_positions, :]
            query = q[:, :, target_position : target_position + 1, :]
            scores = (query.unsqueeze(3) * candidate_k.unsqueeze(2)).sum(-1)
            scores = scores * (1.0 / math.sqrt(d_model // n_heads))
            weights = F.softmax(scores, dim=-1)
            context = (weights.unsqueeze(-1) * candidate_v.unsqueeze(2)).sum(3)
            context = context.transpose(1, 2).contiguous().view(bsz, d_model)
            logits = self.head(self.ln_f(self.out_proj(context)))
            if capture_attention:
                return logits, weights.squeeze(2)
            return logits

    model = OneHopSubQ().to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-4
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + sum(offsets) * 17)
    checkpoints = []
    start = time.time()

    def evaluate(eval_generator):
        model.eval()
        correct = 0
        total = 0
        top1_correct = 0
        top1_total = 0
        target_weight = 0.0
        with torch.no_grad():
            for _ in range(validation_batches):
                x, y, source_slot, _ = make_batch(eval_generator)
                logits, weights = model(x, capture_attention=True)
                prediction = logits.argmax(dim=-1)
                correct += prediction.eq(y).sum().item()
                total += y.numel()
                top1 = weights.mean(dim=1).argmax(dim=-1)
                top1_correct += top1.eq(source_slot).sum().item()
                top1_total += source_slot.numel()
                target_weight += weights.mean(dim=1).gather(
                    1, source_slot[:, None]
                ).sum().item()
        return {
            "accuracy": round(100.0 * correct / total, 4),
            "attention_top1_accuracy": round(100.0 * top1_correct / top1_total, 4),
            "mean_correct_candidate_weight": round(target_weight / top1_total, 6),
            "queries": total,
        }

    for step in range(1, total_steps + 1):
        model.train()
        x, y, _, _ = make_batch(generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step in {1000, 3000, total_steps}:
            eval_generator = torch.Generator(device=device)
            eval_generator.manual_seed(seed + 100000 + step)
            checkpoints.append({"step": step, **evaluate(eval_generator)})

    return {
        "experiment": "S2-017",
        "model": model_name,
        "offsets": offsets,
        "candidate_count": candidate_count,
        "seed": seed,
        "seq_len": seq_len,
        "target_position": target_position,
        "d_model": d_model,
        "n_heads": n_heads,
        "parameters": parameters,
        "steps": total_steps,
        "checkpoints": checkpoints,
        "elapsed_s": round(time.time() - start, 3),
    }


@app.local_entrypoint()
def main():
    import pathlib
    import time

    offsets = [1, 2, 4, 8, 16, 32, 64, 96]
    seeds = [42, 1337, 2026]
    configs = [(f"one_hop_s{seed}", offsets, seed) for seed in seeds]
    calls = [train_one.spawn(name, policy, seed) for name, policy, seed in configs]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    payload = {
        "experiment": "S2-017",
        "description": "One-hop SubQ key-selection diagnostic",
        "source_commit": "working-tree",
        "task": "The query key is guaranteed to be one of the visible candidate keys; output is the query key.",
        "offsets": offsets,
        "random_exact_baseline_percent": 2.5,
        "seeds": seeds,
        "training_steps": 6000,
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = pathlib.Path("season2/results/s2_017_one_hop_key_selection.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
