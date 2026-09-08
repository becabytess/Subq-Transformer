"""S2-005 dense-control convergence audit.

This is a longer dense-only control for S2-005. It preserves the same
one-pair/one-query generator and distance conditions, but trains for 10,000
steps and records validation checkpoints.

Run from the repository root:
    modal run season2/experiments/s2_005_dense_audit.py
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-005-dense-audit")


@app.function(image=image, gpu="A10G", timeout=3600)
def train_one(
    seed: int,
    total_steps: int = 10000,
    seq_len: int = 512,
    batch_size: int = 32,
    d_model: int = 128,
    n_heads: int = 4,
    d_mlp: int = 127,
    validation_batches: int = 100,
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
    query_marker = 1
    min_query_target = 257
    distances = {"local_32": 32, "medium_128": 128, "long_256": 256}
    target_parameters = 329088

    class DenseMQAR(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.blocks = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln1": nn.LayerNorm(d_model),
                            "q": nn.Linear(d_model, d_model, bias=False),
                            "k": nn.Linear(d_model, d_model, bias=False),
                            "v": nn.Linear(d_model, d_model, bias=False),
                            "o": nn.Linear(d_model, d_model, bias=False),
                            "ln2": nn.LayerNorm(d_model),
                            "mlp": nn.Sequential(
                                nn.Linear(d_model, d_mlp),
                                nn.GELU(),
                                nn.Linear(d_mlp, d_model),
                            ),
                        }
                    )
                    for _ in range(2)
                ]
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.register_buffer(
                "causal_mask", torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            )

        def forward(self, idx):
            bsz, length = idx.shape
            positions = torch.arange(length, device=idx.device).unsqueeze(0)
            state = self.tok_emb(idx) + self.pos_emb(positions)
            head_dim = d_model // n_heads
            for block in self.blocks:
                normalized = block["ln1"](state)
                q = block["q"](normalized).view(bsz, length, n_heads, head_dim).transpose(1, 2)
                k = block["k"](normalized).view(bsz, length, n_heads, head_dim).transpose(1, 2)
                v = block["v"](normalized).view(bsz, length, n_heads, head_dim).transpose(1, 2)
                scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
                scores = scores.masked_fill(~self.causal_mask[:length, :length], -1e4)
                weights = F.softmax(scores, dim=-1)
                context = (weights @ v).transpose(1, 2).contiguous().view(bsz, length, d_model)
                state = state + block["o"](context)
                state = state + block["mlp"](block["ln2"](state))
            return self.head(self.ln_f(state))

    def make_batch(value_distance, generator):
        x = torch.randint(
            100, 255, (batch_size, seq_len), generator=generator, device=device
        )
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        target_positions = torch.randint(
            min_query_target,
            seq_len,
            (batch_size,),
            generator=generator,
            device=device,
        )
        keys = torch.randint(
            10, 50, (batch_size,), generator=generator, device=device
        )
        values = torch.randint(
            50, 90, (batch_size,), generator=generator, device=device
        )
        value_positions = target_positions - value_distance
        key_positions = value_positions - 1
        rows = torch.arange(batch_size, device=device)
        x[rows, key_positions] = keys
        x[rows, value_positions] = values
        x[rows, target_positions - 1] = query_marker
        x[rows, target_positions] = keys
        y[rows, target_positions] = values
        return x, y, target_positions

    def evaluate(model, value_distance, generator):
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(validation_batches):
                x, y, target_positions = make_batch(value_distance, generator)
                with torch.amp.autocast(
                    "cuda", dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    logits = model(x)
                    batch_loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100
                    )
                rows = torch.arange(batch_size, device=device)
                prediction = logits[rows, target_positions].argmax(dim=-1)
                correct += prediction.eq(y[rows, target_positions]).sum().item()
                total += batch_size
                val_loss += batch_loss.item() * batch_size
        return {
            "accuracy": round(100.0 * correct / total, 4),
            "loss": round(val_loss / total, 6),
            "queries": total,
        }

    model = DenseMQAR().to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_generator = torch.Generator(device=device)
    train_generator.manual_seed(seed + 17)
    start = time.time()
    checkpoints = []
    checkpoint_steps = {1000, 3000, 6000, total_steps}

    for step in range(1, total_steps + 1):
        model.train()
        distance_index = (step - 1) % len(distances)
        value_distance = list(distances.values())[distance_index]
        x, y, _ = make_batch(value_distance, train_generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step in checkpoint_steps:
            checkpoint_generator = torch.Generator(device=device)
            checkpoint_generator.manual_seed(seed + step * 1000 + 101)
            checkpoint = {
                "step": step,
                "conditions": {
                    name: evaluate(model, distance, checkpoint_generator)
                    for name, distance in distances.items()
                },
            }
            checkpoints.append(checkpoint)

    elapsed = time.time() - start
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "experiment": "S2-005-dense-audit",
        "seed": seed,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_mlp": d_mlp,
        "num_kv_pairs": 1,
        "num_queries": 1,
        "query_target_position_range": [257, 511],
        "value_distances": distances,
        "parameters": parameters,
        "target_parameters": target_parameters,
        "parameter_delta_vs_target": parameters - target_parameters,
        "steps": total_steps,
        "checkpoints": checkpoints,
        "elapsed_s": round(elapsed, 3),
        "tokens_per_s_train": round(total_steps * batch_size * seq_len / elapsed, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }


@app.local_entrypoint()
def main():
    import pathlib
    import time

    seeds = [42, 1337, 2026]
    calls = [train_one.spawn(seed) for seed in seeds]
    results = []
    for seed, call in zip(seeds, calls):
        print(f"Waiting for dense audit seed {seed}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "experiment": "S2-005-dense-audit",
        "description": "10,000-step dense-control convergence audit for one-pair/one-query MQAR",
        "source_commit": "working-tree",
        "dataset": "Synthetic MQAR with one independently randomized key-value pair and one query",
        "causal": True,
        "seq_len": 512,
        "query_target_position_range": [257, 511],
        "value_distances": {"local_32": 32, "medium_128": 128, "long_256": 256},
        "seeds": seeds,
        "training_steps": 10000,
        "validation_batches": 100,
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = pathlib.Path("season2/results/s2_005_dense_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
