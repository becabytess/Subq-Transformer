"""S2-008: Test max-64 path length and forced-route propagation.

This keeps the one-pair randomized-position propagation task fixed while
comparing learned sparse routing with a forced route that always takes the
largest available offset (64). The forced condition removes routing
discovery from the test but retains the same Q/K/V, residual, and MLP update
machinery.

Run from the repository root:
    modal run season2/experiments/s2_008_path_length_oracle.py
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-008-path-length-oracle")


@app.function(image=image, gpu="A10G", timeout=3600)
def train_one(
    model_name: str,
    thought_hops: int,
    value_distance: int,
    route_mode: str,
    seed: int,
    total_steps: int = 6000,
    seq_len: int = 512,
    batch_size: int = 32,
    d_model: int = 128,
    n_heads: int = 4,
    d_mlp: int = 512,
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
    offsets = [0, 1, 2, 4, 8, 16, 32, 64]
    max_offset_index = offsets.index(64)
    min_query_target = value_distance + 1
    target_parameters = 329088

    class SparseResidualMQAR(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.ln_attn = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model),
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            target_indices = torch.zeros((seq_len, len(offsets)), dtype=torch.long)
            valid = torch.zeros((seq_len, len(offsets)), dtype=torch.bool)
            for position in range(seq_len):
                for offset_idx, offset in enumerate(offsets):
                    target = position - offset
                    if target >= 0:
                        target_indices[position, offset_idx] = target
                        valid[position, offset_idx] = True
            self.register_buffer("target_indices", target_indices)
            self.register_buffer("valid", valid)

        def forward(self, idx):
            bsz, length = idx.shape
            positions = torch.arange(length, device=idx.device).unsqueeze(0)
            state = self.tok_emb(idx) + self.pos_emb(positions)
            indices = self.target_indices[:length]
            valid = self.valid[:length].view(1, 1, length, len(offsets))
            scale = 1.0 / math.sqrt(d_model // n_heads)
            residual_scale = 1.0 / math.sqrt(thought_hops)

            for _ in range(thought_hops):
                normalized = self.ln_attn(state)
                q = self.q_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
                k = self.k_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
                v = self.v_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
                k_candidates = k[:, :, indices, :]
                v_candidates = v[:, :, indices, :]

                if route_mode == "forced_64":
                    context = v_candidates[:, :, :, max_offset_index, :]
                else:
                    scores = (q.unsqueeze(3) * k_candidates).sum(-1) * scale
                    scores = scores.masked_fill(~valid, -1e4)
                    weights = F.softmax(scores, dim=-1)
                    context = (weights.unsqueeze(-1) * v_candidates).sum(3)

                context = context.transpose(1, 2).contiguous().view(bsz, length, d_model)
                state = state + residual_scale * self.out_proj(context)
                state = state + residual_scale * self.mlp(self.ln_mlp(state))

            return self.head(self.ln_f(state))

    def make_batch(generator):
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
        keys = torch.randint(10, 50, (batch_size,), generator=generator, device=device)
        values = torch.randint(50, 90, (batch_size,), generator=generator, device=device)
        value_positions = target_positions - value_distance
        key_positions = value_positions - 1
        rows = torch.arange(batch_size, device=device)
        x[rows, key_positions] = keys
        x[rows, value_positions] = values
        x[rows, target_positions - 1] = 1
        x[rows, target_positions] = keys
        y[rows, target_positions] = values
        return x, y, target_positions

    def evaluate(model, generator):
        model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for _ in range(validation_batches):
                x, y, target_positions = make_batch(generator)
                with torch.amp.autocast(
                    "cuda", dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        y.reshape(-1),
                        ignore_index=-100,
                    )
                rows = torch.arange(batch_size, device=device)
                predictions = logits[rows, target_positions].argmax(dim=-1)
                correct += predictions.eq(y[rows, target_positions]).sum().item()
                total += batch_size
                loss_sum += loss.item() * batch_size
        return {
            "accuracy": round(100.0 * correct / total, 4),
            "loss": round(loss_sum / total, 6),
            "queries": total,
        }

    model = SparseResidualMQAR().to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + value_distance * 100 + thought_hops * 1000)
    start = time.time()
    checkpoints = []
    checkpoint_steps = {1000, 3000, total_steps}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, total_steps + 1):
        model.train()
        x, y, _ = make_batch(generator)
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
            checkpoint_generator.manual_seed(seed + value_distance * 10000 + step)
            checkpoints.append({"step": step, **evaluate(model, checkpoint_generator)})

    elapsed = time.time() - start
    return {
        "experiment": "S2-008",
        "model": model_name,
        "update_mode": "sparse_residual",
        "route_mode": route_mode,
        "thought_hops": thought_hops,
        "value_distance": value_distance,
        "offsets": offsets,
        "offset_horizon": max(offsets),
        "forced_offset": 64 if route_mode == "forced_64" else None,
        "residual_scale": 1.0 / math.sqrt(thought_hops),
        "seed": seed,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_mlp": d_mlp,
        "parameters": parameters,
        "target_parameters": target_parameters,
        "parameter_delta_vs_target": parameters - target_parameters,
        "steps": total_steps,
        "checkpoints": checkpoints,
        "elapsed_s": round(elapsed, 3),
        "tokens_per_s_train": round(total_steps * batch_size * seq_len / elapsed, 2),
        "peak_memory_mb": round(
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if device.type == "cuda"
            else 0.0,
            2,
        ),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }


@app.local_entrypoint()
def main():
    import pathlib
    import time

    distances_and_hops = [(128, 2), (192, 3), (256, 4)]
    route_modes = ["learned", "forced_64"]
    seeds = [42, 1337, 2026]
    configs = [
        (
            f"{route_mode}_d{distance}_t{hops}_s{seed}",
            hops,
            distance,
            route_mode,
            seed,
        )
        for seed in seeds
        for distance, hops in distances_and_hops
        for route_mode in route_modes
    ]

    calls = [
        train_one.spawn(name, hops, distance, route_mode, seed)
        for name, hops, distance, route_mode, seed in configs
    ]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "experiment": "S2-008",
        "description": "Max-64 path-length and forced-route propagation diagnostic",
        "source_commit": "working-tree",
        "dataset": "Synthetic one-pair randomized-position propagation task",
        "causal": True,
        "seq_len": 512,
        "query_target_position_range_by_distance": {
            str(distance): [distance + 1, 511]
            for distance, _ in distances_and_hops
        },
        "value_distances": [distance for distance, _ in distances_and_hops],
        "num_kv_pairs": 1,
        "num_queries": 1,
        "random_exact_baseline_percent": 2.5,
        "offsets": [0, 1, 2, 4, 8, 16, 32, 64],
        "training_steps": 6000,
        "validation_batches": 100,
        "seeds": seeds,
        "parameter_target": 329088,
        "conditions": [
            {
                "route_mode": route_mode,
                "value_distance": distance,
                "thought_hops": hops,
            }
            for distance, hops in distances_and_hops
            for route_mode in route_modes
        ],
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = pathlib.Path("season2/results/s2_008_path_length_oracle.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
