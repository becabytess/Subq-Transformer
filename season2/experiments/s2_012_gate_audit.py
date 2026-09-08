"""S2-012: Audit convex and learned writes across distances and routing.

This tests the best fixed convex write and the learned scalar gate at three
propagation distances under both the forced max-64 route and learned sparse
attention. Gate values are recorded by hop for all positions and query
positions.

Run from the repository root:
    modal run season2/experiments/s2_012_gate_audit.py
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-012-gate-audit")


@app.function(image=image, gpu="A10G", timeout=3600)
def train_one(
    model_name: str,
    update_mode: str,
    route_mode: str,
    value_distance: int,
    thought_hops: int,
    seed: int,
    total_steps: int = 6000,
    seq_len: int = 512,
    batch_size: int = 32,
    d_model: int = 128,
    n_heads: int = 4,
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

    class GateAuditModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.ln_attn = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            if update_mode == "learned_gate":
                self.gate_proj = nn.Linear(2 * d_model, 1)
                nn.init.zeros_(self.gate_proj.weight)
                nn.init.zeros_(self.gate_proj.bias)
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

        def forward(self, idx, capture_gates=False):
            bsz, length = idx.shape
            positions = torch.arange(length, device=idx.device).unsqueeze(0)
            state = self.tok_emb(idx) + self.pos_emb(positions)
            indices = self.target_indices[:length]
            valid = self.valid[:length].view(1, 1, length, len(offsets))
            forced_indices = indices[:, max_offset_index]
            scale = 1.0 / math.sqrt(d_model // n_heads)
            gates = []

            for _ in range(thought_hops):
                normalized = self.ln_attn(state)
                q = self.q_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
                k = self.k_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
                v = self.v_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)

                if route_mode == "forced_64":
                    context = v[:, :, forced_indices, :]
                    context = context.transpose(1, 2).contiguous().view(
                        bsz, length, d_model
                    )
                else:
                    k_candidates = k[:, :, indices, :]
                    v_candidates = v[:, :, indices, :]
                    scores = (q.unsqueeze(3) * k_candidates).sum(-1) * scale
                    scores = scores.masked_fill(~valid, -1e4)
                    weights = F.softmax(scores, dim=-1)
                    context = (weights.unsqueeze(-1) * v_candidates).sum(3)
                    context = context.transpose(1, 2).contiguous().view(
                        bsz, length, d_model
                    )

                message = self.out_proj(context)
                if update_mode == "convex_075":
                    state = 0.25 * state + 0.75 * message
                elif update_mode == "learned_gate":
                    gate_input = torch.cat(
                        [self.ln_attn(state), self.ln_attn(message)], dim=-1
                    )
                    gate = torch.sigmoid(self.gate_proj(gate_input))
                    state = (1.0 - gate) * state + gate * message
                    if capture_gates:
                        gates.append(gate.squeeze(-1))
                else:
                    raise ValueError(f"unknown update mode: {update_mode}")

            logits = self.head(self.ln_f(state))
            if capture_gates:
                return logits, gates
            return logits

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
        all_gate_sum = [0.0 for _ in range(thought_hops)]
        target_gate_sum = [0.0 for _ in range(thought_hops)]
        gate_count = 0
        with torch.no_grad():
            for _ in range(validation_batches):
                x, y, target_positions = make_batch(generator)
                with torch.amp.autocast(
                    "cuda", dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    if update_mode == "learned_gate":
                        logits, gates = model(x, capture_gates=True)
                    else:
                        logits = model(x)
                        gates = []
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
                if gates:
                    for hop, gate in enumerate(gates):
                        all_gate_sum[hop] += gate.float().mean().item() * batch_size
                        target_gate_sum[hop] += gate[rows, target_positions].float().mean().item() * batch_size
                    gate_count += batch_size

        result = {
            "accuracy": round(100.0 * correct / total, 4),
            "loss": round(loss_sum / total, 6),
            "queries": total,
        }
        if gate_count:
            result["gate_mean_all_positions_by_hop"] = [
                round(value / gate_count, 6) for value in all_gate_sum
            ]
            result["gate_mean_query_positions_by_hop"] = [
                round(value / gate_count, 6) for value in target_gate_sum
            ]
        return result

    model = GateAuditModel().to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device)
    route_offset = 0 if route_mode == "forced_64" else 500000
    generator.manual_seed(seed + value_distance * 100 + route_offset)
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
            checkpoint_generator.manual_seed(
                seed + value_distance * 10000 + route_offset + step
            )
            checkpoints.append({"step": step, **evaluate(model, checkpoint_generator)})

    elapsed = time.time() - start
    return {
        "experiment": "S2-012",
        "model": model_name,
        "update_mode": update_mode,
        "route_mode": route_mode,
        "value_distance": value_distance,
        "thought_hops": thought_hops,
        "offsets": offsets,
        "offset_horizon": max(offsets),
        "convex_alpha": 0.75 if update_mode == "convex_075" else None,
        "seed": seed,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "parameters": parameters,
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

    seeds = [42, 1337, 2026]
    distances_and_hops = [(128, 2), (192, 3), (256, 4)]
    update_modes = ["convex_075", "learned_gate"]
    route_modes = ["forced_64", "learned"]
    configs = [
        (
            f"{update_mode}_{route_mode}_d{distance}_t{hops}_s{seed}",
            update_mode,
            route_mode,
            distance,
            hops,
            seed,
        )
        for seed in seeds
        for distance, hops in distances_and_hops
        for route_mode in route_modes
        for update_mode in update_modes
    ]

    calls = [
        train_one.spawn(name, update_mode, route_mode, distance, hops, seed)
        for name, update_mode, route_mode, distance, hops, seed in configs
    ]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "experiment": "S2-012",
        "description": "Convex and learned-gate audit across distances and routing modes",
        "source_commit": "working-tree",
        "dataset": "Synthetic one-pair randomized-position propagation task",
        "causal": True,
        "seq_len": 512,
        "query_target_position_range_by_distance": {
            str(distance): [distance + 1, 511]
            for distance, _ in distances_and_hops
        },
        "distances_and_hops": [
            {"value_distance": distance, "thought_hops": hops}
            for distance, hops in distances_and_hops
        ],
        "offsets": [0, 1, 2, 4, 8, 16, 32, 64],
        "random_exact_baseline_percent": 2.5,
        "training_steps": 6000,
        "validation_batches": 100,
        "seeds": seeds,
        "update_modes": update_modes,
        "route_modes": route_modes,
        "gate_definition": "Scalar per-token sigmoid gate over LayerNorm(state) and LayerNorm(message), initialized at 0.5.",
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = pathlib.Path("season2/results/s2_012_gate_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
