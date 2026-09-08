"""S2-016: Routing audit for corrected multi-pair MQAR.

This keeps the corrected MQAR task, learned gated write, model width, hop
count, optimizer, and training protocol fixed. It records learned sparse
attention distributions at query positions to audit route quality.

Run from the repository root:
    modal run season2/experiments/s2_016_routing_audit_mqar.py
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-016-routing-audit-mqar")


@app.function(image=image, gpu="T4", timeout=3600)
def train_one(
    model_name: str,
    offsets: list[int],
    seed: int,
    total_steps: int = 6000,
    seq_len: int = 512,
    batch_size: int = 32,
    d_model: int = 128,
    n_heads: int = 4,
    d_mlp: int = 512,
    num_kv_pairs: int = 16,
    num_queries: int = 8,
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
    query_marker = 1
    thought_hops = 4
    start_q_pos = seq_len - (num_queries * 2) - 2
    query_value_positions = [
        start_q_pos + (query_idx * 2) + 1 for query_idx in range(num_queries)
    ]

    def make_batch(generator):
        x = torch.randint(
            100, 255, (batch_size, seq_len), generator=generator, device=device
        )
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        query_distances = torch.empty(
            (batch_size, num_queries), dtype=torch.long, device=device
        )
        source_positions = torch.empty(
            (batch_size, num_queries), dtype=torch.long, device=device
        )

        for row in range(batch_size):
            keys = torch.randperm(40, generator=generator, device=device)[:num_kv_pairs] + 10
            values = torch.randperm(40, generator=generator, device=device)[:num_kv_pairs] + 50
            kv_positions = torch.randperm(
                350, generator=generator, device=device
            )[: num_kv_pairs * 2].sort().values
            for pair_idx in range(num_kv_pairs):
                key_pos = int(kv_positions[pair_idx * 2].item())
                value_pos = key_pos + 1
                x[row, key_pos] = keys[pair_idx]
                x[row, value_pos] = values[pair_idx]

            query_indices = torch.randperm(
                num_kv_pairs, generator=generator, device=device
            )[:num_queries]
            for query_idx, key_idx_tensor in enumerate(query_indices):
                key_idx = int(key_idx_tensor.item())
                query_pos = start_q_pos + (query_idx * 2)
                x[row, query_pos] = query_marker
                x[row, query_pos + 1] = keys[key_idx]
                y[row, query_pos + 1] = values[key_idx]
                key_pos = int(kv_positions[key_idx * 2].item())
                query_distances[row, query_idx] = query_pos + 1 - key_pos
                source_positions[row, query_idx] = key_pos + 1

        return x, y, query_distances, source_positions

    class GatedSparseMQAR(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.ln_attn = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.gate_proj = nn.Linear(2 * d_model, 1)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)
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

        def forward(self, idx, capture_gates=False, capture_routes=False):
            bsz, length = idx.shape
            positions = torch.arange(length, device=idx.device).unsqueeze(0)
            state = self.tok_emb(idx) + self.pos_emb(positions)
            indices = self.target_indices[:length]
            valid = self.valid[:length].view(1, 1, length, len(offsets))
            scale = 1.0 / math.sqrt(d_model // n_heads)
            residual_scale = 1.0 / math.sqrt(thought_hops)
            gates = []
            routes = []

            for _ in range(thought_hops):
                normalized = self.ln_attn(state)
                q = self.q_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
                k = self.k_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
                v = self.v_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
                k_candidates = k[:, :, indices, :]
                v_candidates = v[:, :, indices, :]
                scores = (q.unsqueeze(3) * k_candidates).sum(-1) * scale
                scores = scores.masked_fill(~valid, -1e4)
                weights = F.softmax(scores, dim=-1)
                if capture_routes:
                    routes.append(weights)
                context = (weights.unsqueeze(-1) * v_candidates).sum(3)
                context = context.transpose(1, 2).contiguous().view(bsz, length, d_model)
                message = self.out_proj(context)

                gate_input = torch.cat(
                    [self.ln_attn(state), self.ln_attn(message)], dim=-1
                )
                gate = torch.sigmoid(self.gate_proj(gate_input))
                state = (1.0 - gate) * state + gate * message
                if capture_gates:
                    gates.append(gate.squeeze(-1))
                state = state + residual_scale * self.mlp(self.ln_mlp(state))

            logits = self.head(self.ln_f(state))
            if capture_gates or capture_routes:
                return logits, gates, routes
            return logits

    model = GatedSparseMQAR().to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + sum(offsets) * 13 + thought_hops * 1000)
    start = time.time()
    checkpoints = []
    checkpoint_steps = {1000, 3000, total_steps}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def evaluate(eval_generator):
        model.eval()
        total_queries = 0
        correct_queries = 0
        val_loss = 0.0
        val_tokens = 0
        distance_bins = {
            "128-255": [128, 256],
            "256-383": [256, 384],
            "384-511": [384, 512],
        }
        distance_stats = {
            name: {"correct": 0, "total": 0} for name in distance_bins
        }
        distance_sum = 0
        gate_sums = [0.0 for _ in range(thought_hops)]
        gate_query_sums = [0.0 for _ in range(thought_hops)]
        route_weight_sums = [
            torch.zeros(len(offsets), device=device) for _ in range(thought_hops)
        ]
        route_entropy_sums = [0.0 for _ in range(thought_hops)]
        route_max_prob_sums = [0.0 for _ in range(thought_hops)]
        route_greedy_hits = [0 for _ in range(thought_hops)]
        route_direct_hits = [0 for _ in range(thought_hops)]
        route_observations = 0
        gate_count = 0
        query_positions = torch.tensor(query_value_positions, device=device)
        offsets_tensor = torch.tensor(offsets, device=device)

        with torch.no_grad():
            for _ in range(validation_batches):
                x, y, query_distances, source_positions = make_batch(eval_generator)
                with torch.amp.autocast(
                    "cuda", dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    logits, gates, routes = model(
                        x, capture_gates=True, capture_routes=True
                    )
                    batch_loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        y.reshape(-1),
                        ignore_index=-100,
                    )
                val_loss += batch_loss.item() * (y != -100).sum().item()
                val_tokens += (y != -100).sum().item()
                query_logits = logits[:, query_value_positions, :]
                query_targets = y[:, query_value_positions]
                query_correct = query_logits.argmax(dim=-1).eq(query_targets)
                correct_queries += query_correct.sum().item()
                total_queries += query_targets.numel()
                distance_sum += query_distances.sum().item()
                for name, (lower, upper) in distance_bins.items():
                    mask = (query_distances >= lower) & (query_distances < upper)
                    distance_stats[name]["correct"] += (query_correct & mask).sum().item()
                    distance_stats[name]["total"] += mask.sum().item()
                rows = torch.arange(batch_size, device=device)
                for hop, gate in enumerate(gates):
                    gate_sums[hop] += gate.float().mean().item() * batch_size
                    gate_query_sums[hop] += gate[rows[:, None], query_positions[None, :]].float().mean().item() * batch_size
                gate_count += batch_size

                greedy_offsets = torch.where(
                    offsets_tensor.view(1, 1, -1) <= query_distances.unsqueeze(-1),
                    offsets_tensor.view(1, 1, -1),
                    torch.zeros(
                        1, 1, len(offsets), device=device, dtype=torch.long
                    ),
                ).max(dim=-1).values
                direct_offsets = query_positions.view(1, -1) - source_positions
                for hop, route in enumerate(routes):
                    query_route = route[:, :, query_positions, :].float()
                    route_weight_sums[hop] += query_route.sum(dim=(0, 1, 2))
                    safe_route = query_route.clamp_min(1e-8)
                    route_entropy_sums[hop] += (
                        -(safe_route * safe_route.log()).sum(-1).sum().item()
                    )
                    route_max_prob_sums[hop] += (
                        query_route.max(dim=-1).values.sum().item()
                    )
                    top_offsets = offsets_tensor[query_route.argmax(dim=-1)]
                    route_greedy_hits[hop] += top_offsets.eq(
                        greedy_offsets.unsqueeze(1)
                    ).sum().item()
                    route_direct_hits[hop] += top_offsets.eq(
                        direct_offsets.unsqueeze(1)
                    ).sum().item()
                route_observations += batch_size * n_heads * num_queries

        for stats in distance_stats.values():
            stats["accuracy"] = round(100.0 * stats["correct"] / stats["total"], 4)
        return {
            "accuracy": round(100.0 * correct_queries / total_queries, 4),
            "val_loss": round(val_loss / val_tokens, 6),
            "queries": total_queries,
            "mean_query_distance": round(distance_sum / total_queries, 3),
            "distance_accuracy": distance_stats,
            "gate_mean_all_positions_by_hop": [
                round(value / gate_count, 6) for value in gate_sums
            ],
            "gate_mean_query_positions_by_hop": [
                round(value / gate_count, 6) for value in gate_query_sums
            ],
            "route_offsets": offsets,
            "route_mean_query_weights_by_hop": [
                [round(value / route_observations, 6) for value in sums.tolist()]
                for sums in route_weight_sums
            ],
            "route_query_entropy_by_hop": [
                round(value / route_observations, 6)
                for value in route_entropy_sums
            ],
            "route_query_max_prob_by_hop": [
                round(value / route_observations, 6)
                for value in route_max_prob_sums
            ],
            "route_greedy_offset_hit_by_hop": [
                round(100.0 * value / route_observations, 4)
                for value in route_greedy_hits
            ],
            "route_direct_offset_hit_by_hop": [
                round(100.0 * value / route_observations, 4)
                for value in route_direct_hits
            ],
        }

    for step in range(1, total_steps + 1):
        model.train()
        x, y, _, _ = make_batch(generator)
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
            eval_generator = torch.Generator(device=device)
            eval_generator.manual_seed(seed + sum(offsets) * 10000 + step)
            checkpoints.append({"step": step, **evaluate(eval_generator)})

    elapsed = time.time() - start
    return {
        "experiment": "S2-016",
        "model": model_name,
        "update_mode": "learned_gate_full",
        "route_mode": "learned_audited",
        "thought_hops": thought_hops,
        "offsets": offsets,
        "offset_horizon": max(offsets),
        "k": len(offsets),
        "seed": seed,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_mlp": d_mlp,
        "num_kv_pairs": num_kv_pairs,
        "num_queries": num_queries,
        "parameters": parameters,
        "target_parameters": 329088,
        "parameter_delta_vs_target": parameters - 329088,
        "steps": total_steps,
        "checkpoints": checkpoints,
        "elapsed_s": round(elapsed, 3),
        "train_tokens_per_s": round(total_steps * batch_size * seq_len / elapsed, 2),
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
    offset_policies = {
        "h64": [0, 1, 2, 4, 8, 16, 32, 64],
        "h128": [0, 1, 2, 4, 8, 16, 32, 128],
        "h256": [0, 1, 2, 4, 8, 16, 32, 256],
        "h384": [0, 1, 2, 4, 8, 16, 32, 384],
    }
    audit_policies = {
        "h64": offset_policies["h64"],
        "h384": offset_policies["h384"],
    }
    configs = [
        (f"{name}_t4_s{seed}", offsets, seed)
        for seed in seeds
        for name, offsets in audit_policies.items()
    ]
    calls = [train_one.spawn(name, offsets, seed) for name, offsets, seed in configs]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "experiment": "S2-016",
        "description": "Learned routing audit for corrected gated MQAR",
        "source_commit": "working-tree",
        "dataset": "Synthetic MQAR with 16 key-value pairs and 8 late queries",
        "causal": True,
        "seq_len": 512,
        "vocab_size": 256,
        "num_kv_pairs": 16,
        "num_queries": 8,
        "key_value_assignment": "independent random permutation per example",
        "thought_hops": 4,
        "route_mode": "learned_audited",
        "random_exact_baseline_percent": 2.5,
        "training_steps": 6000,
        "validation_batches": 50,
        "seeds": seeds,
        "k": 8,
        "offset_policies": audit_policies,
        "update_mode": "learned_gate_full",
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = pathlib.Path("season2/results/s2_016_routing_audit_mqar.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
