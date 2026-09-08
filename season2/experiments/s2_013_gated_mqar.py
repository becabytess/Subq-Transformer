"""S2-013: Corrected multi-pair MQAR with gated state writes.

This transfers the fixed-route write findings to corrected multi-pair MQAR.
Values are independently randomized per example, and the sparse router is
learned over the standard dyadic offsets. The comparison varies only the
state-write rule around the shared nonlinear message path.

Run from the repository root:
    modal run season2/experiments/s2_013_gated_mqar.py
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-013-gated-mqar")


@app.function(image=image, gpu="A10G", timeout=3600)
def train_one(
    model_name: str,
    update_mode: str,
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
    offsets = [0, 1, 2, 4, 8, 16, 32, 64]
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

        return x, y, query_distances

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

        def forward(self, idx, capture_gates=False):
            bsz, length = idx.shape
            positions = torch.arange(length, device=idx.device).unsqueeze(0)
            state = self.tok_emb(idx) + self.pos_emb(positions)
            indices = self.target_indices[:length]
            valid = self.valid[:length].view(1, 1, length, len(offsets))
            scale = 1.0 / math.sqrt(d_model // n_heads)
            residual_scale = 1.0 / math.sqrt(thought_hops)
            gates = []

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
                context = (weights.unsqueeze(-1) * v_candidates).sum(3)
                context = context.transpose(1, 2).contiguous().view(bsz, length, d_model)
                message = self.out_proj(context)

                if update_mode == "residual_full":
                    state = state + residual_scale * message
                elif update_mode == "convex_full":
                    state = 0.25 * state + 0.75 * message
                elif update_mode in {"learned_gate_full", "learned_gate_no_mlp"}:
                    gate_input = torch.cat(
                        [self.ln_attn(state), self.ln_attn(message)], dim=-1
                    )
                    gate = torch.sigmoid(self.gate_proj(gate_input))
                    state = (1.0 - gate) * state + gate * message
                    if capture_gates:
                        gates.append(gate.squeeze(-1))
                else:
                    raise ValueError(f"unknown update mode: {update_mode}")

                if update_mode != "learned_gate_no_mlp":
                    state = state + residual_scale * self.mlp(self.ln_mlp(state))

            logits = self.head(self.ln_f(state))
            if capture_gates:
                return logits, gates
            return logits

    model = GatedSparseMQAR().to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + thought_hops * 1000 + 17)
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
        gate_count = 0

        with torch.no_grad():
            for _ in range(validation_batches):
                x, y, query_distances = make_batch(eval_generator)
                with torch.amp.autocast(
                    "cuda", dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    if update_mode in {"learned_gate_full", "learned_gate_no_mlp"}:
                        logits, gates = model(x, capture_gates=True)
                    else:
                        logits = model(x)
                        gates = []
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
                if gates:
                    rows = torch.arange(batch_size, device=device)
                    for hop, gate in enumerate(gates):
                        gate_sums[hop] += gate.float().mean().item() * batch_size
                        gate_query_sums[hop] += gate[rows[:, None], torch.tensor(query_value_positions, device=device)[None, :]].float().mean().item() * batch_size
                    gate_count += batch_size

        for stats in distance_stats.values():
            stats["accuracy"] = round(100.0 * stats["correct"] / stats["total"], 4)
        result = {
            "accuracy": round(100.0 * correct_queries / total_queries, 4),
            "val_loss": round(val_loss / val_tokens, 6),
            "queries": total_queries,
            "mean_query_distance": round(distance_sum / total_queries, 3),
            "distance_accuracy": distance_stats,
        }
        if gate_count:
            result["gate_mean_all_positions_by_hop"] = [
                round(value / gate_count, 6) for value in gate_sums
            ]
            result["gate_mean_query_positions_by_hop"] = [
                round(value / gate_count, 6) for value in gate_query_sums
            ]
        return result

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
            eval_generator = torch.Generator(device=device)
            eval_generator.manual_seed(seed + 100000 + step)
            checkpoints.append({"step": step, **evaluate(eval_generator)})

    elapsed = time.time() - start
    return {
        "experiment": "S2-013",
        "model": model_name,
        "update_mode": update_mode,
        "route_mode": "learned",
        "thought_hops": thought_hops,
        "offsets": offsets,
        "offset_horizon": max(offsets),
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
    update_modes = [
        "residual_full",
        "convex_full",
        "learned_gate_full",
        "learned_gate_no_mlp",
    ]
    configs = [
        (f"{update_mode}_t4_s{seed}", update_mode, seed)
        for seed in seeds
        for update_mode in update_modes
    ]
    calls = [train_one.spawn(name, update_mode, seed) for name, update_mode, seed in configs]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "experiment": "S2-013",
        "description": "Corrected multi-pair MQAR comparison with gated state writes",
        "source_commit": "working-tree",
        "dataset": "Synthetic MQAR with 16 key-value pairs and 8 late queries",
        "causal": True,
        "seq_len": 512,
        "vocab_size": 256,
        "num_kv_pairs": 16,
        "num_queries": 8,
        "key_value_assignment": "independent random permutation per example",
        "offsets": [0, 1, 2, 4, 8, 16, 32, 64],
        "thought_hops": 4,
        "route_mode": "learned",
        "random_exact_baseline_percent": 2.5,
        "training_steps": 6000,
        "validation_batches": 50,
        "seeds": seeds,
        "update_modes": update_modes,
        "mode_definitions": {
            "residual_full": "state + 1/sqrt(T) * message, followed by the standard MLP residual",
            "convex_full": "0.25 * state + 0.75 * message, followed by the standard MLP residual",
            "learned_gate_full": "learned scalar write gate, followed by the standard MLP residual",
            "learned_gate_no_mlp": "learned scalar write gate without the MLP",
        },
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = pathlib.Path("season2/results/s2_013_gated_mqar.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
