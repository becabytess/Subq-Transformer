"""S2-003: Transfer the local-refinement comparison to exact MQAR recall.

The task uses 16 random key-value pairs followed by 8 late queries in a
causal sequence of length 512. Each model/seed pair runs as an independent
Modal invocation so the comparison gets one GPU per job.

Run from the repository root:
    modal run season2/experiments/s2_003_mqar_transfer.py
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-003-mqar-transfer")


@app.function(image=image, gpu="A10G", timeout=3600)
def train_one(
    model_name: str,
    update_mode: str,
    thought_hops: int,
    d_mlp: int,
    seed: int,
    total_steps: int = 1500,
    seq_len: int = 512,
    batch_size: int = 32,
    d_model: int = 128,
    n_heads: int = 4,
    num_kv_pairs: int = 16,
    num_queries: int = 8,
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
    offsets = [0, 1, 2, 4, 8, 16, 32, 64]
    start_q_pos = seq_len - (num_queries * 2) - 2
    query_value_positions = [start_q_pos + (q_idx * 2) + 1 for q_idx in range(num_queries)]

    def generate_mqar_batch():
        """Generate random dictionaries and late queries for exact recall."""
        x = torch.randint(100, 255, (batch_size, seq_len), device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        query_distances = torch.empty(
            (batch_size, num_queries), dtype=torch.long, device=device
        )

        for b in range(batch_size):
            # Values are independently permuted for every example. A fixed
            # key-to-value arithmetic rule would turn recall into lookup.
            keys = torch.randperm(40, device=device)[:num_kv_pairs] + 10
            values = torch.randperm(40, device=device)[:num_kv_pairs] + 50
            kv_positions = torch.randperm(device=device, n=350)[: num_kv_pairs * 2].sort().values
            for pair_idx in range(num_kv_pairs):
                key_pos = int(kv_positions[pair_idx * 2].item())
                value_pos = key_pos + 1
                x[b, key_pos] = keys[pair_idx]
                x[b, value_pos] = values[pair_idx]

            query_indices = torch.randperm(device=device, n=num_kv_pairs)[:num_queries]
            for query_idx, key_idx in enumerate(query_indices):
                query_pos = start_q_pos + (query_idx * 2)
                x[b, query_pos] = query_marker
                x[b, query_pos + 1] = keys[key_idx]
                y[b, query_pos + 1] = values[key_idx]
                key_position = int(kv_positions[int(key_idx.item()) * 2].item())
                query_distances[b, query_idx] = query_pos + 1 - key_position

        return x, y, query_distances

    class SparseIterativeMQAR(nn.Module):
        def __init__(self, use_gru: bool):
            super().__init__()
            self.use_gru = use_gru
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.ln_attn = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            if use_gru:
                self.accumulator = nn.GRUCell(d_model, d_model)
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
                scores = (q.unsqueeze(3) * k_candidates).sum(-1) * scale
                scores = scores.masked_fill(~valid, -1e4)
                weights = F.softmax(scores, dim=-1)
                context = (weights.unsqueeze(-1) * v_candidates).sum(3)
                context = context.transpose(1, 2).contiguous().view(bsz, length, d_model)
                message = self.out_proj(context)

                if self.use_gru:
                    flat_state = state.reshape(bsz * length, d_model)
                    flat_message = message.reshape(bsz * length, d_model)
                    updated = self.accumulator(flat_message, flat_state)
                    state = state + residual_scale * (
                        updated.view(bsz, length, d_model) - state
                    )
                else:
                    state = state + residual_scale * message
                state = state + residual_scale * self.mlp(self.ln_mlp(state))

            return self.head(self.ln_f(state))

    class DenseOneLayerMQAR(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.ln1 = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.o = nn.Linear(d_model, d_model, bias=False)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model),
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
            normalized = self.ln1(state)
            q = self.q(normalized).view(bsz, length, n_heads, head_dim).transpose(1, 2)
            k = self.k(normalized).view(bsz, length, n_heads, head_dim).transpose(1, 2)
            v = self.v(normalized).view(bsz, length, n_heads, head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            scores = scores.masked_fill(~self.causal_mask[:length, :length], -1e4)
            weights = F.softmax(scores, dim=-1)
            context = (weights @ v).transpose(1, 2).contiguous().view(bsz, length, d_model)
            state = state + self.o(context)
            state = state + self.mlp(self.ln2(state))
            return self.head(self.ln_f(state))

    if update_mode == "sparse_residual":
        model = SparseIterativeMQAR(use_gru=False)
    elif update_mode == "sparse_gru":
        model = SparseIterativeMQAR(use_gru=True)
    elif update_mode == "dense":
        model = DenseOneLayerMQAR()
    else:
        raise ValueError(f"unknown update_mode: {update_mode}")

    model = model.to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target_parameters = 329088
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, total_steps + 1):
        model.train()
        x, y, _ = generate_mqar_batch()
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

    train_elapsed = time.time() - start
    model.eval()
    total_queries = 0
    correct_queries = 0
    val_loss = 0.0
    val_loss_tokens = 0
    distance_bins = {
        "128-255": [128, 256],
        "256-383": [256, 384],
        "384-511": [384, 512],
    }
    distance_stats = {
        name: {"correct": 0, "total": 0} for name in distance_bins
    }
    distance_sum = 0

    with torch.no_grad():
        for eval_step in range(50):
            x, y, query_distances = generate_mqar_batch()
            with torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(x)
                batch_loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100
                )
            val_loss += batch_loss.item() * (y != -100).sum().item()
            val_loss_tokens += (y != -100).sum().item()

            query_logits = logits[:, query_value_positions, :]
            query_targets = y[:, query_value_positions]
            query_correct = query_logits.argmax(dim=-1).eq(query_targets)
            correct_queries += query_correct.sum().item()
            total_queries += query_targets.numel()
            distance_sum += query_distances.sum().item()
            for name, (lower, upper) in distance_bins.items():
                bin_mask = (query_distances >= lower) & (query_distances < upper)
                distance_stats[name]["correct"] += (query_correct & bin_mask).sum().item()
                distance_stats[name]["total"] += bin_mask.sum().item()

    for stats in distance_stats.values():
        stats["accuracy"] = round(100.0 * stats["correct"] / stats["total"], 4)
    elapsed = time.time() - start
    mean_val_loss = val_loss / val_loss_tokens
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "experiment": "S2-003",
        "model": model_name,
        "update_mode": update_mode,
        "thought_hops": thought_hops,
        "d_mlp": d_mlp,
        "offsets": offsets if update_mode.startswith("sparse") else None,
        "seed": seed,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "num_kv_pairs": num_kv_pairs,
        "num_queries": num_queries,
        "parameters": parameters,
        "target_parameters": target_parameters,
        "parameter_delta_vs_target": parameters - target_parameters,
        "steps": total_steps,
        "val_loss": round(mean_val_loss, 6),
        "val_exact_retrieval_accuracy": round(100.0 * correct_queries / total_queries, 4),
        "mean_query_distance": round(distance_sum / total_queries, 3),
        "distance_accuracy": distance_stats,
        "elapsed_s": round(elapsed, 3),
        "train_elapsed_s": round(train_elapsed, 3),
        "tokens_per_s_train": round(total_steps * batch_size * seq_len / train_elapsed, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }


@app.local_entrypoint()
def main():
    import pathlib
    import time

    configs = []
    for seed in (42, 1337, 2026):
        configs.extend(
            [
                (f"sparse_residual_t1_s{seed}", "sparse_residual", 1, 512, seed),
                (f"sparse_residual_t4_s{seed}", "sparse_residual", 4, 512, seed),
                (f"sparse_residual_t8_s{seed}", "sparse_residual", 8, 512, seed),
                (f"sparse_gru_t1_s{seed}", "sparse_gru", 1, 126, seed),
                (f"sparse_gru_t4_s{seed}", "sparse_gru", 4, 126, seed),
                (f"sparse_gru_t8_s{seed}", "sparse_gru", 8, 126, seed),
                (f"dense_1l_s{seed}", "dense", 1, 512, seed),
            ]
        )

    calls = [
        train_one.spawn(name, mode, hops, mlp, seed)
        for name, mode, hops, mlp, seed in configs
    ]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "experiment": "S2-003",
        "description": "Three-seed MQAR transfer comparison of residual sparse refinement and inside-loop accumulator GRU",
        "source_commit": "working-tree",
        "dataset": "Synthetic MQAR: 16 random key-value pairs and 8 late queries",
        "causal": True,
        "seq_len": 512,
        "vocab_size": 256,
        "num_kv_pairs": 16,
        "num_queries": 8,
        "key_value_assignment": "independent random permutation per example",
        "offsets": [0, 1, 2, 4, 8, 16, 32, 64],
        "parameter_matching": {
            "target": 329088,
            "residual_d_mlp": 512,
            "gru_d_mlp": 126,
            "note": "GRU variant is within 130 parameters of target because the integer MLP width cannot match exactly.",
        },
        "seeds": [42, 1337, 2026],
        "validation_batches": 50,
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = pathlib.Path("season2/results/s2_003_mqar_transfer.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
