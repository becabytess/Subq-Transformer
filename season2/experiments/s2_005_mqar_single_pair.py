"""S2-005: One-pair/one-query MQAR learnability gate.

This removes multi-pair interference from MQAR. Each example contains one
random key-value mapping and one late query. The value is placed 32, 128, or
256 positions before the query. The dense two-layer control should approach
perfect recall before sparse results are interpreted.

Run from the repository root:
    modal run season2/experiments/s2_005_mqar_single_pair.py
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-005-mqar-single-pair")


@app.function(image=image, gpu="A10G", timeout=3600)
def train_one(
    model_name: str,
    update_mode: str,
    thought_hops: int,
    d_mlp: int,
    seed: int,
    total_steps: int = 3000,
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
    query_marker = 1
    offsets = [0, 1, 2, 4, 8, 16, 32, 64]
    distances = {"local_32": 32, "medium_128": 128, "long_256": 256}
    min_query_target = 257

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
        value_pos = target_positions - value_distance
        key_pos = value_pos - 1
        rows = torch.arange(batch_size, device=device)
        x[rows, key_pos] = keys
        x[rows, value_pos] = values
        x[rows, target_positions - 1] = query_marker
        x[rows, target_positions] = keys
        y[rows, target_positions] = values
        return x, y, target_positions

    class SparseIterativeMQAR(nn.Module):
        def __init__(self, use_gru):
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

    condition_results = {}
    for condition_index, (condition_name, value_distance) in enumerate(distances.items()):
        condition_seed = seed + condition_index * 100000
        torch.manual_seed(condition_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(condition_seed)
        generator = torch.Generator(device=device)
        generator.manual_seed(condition_seed + 17)

        if update_mode == "sparse_residual":
            model = SparseIterativeMQAR(use_gru=False)
        elif update_mode == "sparse_gru":
            model = SparseIterativeMQAR(use_gru=True)
        elif update_mode == "dense2":
            model = DenseMQAR()
        else:
            raise ValueError(f"unknown update_mode: {update_mode}")

        model = model.to(device)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-4
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for _ in range(total_steps):
            model.train()
            x, y, _ = make_batch(value_distance, generator)
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

        elapsed = time.time() - start
        condition_results[condition_name] = {
            "value_distance": value_distance,
            "parameters": parameters,
            "target_parameters": 329088,
            "parameter_delta_vs_target": parameters - 329088,
            "steps": total_steps,
            "val_loss": round(val_loss / total, 6),
            "val_exact_retrieval_accuracy": round(100.0 * correct / total, 4),
            "validation_queries": total,
            "elapsed_s": round(elapsed, 3),
            "train_elapsed_s": round(train_elapsed, 3),
            "tokens_per_s_train": round(
                total_steps * batch_size * seq_len / train_elapsed, 2
            ),
            "peak_memory_mb": round(
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if device.type == "cuda"
                else 0.0,
                2,
            ),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        }
        del model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "experiment": "S2-005",
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
        "num_kv_pairs": 1,
        "num_queries": 1,
        "query_target_position_range": [257, 511],
        "conditions": condition_results,
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
                (f"sparse_residual_t2_s{seed}", "sparse_residual", 2, 512, seed),
                (f"sparse_residual_t4_s{seed}", "sparse_residual", 4, 512, seed),
                (f"sparse_residual_t8_s{seed}", "sparse_residual", 8, 512, seed),
                (f"sparse_gru_t1_s{seed}", "sparse_gru", 1, 126, seed),
                (f"sparse_gru_t2_s{seed}", "sparse_gru", 2, 126, seed),
                (f"sparse_gru_t4_s{seed}", "sparse_gru", 4, 126, seed),
                (f"sparse_gru_t8_s{seed}", "sparse_gru", 8, 126, seed),
                (f"dense2_matched_s{seed}", "dense2", 2, 127, seed),
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
        "experiment": "S2-005",
        "description": "One-pair/one-query MQAR learnability gate across controlled retrieval distances",
        "source_commit": "working-tree",
        "dataset": "Synthetic MQAR with one independently randomized key-value pair and one query",
        "causal": True,
        "seq_len": 512,
        "vocab_size": 256,
        "num_kv_pairs": 1,
        "num_queries": 1,
        "query_target_position_range": [257, 511],
        "value_distances": {"local_32": 32, "medium_128": 128, "long_256": 256},
        "offsets": [0, 1, 2, 4, 8, 16, 32, 64],
        "random_exact_baseline_percent": 2.5,
        "training_steps": 3000,
        "validation_batches": 100,
        "parameter_matching": {
            "target": 329088,
            "residual_d_mlp": 512,
            "gru_d_mlp": 126,
            "dense2_d_mlp": 127,
            "note": "GRU and two-layer dense are within 130 parameters of the residual target because integer MLP widths cannot match exactly.",
        },
        "seeds": [42, 1337, 2026],
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = pathlib.Path("season2/results/s2_005_mqar_single_pair.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
