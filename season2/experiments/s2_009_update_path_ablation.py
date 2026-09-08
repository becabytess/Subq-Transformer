"""S2-009: Diagnose four-hop signal preservation under a forced route.

The offset route is fixed to four successive 64-token jumps at distance 256.
Only the state-update mechanism changes. This compares the current nonlinear
residual update with simpler propagation updates and the established
inside-loop accumulator GRU.

Run from the repository root:
    modal run season2/experiments/s2_009_update_path_ablation.py
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-009-update-path-ablation")


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
    value_distance = 256
    thought_hops = 4
    offsets = [0, 1, 2, 4, 8, 16, 32, 64]
    max_offset_index = offsets.index(64)
    min_query_target = value_distance + 1
    target_parameters = 329088

    class SparseUpdateDiagnostic(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.ln_attn = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            if update_mode == "gru":
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
            forced_indices = indices[:, max_offset_index]
            residual_scale = 1.0 / math.sqrt(thought_hops)

            for _ in range(thought_hops):
                normalized = self.ln_attn(state)
                q = self.q_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
                k = self.k_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
                v = self.v_proj(normalized).view(bsz, length, n_heads, -1).transpose(1, 2)
                del q, k
                source_state = state[:, forced_indices, :]
                source_value = v[:, :, forced_indices, :]
                source_value = source_value.transpose(1, 2).contiguous().view(
                    bsz, length, d_model
                )

                if update_mode == "raw_copy":
                    state = source_state
                elif update_mode == "v_residual":
                    state = state + residual_scale * source_value
                else:
                    message = self.out_proj(source_value)
                    if update_mode == "gru":
                        flat_state = state.reshape(bsz * length, d_model)
                        flat_message = message.reshape(bsz * length, d_model)
                        updated = self.accumulator(flat_message, flat_state)
                        state = state + residual_scale * (
                            updated.view(bsz, length, d_model) - state
                        )
                    else:
                        state = state + residual_scale * message

                if update_mode in {"full", "gru"}:
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

    model = SparseUpdateDiagnostic().to(device)
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
        "experiment": "S2-009",
        "model": model_name,
        "update_mode": update_mode,
        "route_mode": "forced_64",
        "thought_hops": thought_hops,
        "value_distance": value_distance,
        "offsets": offsets,
        "offset_horizon": max(offsets),
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

    seeds = [42, 1337, 2026]
    update_modes = ["full", "no_mlp", "v_residual", "raw_copy", "gru"]
    configs = [
        (f"{update_mode}_d256_t4_s{seed}", update_mode, seed)
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
        "experiment": "S2-009",
        "description": "Four-hop distance-256 update-path preservation ablation",
        "source_commit": "working-tree",
        "dataset": "Synthetic one-pair randomized-position propagation task",
        "causal": True,
        "seq_len": 512,
        "query_target_position_range": [257, 511],
        "value_distance": 256,
        "num_kv_pairs": 1,
        "num_queries": 1,
        "random_exact_baseline_percent": 2.5,
        "offsets": [0, 1, 2, 4, 8, 16, 32, 64],
        "offset_horizon": 64,
        "thought_hops": 4,
        "route_mode": "forced_64",
        "training_steps": 6000,
        "validation_batches": 100,
        "seeds": seeds,
        "parameter_target": 329088,
        "update_modes": update_modes,
        "mode_definitions": {
            "full": "Forced offset-64 route, output projection, residual update, and nonlinear MLP",
            "no_mlp": "Forced offset-64 route, output projection, and residual update; no MLP",
            "v_residual": "Forced offset-64 route, value projection, and residual update; no output projection or MLP",
            "raw_copy": "Forced offset-64 route with direct source-state replacement",
            "gru": "Established accumulator GRU update followed by the MLP",
        },
        "parameter_matching": {
            "full_width": 512,
            "gru_width": 512,
            "note": "The GRU adds parameters; this is an update-mechanism diagnostic, not a matched-capacity comparison.",
        },
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = pathlib.Path("season2/results/s2_009_update_path_ablation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
