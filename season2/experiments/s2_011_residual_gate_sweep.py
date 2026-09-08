"""S2-011: Residual mixing coefficient and learned-gate sweep.

This keeps the four-hop distance-256 route and transformed message fixed. It
changes only how the message is written into the receiving state.

Run from the repository root:
    modal run season2/experiments/s2_011_residual_gate_sweep.py
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-011-residual-gate-sweep")


@app.function(image=image, gpu="A10G", timeout=3600)
def train_one(
    model_name: str,
    update_mode: str,
    alpha: float,
    seed: int,
    total_steps: int = 6000,
    seq_len: int = 512,
    batch_size: int = 32,
    d_model: int = 128,
    validation_batches: int = 100,
    probe_batches: int = 40,
    probe_steps: int = 200,
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
    value_classes = 40
    value_distance = 256
    thought_hops = 4
    offsets = [0, 1, 2, 4, 8, 16, 32, 64]
    max_offset_index = offsets.index(64)
    min_query_target = value_distance + 1

    class ResidualGateModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.ln_attn = nn.LayerNorm(d_model)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            if update_mode == "learned_gate":
                self.gate = nn.Linear(2 * d_model, 1)
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)

            target_indices = torch.zeros((seq_len, len(offsets)), dtype=torch.long)
            for position in range(seq_len):
                for offset_idx, offset in enumerate(offsets):
                    target = position - offset
                    if target >= 0:
                        target_indices[position, offset_idx] = target
            self.register_buffer("target_indices", target_indices)

        def forward(self, idx, capture_states=False):
            bsz, length = idx.shape
            positions = torch.arange(length, device=idx.device).unsqueeze(0)
            state = self.tok_emb(idx) + self.pos_emb(positions)
            forced_indices = self.target_indices[:length, max_offset_index]
            captured = []

            for _ in range(thought_hops):
                source_state = state[:, forced_indices, :]
                message = self.out_proj(self.v_proj(self.ln_attn(source_state)))

                if update_mode == "replacement":
                    state = message
                elif update_mode == "additive":
                    state = state + alpha * message
                elif update_mode == "convex":
                    state = (1.0 - alpha) * state + alpha * message
                elif update_mode == "learned_gate":
                    gate_input = torch.cat(
                        [self.ln_attn(state), self.ln_attn(message)], dim=-1
                    )
                    gate = torch.sigmoid(self.gate(gate_input))
                    state = (1.0 - gate) * state + gate * message
                else:
                    raise ValueError(f"unknown update mode: {update_mode}")

                if capture_states:
                    captured.append(state)

            logits = self.head(self.ln_f(state))
            if capture_states:
                return logits, captured
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

    def collect_probe_data(model, generator):
        model.eval()
        states_by_hop = [[] for _ in range(thought_hops)]
        labels = []
        with torch.no_grad():
            for _ in range(probe_batches):
                x, y, target_positions = make_batch(generator)
                _, captured = model(x, capture_states=True)
                rows = torch.arange(batch_size, device=device)
                labels.append((y[rows, target_positions] - 50).long().cpu())
                for hop, state in enumerate(captured):
                    states_by_hop[hop].append(
                        state[rows, target_positions].float().cpu()
                    )
        return [torch.cat(items) for items in states_by_hop], torch.cat(labels)

    def train_probes(model, generator):
        train_states, train_labels = collect_probe_data(model, generator)
        eval_states, eval_labels = collect_probe_data(model, generator)
        results = []
        for hop, (train_x, eval_x) in enumerate(zip(train_states, eval_states), start=1):
            probe = nn.Linear(d_model, value_classes).to(device)
            optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
            train_x = train_x.to(device)
            train_y = train_labels.to(device)
            eval_x = eval_x.to(device)
            eval_y = eval_labels.to(device)
            for _ in range(probe_steps):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(probe(train_x), train_y)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                accuracy = (
                    probe(eval_x).argmax(dim=-1).eq(eval_y).float().mean().item() * 100.0
                )
            results.append({"hop": hop, "accuracy": round(accuracy, 4)})
        return results

    model = ResidualGateModel().to(device)
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

    probe_generator = torch.Generator(device=device)
    probe_generator.manual_seed(seed + value_distance * 20000 + 77)
    probe_accuracy = train_probes(model, probe_generator)
    elapsed = time.time() - start
    return {
        "experiment": "S2-011",
        "model": model_name,
        "update_mode": update_mode,
        "alpha": alpha,
        "route_mode": "forced_64",
        "thought_hops": thought_hops,
        "value_distance": value_distance,
        "offsets": offsets,
        "offset_horizon": max(offsets),
        "seed": seed,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "d_model": d_model,
        "parameters": parameters,
        "steps": total_steps,
        "checkpoints": checkpoints,
        "intermediate_linear_probe": probe_accuracy,
        "probe_batches": probe_batches,
        "probe_steps": probe_steps,
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
    conditions = [
        ("replacement", 1.0),
        ("additive", 0.125),
        ("additive", 0.25),
        ("additive", 0.5),
        ("additive", 0.75),
        ("additive", 1.0),
        ("convex", 0.25),
        ("convex", 0.5),
        ("convex", 0.75),
        ("learned_gate", 0.0),
    ]
    configs = [
        (f"{mode}_a{str(alpha).replace('.', 'p')}_d256_t4_s{seed}", mode, alpha, seed)
        for seed in seeds
        for mode, alpha in conditions
    ]

    calls = [
        train_one.spawn(name, mode, alpha, seed)
        for name, mode, alpha, seed in configs
    ]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "experiment": "S2-011",
        "description": "Residual mixing coefficient and learned-gate sweep",
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
        "message_transform": "LayerNorm -> tied value projection -> tied output projection",
        "training_steps": 6000,
        "validation_batches": 100,
        "probe_batches": 40,
        "probe_steps": 200,
        "seeds": seeds,
        "conditions": [
            {"update_mode": mode, "alpha": alpha}
            for mode, alpha in conditions
        ],
        "mode_definitions": {
            "replacement": "state = message",
            "additive": "state = state + alpha * message",
            "convex": "state = (1-alpha) * state + alpha * message",
            "learned_gate": "state = (1-gate) * state + gate * message, with a learned scalar gate per token",
        },
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = pathlib.Path("season2/results/s2_011_residual_gate_sweep.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
