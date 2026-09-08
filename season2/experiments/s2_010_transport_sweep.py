"""S2-010: Fixed-route transport and intermediate-probe sweep.

This keeps the four-hop distance-256 route fixed and removes the MLP. It
isolates repeated representation transport by comparing raw copying, value
and output projections, tied versus untied projections, and residual versus
replacement transport. A linear probe is trained on each intermediate state
to separate transport loss from final-decoder failure.

Run from the repository root:
    modal run season2/experiments/s2_010_transport_sweep.py
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-010-transport-sweep")


@app.function(image=image, gpu="A10G", timeout=3600)
def train_one(
    model_name: str,
    transport_mode: str,
    seed: int,
    total_steps: int = 6000,
    seq_len: int = 512,
    batch_size: int = 32,
    d_model: int = 128,
    validation_batches: int = 100,
    probe_batches: int = 50,
    probe_steps: int = 300,
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

    class TransportModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.ln_attn = nn.LayerNorm(d_model)
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            if transport_mode in {"v_replace", "vo_replace_tied", "vo_residual"}:
                self.v_proj = nn.Linear(d_model, d_model, bias=False)
            if transport_mode in {"out_replace", "vo_replace_tied", "vo_residual"}:
                self.out_proj = nn.Linear(d_model, d_model, bias=False)
            if transport_mode == "vo_replace_untied":
                self.v_proj = nn.ModuleList(
                    [nn.Linear(d_model, d_model, bias=False) for _ in range(thought_hops)]
                )
                self.out_proj = nn.ModuleList(
                    [nn.Linear(d_model, d_model, bias=False) for _ in range(thought_hops)]
                )

            target_indices = torch.zeros((seq_len, len(offsets)), dtype=torch.long)
            for position in range(seq_len):
                for offset_idx, offset in enumerate(offsets):
                    target = position - offset
                    if target >= 0:
                        target_indices[position, offset_idx] = target
            self.register_buffer("target_indices", target_indices)

        def _project(self, source_state, hop):
            if transport_mode == "raw_copy":
                return source_state
            if transport_mode == "v_replace":
                return self.v_proj(self.ln_attn(source_state))
            if transport_mode == "out_replace":
                return self.out_proj(source_state)
            if transport_mode == "vo_replace_tied":
                value = self.v_proj(self.ln_attn(source_state))
                return self.out_proj(value)
            if transport_mode == "vo_replace_untied":
                value = self.v_proj[hop](self.ln_attn(source_state))
                return self.out_proj[hop](value)
            if transport_mode == "vo_residual":
                value = self.v_proj(self.ln_attn(source_state))
                return self.out_proj(value)
            raise ValueError(f"unknown transport mode: {transport_mode}")

        def forward(self, idx, capture_states=False):
            bsz, length = idx.shape
            positions = torch.arange(length, device=idx.device).unsqueeze(0)
            state = self.tok_emb(idx) + self.pos_emb(positions)
            forced_indices = self.target_indices[:length, max_offset_index]
            residual_scale = 1.0 / math.sqrt(thought_hops)
            captured = []

            for hop in range(thought_hops):
                source_state = state[:, forced_indices, :]
                transported = self._project(source_state, hop)
                if transport_mode == "vo_residual":
                    state = state + residual_scale * transported
                else:
                    state = transported
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

    model = TransportModel().to(device)
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
        "experiment": "S2-010",
        "model": model_name,
        "transport_mode": transport_mode,
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
    transport_modes = [
        "raw_copy",
        "v_replace",
        "out_replace",
        "vo_replace_tied",
        "vo_replace_untied",
        "vo_residual",
    ]
    configs = [
        (f"{transport_mode}_d256_t4_s{seed}", transport_mode, seed)
        for seed in seeds
        for transport_mode in transport_modes
    ]

    calls = [
        train_one.spawn(name, transport_mode, seed)
        for name, transport_mode, seed in configs
    ]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "experiment": "S2-010",
        "description": "Fixed-route transport and intermediate linear-probe sweep",
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
        "probe_batches": 50,
        "probe_steps": 300,
        "seeds": seeds,
        "transport_modes": transport_modes,
        "mode_definitions": {
            "raw_copy": "Direct source-state replacement at every hop",
            "v_replace": "LayerNorm, value projection, then replacement",
            "out_replace": "Output projection of source state, then replacement",
            "vo_replace_tied": "LayerNorm, value projection, output projection, then replacement with tied weights",
            "vo_replace_untied": "LayerNorm, per-hop value and output projections, then replacement",
            "vo_residual": "LayerNorm, value projection, output projection, then residual interpolation",
        },
        "probe_definition": "A fresh linear classifier trained on frozen target-position states after each hop to predict the randomized value class 50-89.",
        "capacity_note": "This is a transport diagnostic; projection modes have different parameter counts and are not a matched-capacity benchmark.",
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = pathlib.Path("season2/results/s2_010_transport_sweep.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
