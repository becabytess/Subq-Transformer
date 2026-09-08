"""S2-034: Dyck-4 Residual Shootout (Testing Attention Residual Hypotheses on Physical Depth).

Tests whether restoring the attention residual connection fixes the collapse of physical depth in SubQ:
1. subq_4layer_block_residual: 4 physical layers, 8 linear hops per layer, block residual state = residual + state (762,624 params).
   Direct ablation of S2-031: identical parameter count, only difference is state = residual + state.
2. subq_4layer_proj_residual: 4 physical layers, 8 linear hops per layer, block residual with out_proj state = residual + o(state) (828,160 params, exact match to Dense 4L).
3. subq_4layer_per_hop_residual: 4 physical layers, per-hop residual state = state + (1/sqrt(8)) * o(context) (828,160 params).

Evaluated across seeds 42, 1337, 2026 (9 parallel runs on Modal A10G).
Compared directly against existing reference baselines:
- dense_4layer: 94.76% (828,160 params)
- subq_1layer_final_mlp: 75.87% (218,496 params)
- subq_4layer_physical_replacement: 24.94% (762,624 params)
"""

import json
import pathlib
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-034-dyck4-residual-shootout")


@app.function(image=image, gpu="A10G", timeout=2400)
def train_one(
    model_name: str,
    architecture: str,
    seed: int,
    total_steps: int = 2000,
):
    import math
    import random
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 16
    seq_len = 256
    batch_size = 32
    d_model, d_mlp, n_heads = 128, 512, 4
    offsets = [0, 1, 2, 4, 8, 16, 63, 127, 128]

    open_to_close = {1: 2, 3: 4, 5: 6, 7: 8}
    open_brackets = [1, 3, 5, 7]

    def generate_dyck_sequence(target_len=256, max_depth=30, rng=None):
        _rand = rng.random if rng is not None else random.random
        _choice = rng.choice if rng is not None else random.choice

        tokens = []
        stack = []
        depth_at_pos = []

        while len(tokens) < target_len - 1:
            curr_depth = len(stack)
            p_close = 0.0 if curr_depth == 0 else (0.45 if curr_depth < max_depth else 0.90)

            if len(tokens) + curr_depth >= target_len - 1:
                p_close = 1.0

            if _rand() < p_close and curr_depth > 0:
                last_open = stack.pop()
                expected_close = open_to_close[last_open]
                tokens.append(expected_close)
                depth_at_pos.append(curr_depth)
            else:
                b = _choice(open_brackets)
                stack.append(b)
                tokens.append(b)
                depth_at_pos.append(len(stack))

        while stack and len(tokens) < target_len:
            last_open = stack.pop()
            tokens.append(open_to_close[last_open])
            depth_at_pos.append(len(stack) + 1)

        while len(tokens) < target_len:
            tokens.append(0)
            depth_at_pos.append(0)

        return tokens[:target_len], depth_at_pos[:target_len]

    def generate_dyck_batch(batch_size, rng):
        x = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        depths = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

        for b in range(batch_size):
            tokens, depth_list = generate_dyck_sequence(seq_len, max_depth=30, rng=rng)
            x[b] = torch.tensor(tokens, device=device)
            depths[b] = torch.tensor(depth_list, device=device)

            for t in range(seq_len - 1):
                next_tok = tokens[t + 1]
                if next_tok in [2, 4, 6, 8]:
                    y[b, t] = next_tok

        return x, y, depths

    targets = torch.zeros((seq_len, len(offsets)), dtype=torch.long)
    valid = torch.zeros((seq_len, len(offsets)), dtype=torch.bool)
    for pos in range(seq_len):
        for idx, offset in enumerate(offsets):
            source = pos - offset
            if source >= 0:
                targets[pos, idx] = source
                valid[pos, idx] = True

    # 1. SubQ 4-Layer Block Residual (state = residual + state, 762k params)
    class SubQ4LayerBlockResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "ln_attn": nn.LayerNorm(d_model),
                    "q": nn.Linear(d_model, d_model, bias=False),
                    "k": nn.Linear(d_model, d_model, bias=False),
                    "v": nn.Linear(d_model, d_model, bias=False),
                    "ln_mlp": nn.LayerNorm(d_model),
                    "mlp": nn.Sequential(
                        nn.Linear(d_model, d_mlp),
                        nn.GELU(),
                        nn.Linear(d_mlp, d_model),
                    ),
                }) for _ in range(4)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.register_buffer("targets", targets)
            self.register_buffer("valid", valid)

        def forward(self, idx):
            bsz, length = idx.shape
            state = self.tok(idx) + self.pos(torch.arange(length, device=idx.device).unsqueeze(0))
            indices = self.targets[:length]
            valid_mask = self.valid[:length].view(1, 1, length, len(offsets))
            scale = 1.0 / math.sqrt(d_model // n_heads)
            for layer in self.layers:
                residual = state
                for _ in range(8):
                    z = layer["ln_attn"](state)
                    q = layer["q"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                    k = layer["k"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                    v = layer["v"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                    scores = (q.unsqueeze(3) * k[:, :, indices, :]).sum(-1) * scale
                    weights = F.softmax(scores.masked_fill(~valid_mask, -1e4), dim=-1)
                    context = (weights.unsqueeze(-1) * v[:, :, indices, :]).sum(3)
                    state = context.transpose(1, 2).contiguous().view(bsz, length, d_model)
                state = residual + state  # Block residual
                state = state + layer["mlp"](layer["ln_mlp"](state))
            return self.head(self.ln_f(state))

    # 2. SubQ 4-Layer Proj Residual (state = residual + out_proj(state), 828k params)
    class SubQ4LayerProjResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "ln_attn": nn.LayerNorm(d_model),
                    "q": nn.Linear(d_model, d_model, bias=False),
                    "k": nn.Linear(d_model, d_model, bias=False),
                    "v": nn.Linear(d_model, d_model, bias=False),
                    "o": nn.Linear(d_model, d_model, bias=False),
                    "ln_mlp": nn.LayerNorm(d_model),
                    "mlp": nn.Sequential(
                        nn.Linear(d_model, d_mlp),
                        nn.GELU(),
                        nn.Linear(d_mlp, d_model),
                    ),
                }) for _ in range(4)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.register_buffer("targets", targets)
            self.register_buffer("valid", valid)

        def forward(self, idx):
            bsz, length = idx.shape
            state = self.tok(idx) + self.pos(torch.arange(length, device=idx.device).unsqueeze(0))
            indices = self.targets[:length]
            valid_mask = self.valid[:length].view(1, 1, length, len(offsets))
            scale = 1.0 / math.sqrt(d_model // n_heads)
            for layer in self.layers:
                residual = state
                for _ in range(8):
                    z = layer["ln_attn"](state)
                    q = layer["q"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                    k = layer["k"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                    v = layer["v"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                    scores = (q.unsqueeze(3) * k[:, :, indices, :]).sum(-1) * scale
                    weights = F.softmax(scores.masked_fill(~valid_mask, -1e4), dim=-1)
                    context = (weights.unsqueeze(-1) * v[:, :, indices, :]).sum(3)
                    state = context.transpose(1, 2).contiguous().view(bsz, length, d_model)
                state = residual + layer["o"](state)  # Block residual + out_proj
                state = state + layer["mlp"](layer["ln_mlp"](state))
            return self.head(self.ln_f(state))

    # 3. SubQ 4-Layer Per-Hop Residual (state = state + 1/sqrt(T) * out_proj(context), 828k params)
    class SubQ4LayerPerHopResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "ln_attn": nn.LayerNorm(d_model),
                    "q": nn.Linear(d_model, d_model, bias=False),
                    "k": nn.Linear(d_model, d_model, bias=False),
                    "v": nn.Linear(d_model, d_model, bias=False),
                    "o": nn.Linear(d_model, d_model, bias=False),
                    "ln_mlp": nn.LayerNorm(d_model),
                    "mlp": nn.Sequential(
                        nn.Linear(d_model, d_mlp),
                        nn.GELU(),
                        nn.Linear(d_mlp, d_model),
                    ),
                }) for _ in range(4)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.register_buffer("targets", targets)
            self.register_buffer("valid", valid)

        def forward(self, idx):
            bsz, length = idx.shape
            state = self.tok(idx) + self.pos(torch.arange(length, device=idx.device).unsqueeze(0))
            indices = self.targets[:length]
            valid_mask = self.valid[:length].view(1, 1, length, len(offsets))
            scale = 1.0 / math.sqrt(d_model // n_heads)
            inv_sqrt_T = 1.0 / math.sqrt(8.0)
            for layer in self.layers:
                for _ in range(8):
                    z = layer["ln_attn"](state)
                    q = layer["q"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                    k = layer["k"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                    v = layer["v"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                    scores = (q.unsqueeze(3) * k[:, :, indices, :]).sum(-1) * scale
                    weights = F.softmax(scores.masked_fill(~valid_mask, -1e4), dim=-1)
                    context = (weights.unsqueeze(-1) * v[:, :, indices, :]).sum(3)
                    msg = layer["o"](context.transpose(1, 2).contiguous().view(bsz, length, d_model))
                    state = state + inv_sqrt_T * msg  # Per-hop residual
                state = state + layer["mlp"](layer["ln_mlp"](state))
            return self.head(self.ln_f(state))

    if architecture == "subq_4layer_block_residual":
        model = SubQ4LayerBlockResidual().to(device)
    elif architecture == "subq_4layer_proj_residual":
        model = SubQ4LayerProjResidual().to(device)
    elif architecture == "subq_4layer_per_hop_residual":
        model = SubQ4LayerPerHopResidual().to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_rng = random.Random(seed + 1000)
    start = time.time()

    for step in range(total_steps):
        model.train()
        x, y, _ = generate_dyck_batch(batch_size, train_rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            loss = F.cross_entropy(
                model(x).reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    model.eval()
    val_rng = random.Random(seed + 5000)
    tiers = {
        "tier1_shallow_1_5": [1, 6],
        "tier2_medium_6_15": [6, 16],
        "tier3_deep_16_30": [16, 100],
    }
    tier_stats = {t: [0, 0] for t in tiers}
    total_correct = total_eval = 0

    with torch.no_grad():
        for _ in range(50):
            x, y, depths = generate_dyck_batch(batch_size, val_rng)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            preds = logits.argmax(-1)
            mask = y != -100
            correct_mask = (preds == y) & mask

            total_correct += correct_mask.sum().item()
            total_eval += mask.sum().item()

            for tname, (d_lo, d_hi) in tiers.items():
                tier_pos = mask & (depths >= d_lo) & (depths < d_hi)
                tier_stats[tname][0] += ((preds == y) & tier_pos).sum().item()
                tier_stats[tname][1] += tier_pos.sum().item()

    return {
        "model": model_name,
        "architecture": architecture,
        "seed": seed,
        "parameters": parameters,
        "steps": total_steps,
        "overall_accuracy": round(100.0 * total_correct / total_eval, 4),
        "tier_accuracy": {
            tname: round(100.0 * hits / count, 4) if count > 0 else 0.0
            for tname, (hits, count) in tier_stats.items()
        },
        "total_evaluated_tokens": total_eval,
        "elapsed_s": round(time.time() - start, 3),
    }


@app.local_entrypoint()
def main():
    seeds = [42, 1337, 2026]
    architectures = [
        "subq_4layer_block_residual",
        "subq_4layer_proj_residual",
        "subq_4layer_per_hop_residual",
    ]
    configs = [
        (f"{arch}_s{seed}", arch, seed)
        for arch in architectures
        for seed in seeds
    ]

    print(f"Launching {len(configs)} S2-034 Dyck-4 residual jobs on Modal in parallel...", flush=True)
    calls = [
        train_one.spawn(name, arch, seed)
        for name, arch, seed in configs
    ]

    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "experiment": "S2-034",
        "description": "Dyck-4 Residual Shootout: Testing Block Residual vs Proj Residual vs Per-Hop Residual on 4-layer SubQ",
        "task": "Dyck-4 bracket completion across nesting depths 1 to 30+",
        "seeds": seeds,
        "training_steps": 2000,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_baselines": {
            "dense_4layer": 94.76,
            "subq_1layer_final_mlp": 75.87,
            "subq_4layer_physical_replacement": 24.94,
        },
        "results": results,
    }
    path = pathlib.Path("season2/results/s2_034_dyck4_residual_shootout.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {path}", flush=True)


if __name__ == "__main__":
    main()
