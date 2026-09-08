"""S2-002: Parameter-matched accumulator-GRU ablation.

This follows S2-001, but removes the main confound: the inside-the-loop
accumulator GRU is given a narrower MLP so its total parameter count matches
the residual sparse model. Three seeds are run for every variant, with one
independent Modal invocation per model/seed pair.

Run from the repository root:
    modal run season2/experiments/s2_002_parameter_matched_gru.py
"""

import json
import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy", "requests"
)
app = modal.App("season2-s2-002-parameter-matched-gru")


@app.function(image=image, gpu="A10G", timeout=3600)
def train_one(
    model_name: str,
    update_mode: str,
    thought_hops: int,
    d_mlp: int,
    seed: int,
    total_steps: int = 2000,
    seq_len: int = 256,
    batch_size: int = 32,
    d_model: int = 128,
    n_heads: int = 4,
):
    import math
    import time
    import urllib.request

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    vocab_size = len(chars)
    offsets = [0, 1, 2, 4, 8, 16, 32, 64]

    def get_batch(source, batch_seed):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(batch_seed)
        starts = torch.randint(
            0, len(source) - seq_len - 1, (batch_size,), generator=generator
        )
        x = torch.stack([source[int(i) : int(i) + seq_len] for i in starts])
        y = torch.stack([source[int(i) + 1 : int(i) + seq_len + 1] for i in starts])
        return x.to(device), y.to(device)

    class SparseIterativeLM(nn.Module):
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
            for i in range(seq_len):
                for j, offset in enumerate(offsets):
                    target = i - offset
                    if target >= 0:
                        target_indices[i, j] = target
                        valid[i, j] = True
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
                    state = state + residual_scale * (updated.view(bsz, length, d_model) - state)
                else:
                    state = state + residual_scale * message
                state = state + residual_scale * self.mlp(self.ln_mlp(state))

            return self.head(self.ln_f(state))

    class DenseOneLayerLM(nn.Module):
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
        model = SparseIterativeLM(use_gru=False)
    elif update_mode == "sparse_gru":
        model = SparseIterativeLM(use_gru=True)
    elif update_mode == "dense":
        model = DenseOneLayerLM()
    else:
        raise ValueError(f"unknown update_mode: {update_mode}")
    model = model.to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    start = time.time()

    for step in range(1, total_steps + 1):
        model.train()
        x, y = get_batch(train_data, step * 1000 + 42)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    model.eval()
    val_loss = 0.0
    val_tokens = 0
    with torch.no_grad():
        for eval_step in range(30):
            x, y = get_batch(val_data, eval_step * 5000 + 1337)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                batch_loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            val_loss += batch_loss.item() * y.numel()
            val_tokens += y.numel()

    elapsed = time.time() - start
    mean_val_loss = val_loss / val_tokens
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "experiment": "S2-002",
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
        "parameters": parameters,
        "target_parameters": 247424,
        "parameter_delta_vs_target": parameters - 247424,
        "steps": total_steps,
        "val_loss": round(mean_val_loss, 6),
        "val_ppl": round(math.exp(mean_val_loss), 6),
        "elapsed_s": round(elapsed, 3),
        "tokens_per_s_train": round(total_steps * batch_size * seq_len / elapsed, 2),
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

    calls = [train_one.spawn(name, mode, hops, mlp, seed) for name, mode, hops, mlp, seed in configs]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "experiment": "S2-002",
        "description": "Three-seed parameter-matched comparison of residual sparse refinement and inside-loop accumulator GRU",
        "source_commit": "working-tree",
        "dataset": "TinyShakespeare character language modeling",
        "causal": True,
        "parameter_matching": {
            "target": 247424,
            "residual_d_mlp": 512,
            "gru_d_mlp": 126,
            "note": "GRU variant is within 130 parameters of target because the integer MLP width cannot match exactly.",
        },
        "seeds": [42, 1337, 2026],
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = pathlib.Path("season2/results/s2_002_parameter_matched_gru.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
