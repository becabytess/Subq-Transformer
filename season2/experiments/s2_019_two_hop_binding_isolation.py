"""S2-019: isolate two-hop key/value binding from query retrieval.

Modes:
  learned: normal Q/K attention on both hops.
  force_hop1: force each value token to read its preceding key on hop 1.
  force_hop2: force the query to read the correct value token on hop 2.
  force_both: force both controls.
"""

import json

import modal


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season2-s2-019-binding-isolation")


@app.function(image=image, gpu="T4", timeout=3600)
def train_one(model_name: str, mode: str, offsets: list[int], seed: int, total_steps: int = 6000):
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

    seq_len, batch_size, d_model, n_heads = 256, 64, 128, 4
    vocab_size, key_low, key_high, value_low, value_high = 256, 10, 50, 50, 90
    target_position = 200
    value_offsets = [4, 8, 16, 32, 64, 96, 128, 160]
    value_positions = torch.tensor([target_position - x for x in value_offsets], device=device)
    offset1_index = offsets.index(1)
    value_offset_indices = torch.tensor([offsets.index(x) for x in value_offsets], device=device)

    def make_batch(generator):
        x = torch.randint(100, 255, (batch_size, seq_len), generator=generator, device=device)
        y = torch.empty(batch_size, dtype=torch.long, device=device)
        slot = torch.randint(0, len(value_offsets), (batch_size,), generator=generator, device=device)
        for row in range(batch_size):
            keys = torch.randperm(key_high - key_low, generator=generator, device=device)[: len(value_offsets)] + key_low
            values = torch.randperm(value_high - value_low, generator=generator, device=device)[: len(value_offsets)] + value_low
            for idx, offset in enumerate(value_offsets):
                value_pos = target_position - offset
                x[row, value_pos - 1] = keys[idx]
                x[row, value_pos] = values[idx]
            x[row, target_position] = keys[slot[row]]
            y[row] = values[slot[row]]
        return x, y, slot

    class Model(nn.Module):
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
            self.mlp = nn.Sequential(nn.Linear(d_model, 512), nn.GELU(), nn.Linear(512, d_model))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            targets = torch.zeros((seq_len, len(offsets)), dtype=torch.long)
            valid = torch.zeros((seq_len, len(offsets)), dtype=torch.bool)
            for pos in range(seq_len):
                for idx, offset in enumerate(offsets):
                    source = pos - offset
                    if source >= 0:
                        targets[pos, idx] = source
                        valid[pos, idx] = True
            self.register_buffer("targets", targets)
            self.register_buffer("valid", valid)

        def forward(self, idx, slot, capture=False):
            bsz, length = idx.shape
            pos = torch.arange(length, device=idx.device).unsqueeze(0)
            state = self.tok_emb(idx) + self.pos_emb(pos)
            indices = self.targets[:length]
            valid = self.valid[:length].view(1, 1, length, len(offsets))
            scale = 1.0 / math.sqrt(d_model // n_heads)
            routes = []
            for hop in range(2):
                norm = self.ln_attn(state)
                q = self.q_proj(norm).view(bsz, length, n_heads, -1).transpose(1, 2)
                k = self.k_proj(norm).view(bsz, length, n_heads, -1).transpose(1, 2)
                v = self.v_proj(norm).view(bsz, length, n_heads, -1).transpose(1, 2)
                kc, vc = k[:, :, indices, :], v[:, :, indices, :]
                scores = (q.unsqueeze(3) * kc).sum(-1) * scale
                scores = scores.masked_fill(~valid, -1e4)
                weights = F.softmax(scores, dim=-1)
                forced = torch.zeros_like(weights)
                if mode in {"force_hop1", "force_both"} and hop == 0:
                    forced[:, :, value_positions, offset1_index] = 1.0
                    value_mask = torch.zeros(length, dtype=torch.bool, device=device)
                    value_mask[value_positions] = True
                    weights = torch.where(value_mask.view(1, 1, length, 1), forced, weights)
                if mode in {"force_hop2", "force_both"} and hop == 1:
                    query_forced = torch.zeros_like(weights)
                    query_forced[torch.arange(bsz, device=device)[:, None], :, target_position, value_offset_indices[slot][:, None]] = 1.0
                    query_mask = torch.zeros(length, dtype=torch.bool, device=device)
                    query_mask[target_position] = True
                    weights = torch.where(query_mask.view(1, 1, length, 1), query_forced, weights)
                routes.append(weights)
                context = (weights.unsqueeze(-1) * vc).sum(3)
                message = self.out_proj(context.transpose(1, 2).contiguous().view(bsz, length, d_model))
                gate_input = torch.cat([self.ln_attn(state), self.ln_attn(message)], dim=-1)
                gate = torch.sigmoid(self.gate_proj(gate_input))
                state = (1.0 - gate) * state + gate * message
                state = state + (1.0 / math.sqrt(2.0)) * self.mlp(self.ln_mlp(state))
            logits = self.head(self.ln_f(state[:, target_position]))
            return (logits, routes) if capture else logits

    model = Model().to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + sum(offsets) * 23 + len(mode) * 101)
    checkpoints, start = [], time.time()

    def evaluate(eval_generator):
        model.eval()
        correct = total = 0
        hop1_binding_top1 = hop2_query_top1 = 0
        with torch.no_grad():
            for _ in range(50):
                x, y, slot = make_batch(eval_generator)
                logits, routes = model(x, slot, capture=True)
                correct += logits.argmax(-1).eq(y).sum().item()
                total += y.numel()
                hop1 = routes[0][:, :, value_positions, :].float().mean(1)
                hop1_binding_top1 += hop1.argmax(-1).eq(torch.full_like(hop1.argmax(-1), offset1_index)).sum().item()
                hop2 = routes[1][:, :, target_position, :].float().mean(1)
                expected = value_offset_indices[slot]
                hop2_query_top1 += hop2.argmax(-1).eq(expected).sum().item()
        return {"accuracy": round(100.0 * correct / total, 4), "hop1_binding_top1": round(100.0 * hop1_binding_top1 / (total * len(value_positions)), 4), "hop2_query_top1": round(100.0 * hop2_query_top1 / total, 4), "queries": total}

    for step in range(1, total_steps + 1):
        model.train()
        x, y, slot = make_batch(generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x, slot), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step in {1000, 3000, total_steps}:
            eg = torch.Generator(device=device)
            eg.manual_seed(seed + 100000 + step)
            checkpoints.append({"step": step, **evaluate(eg)})
    return {"experiment": "S2-019", "model": model_name, "mode": mode, "offsets": offsets, "value_offsets": value_offsets, "seed": seed, "parameters": parameters, "steps": total_steps, "checkpoints": checkpoints, "elapsed_s": round(time.time() - start, 3)}


@app.local_entrypoint()
def main():
    import pathlib
    import time

    offsets = [1, 2, 4, 8, 16, 32, 64, 96, 128, 160]
    modes = ["learned", "force_hop1", "force_hop2", "force_both"]
    seeds = [42, 1337, 2026]
    configs = [(f"{mode}_s{seed}", mode, offsets, seed) for mode in modes for seed in seeds]
    calls = [train_one.spawn(name, mode, policy, seed) for name, mode, policy, seed in configs]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    payload = {"experiment": "S2-019", "description": "Two-hop binding versus retrieval isolation", "source_commit": "working-tree", "offsets": offsets, "value_offsets": [4, 8, 16, 32, 64, 96, 128, 160], "modes": modes, "random_exact_baseline_percent": 2.5, "seeds": seeds, "training_steps": 6000, "results": results, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = pathlib.Path("season2/results/s2_019_two_hop_binding_isolation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}", flush=True)
