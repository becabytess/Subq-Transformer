"""S4-002: Attention-Driven Parallel GRU Cell on High-Res CIFAR-100 (d=256, L=256, T=8 Hops).

Scientific Objective:
Tests whether the Spatiotemporal Lattice of Coupled Parallel RNNs transfers to 2D Vision:
1. Spatial Patches (16x16 = 256 tokens) flattened into a lattice.
2. Sensory Antenna: Mirrored Bilateral Wave Router (P=4 positive crests -> K=9 bilateral candidates:
   anchor 0, left -Δ, right +Δ with strict boundary masking, zero wrap-around).
3. Thought Depth: T=8 recurrent hops of parallel thought.
4. Flagship: attention_driven_gru_vit (Season 4). Attention gathers spatial context c_i^(t);
   ParallelGRUCell updates the token hidden state. Zero MLPs in the model!

Compared against:
- canonical_subq_final_mlp_vit: Season 2 Baseline (8 linear attention hops + 1 final 4x MLP).
- canonical_subq_per_hop_mlp_vit: Season 2 Per-Hop MLP Baseline (8 hops with post-attention LN + 4x MLP).

Configuration:
- Dataset: High-Res CIFAR-100 (L=256, 2x2 patches, 100 classes)
- d_model = 256, n_heads = 4, d_k = 64, d_mlp = 1024 (4x)
- Epochs: 20, Batch Size: 128, Optimizer: AdamW (lr=1e-3, warmup=1 ep, cosine decay to 1e-4)
- Hardware: Modal NVIDIA A10G, concurrent parallel execution
"""

import json
import math
import os
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "torchvision>=0.17.0", "datasets>=2.18.0", "numpy", "pillow"
)
app = modal.App("season4-s4-002-cifar100-gru", image=image)
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def train_cifar100_model(
    model_type: str,
    epochs: int = 20,
    n_hops: int = 8,
    p_peaks: int = 4,
    num_waves: int = 12,
    d_model: int = 128,
    batch_size: int = 128,
    seed: int = 42,
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 256  # 16x16 = 256 patches (2x2 stride 2 on 32x32 image)
    k_total = 1 + 2 * p_peaks  # 9 candidates (anchor 0, 4 left, 4 right)
    num_classes = 100
    n_heads = 4
    d_k = d_model // n_heads
    d_mlp = d_model * 4

    print("=" * 105, flush=True)
    print(f"  STARTING CIFAR-100 RUN: {model_type} (d={d_model}, L={seq_len}, T={n_hops} Hops, {epochs} Epochs)", flush=True)
    print(f"  Bilateral Candidates K = {k_total} (P = {p_peaks} pairs) | Device: {torch.cuda.get_device_name(0)}", flush=True)
    print("=" * 105, flush=True)

    # 1. Dataset Loading with Fast Pre-fetching
    raw = load_dataset("uoft-cs/cifar100")
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    class CifarDataset(Dataset):
        def __init__(self, split_name, transform):
            self.items = raw[split_name]
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            item = self.items[index]
            img = self.transform(item.get("img", item.get("image")))
            label = item.get("fine_label", item.get("label"))
            return img, label

    train_loader = DataLoader(
        CifarDataset("train", train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        CifarDataset("test", test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # Mirrored Bilateral Wave Router (P=4 -> K=9, zero cyclic wrapping)
    # -------------------------------------------------------------------------
    class BilateralWaveRouter(nn.Module):
        def __init__(self, num_waves=12, p_peaks=4, max_d=256):
            super().__init__()
            self.num_waves = num_waves
            self.p_peaks = p_peaks
            self.k_total = 1 + 2 * p_peaks
            self.max_d = max_d

            self.init_wave_latent = nn.Parameter(torch.randn(1, num_waves * 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 64),
                nn.GELU(),
                nn.Linear(64, num_waves * 4),
            )

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

        def forward(self, wave_latent, B, device):
            curr_params = wave_latent.view(1, self.num_waves, 4)
            amp = torch.tanh(curr_params[..., 0]).view(1, 1, 1, self.num_waves)
            omega = (F.softplus(curr_params[..., 1]).view(1, 1, 1, self.num_waves) * self.base_freqs)
            phi = (curr_params[..., 2] * math.pi).view(1, 1, 1, self.num_waves)
            decay = (F.softplus(curr_params[..., 3]) * 0.05).view(1, 1, 1, self.num_waves)

            wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
            wave_1d = wave_comps.sum(dim=-1).expand(B, 1, -1)  # (B, 1, max_d - 1)

            topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.p_peaks, dim=-1)
            past_peak_offsets = past_peak_offsets + 1  # offsets in [1, max_d - 1]

            zero_offset = torch.zeros((B, 1, 1), dtype=torch.long, device=device)
            zero_val = torch.zeros((B, 1, 1), dtype=torch.float, device=device)

            # Bilateral reflection: center (0), left (+Δ -> pos - Δ), right (-Δ -> pos + Δ)
            peak_offsets = torch.cat([zero_offset, past_peak_offsets, -past_peak_offsets], dim=-1)
            peak_vals = torch.cat([zero_val, topk_vals, topk_vals], dim=-1)

            # Broadcast to all attention heads
            peak_offsets = peak_offsets.expand(B, n_heads, self.k_total)
            peak_vals = peak_vals.expand(B, n_heads, self.k_total)

            next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
            return peak_offsets, peak_vals, next_wave_latent

    # -------------------------------------------------------------------------
    # Bilateral Sensory Antenna
    # -------------------------------------------------------------------------
    class BilateralAttentionAntenna(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.router = BilateralWaveRouter(num_waves=num_waves, p_peaks=p_peaks, max_d=seq_len)
            self.scale = 1.0 / math.sqrt(d_k)

        def forward(self, s, wave_latent):
            B, L, D = s.shape
            device = s.device

            peak_offsets, peak_vals, next_wave = self.router(wave_latent, B, device)

            Q = self.q_proj(s).view(B, L, n_heads, d_k).transpose(1, 2)
            K = self.k_proj(s).view(B, L, n_heads, d_k).transpose(1, 2)
            V = self.v_proj(s).view(B, L, n_heads, d_k).transpose(1, 2)

            q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
            target_indices = q_pos - peak_offsets.unsqueeze(2)  # (B, H, L, K)

            # Strict non-cyclic spatial boundaries: positions outside [0, L-1] are masked cleanly
            valid_mask = (target_indices >= 0) & (target_indices < L)
            target_clamped = torch.clamp(target_indices, min=0, max=L - 1)

            idx_exp = target_clamped.unsqueeze(-1).expand(B, n_heads, L, k_total, d_k)
            K_gathered = torch.gather(K.unsqueeze(3).expand(B, n_heads, L, k_total, d_k), dim=2, index=idx_exp)
            V_gathered = torch.gather(V.unsqueeze(3).expand(B, n_heads, L, k_total, d_k), dim=2, index=idx_exp)

            Q_exp = Q.unsqueeze(3)
            scores = (Q_exp * K_gathered).sum(dim=-1) * self.scale + peak_vals.unsqueeze(2)
            scores = scores.masked_fill(~valid_mask, float("-inf"))
            weights = F.softmax(scores, dim=-1) * valid_mask.float()
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

            context = (weights.unsqueeze(-1) * V_gathered).sum(dim=3)
            context = context.transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(context), next_wave

    # -------------------------------------------------------------------------
    # Parallel GRU Cell (The Spatiotemporal Transition Engine)
    # -------------------------------------------------------------------------
    class ParallelGRUCell(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.w_ih = nn.Linear(d_model, 3 * d_model)
            self.w_hh = nn.Linear(d_model, 3 * d_model)

        def forward(self, x, h):
            gates_i = self.w_ih(x)
            gates_h = self.w_hh(h)
            i_r, i_z, i_n = gates_i.chunk(3, dim=-1)
            h_r, h_z, h_n = gates_h.chunk(3, dim=-1)

            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h_next = (1.0 - z) * n + z * h
            return h_next, z.detach().mean().item(), r.detach().mean().item()

    # -------------------------------------------------------------------------
    # Vision Architecture
    # -------------------------------------------------------------------------
    class SpatiotemporalViT(nn.Module):
        def __init__(self, mode: str):
            super().__init__()
            self.mode = mode
            self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
            self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            self.antenna = BilateralAttentionAntenna()
            self.ln_in = nn.LayerNorm(d_model)

            if mode == "canonical_subq_final_mlp_vit":
                self.ln_mlp = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_mlp),
                    nn.GELU(),
                    nn.Linear(d_mlp, d_model),
                )
            elif mode == "canonical_subq_per_hop_mlp_vit":
                self.ln_attn = nn.LayerNorm(d_model)
                self.ln_mlp = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_mlp),
                    nn.GELU(),
                    nn.Linear(d_mlp, d_model),
                )
            elif mode == "attention_driven_gru_vit":
                self.gru_cell = ParallelGRUCell(d_model)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes, bias=False)

        def forward(self, x):
            B = x.shape[0]
            # 1. Patch projection + spatial position embedding
            tokens = self.patch(x).flatten(2).transpose(1, 2)  # [B, 256, d_model]
            s = tokens + self.pos

            w = self.antenna.router.init_wave_latent
            inv_sqrt_T = 1.0 / math.sqrt(n_hops)
            gate_stats = []

            # 2. Spatiotemporal Recurrent Thought Loop (T=8 hops)
            for hop in range(1, n_hops + 1):
                z_norm = self.ln_in(s)
                c, w = self.antenna(z_norm, wave_latent=w)

                if self.mode == "canonical_subq_final_mlp_vit":
                    s = s + inv_sqrt_T * c
                elif self.mode == "canonical_subq_per_hop_mlp_vit":
                    s = s + inv_sqrt_T * c
                    s = self.ln_attn(s)
                    s = s + inv_sqrt_T * self.mlp(self.ln_mlp(s))
                elif self.mode == "attention_driven_gru_vit":
                    s, mean_z, mean_r = self.gru_cell(x=c, h=s)
                    gate_stats.append({"hop": hop, "mean_update_z": mean_z, "mean_reset_r": mean_r})

            if self.mode == "canonical_subq_final_mlp_vit":
                s = s + self.mlp(self.ln_mlp(s))

            # 3. Global Average Pooling across all 256 image patches
            pooled = s.mean(dim=1)
            logits = self.head(self.ln_f(pooled))
            return logits, gate_stats

    model = SpatiotemporalViT(model_type).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_type}] Total Trainable Parameters: {param_count:,}", flush=True)

    # -------------------------------------------------------------------------
    # Training Loop with Warmup + Cosine Annealing
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    total_steps = epochs * len(train_loader)
    warmup_steps = len(train_loader)

    def get_lr(step):
        if step < warmup_steps:
            return 1e-3 * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1.0 + math.cos(math.pi * progress))

    t0 = time.time()
    epoch_records = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == by).sum().item()
            train_total += bx.size(0)
            global_step += 1

        avg_train_loss = train_loss / train_total
        avg_train_acc = 100.0 * train_correct / train_total

        # Evaluation on Test Split
        model.eval()
        test_correct = 0
        test_total = 0
        last_gate_stats = []

        with torch.no_grad():
            for b_idx, (vx, vy) in enumerate(test_loader):
                vx, vy = vx.to(device), vy.to(device)
                v_logits, g_stats = model(vx)
                preds = v_logits.argmax(dim=-1)
                test_correct += (preds == vy).sum().item()
                test_total += vx.size(0)
                if b_idx == 0 and g_stats:
                    last_gate_stats = g_stats

        avg_test_acc = 100.0 * test_correct / test_total
        ep_time = time.time() - ep_t0
        total_time = time.time() - t0

        print(
            f"[{model_type}] Epoch {epoch:>2}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {avg_train_acc:>5.2f}% | Test Acc: {avg_test_acc:>5.2f}% | "
            f"Ep Time: {ep_time:.1f}s | Total: {total_time:.1f}s",
            flush=True,
        )

        epoch_records.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "test_acc": avg_test_acc,
            "ep_time_s": ep_time,
            "gate_stats": last_gate_stats,
        })

    best_test_acc = max(r["test_acc"] for r in epoch_records)
    return {
        "model_type": model_type,
        "parameters": param_count,
        "d_model": d_model,
        "seq_len": seq_len,
        "n_hops": n_hops,
        "epochs": epochs,
        "final_test_acc": epoch_records[-1]["test_acc"],
        "best_test_acc": best_test_acc,
        "total_time_s": time.time() - t0,
        "history": epoch_records,
    }


@app.local_entrypoint()
def main():
    models_to_test = [
        "canonical_subq_final_mlp_vit",
        "canonical_subq_per_hop_mlp_vit",
        "attention_driven_gru_vit",
    ]

    print("=" * 115)
    print("  LAUNCHING STUDY S4-002: HIGH-RES CIFAR-100 SPATIOTEMPORAL LATTICE SHOOTOUT")
    print("  Testing Attention-Driven Parallel GRU Cell (d=128, L=256, T=8 Hops, 20 Epochs)")
    print("  Models running concurrently on Modal NVIDIA A10G GPUs")
    print("=" * 115)

    t0 = time.time()
    # Spawn all 3 vision models in parallel
    calls = [train_cifar100_model.spawn(m, epochs=20, n_hops=8, d_model=128, batch_size=128, seed=42) for m in models_to_test]

    results = {}
    for m, call in zip(models_to_test, calls):
        print(f"Waiting for {m}...", flush=True)
        res = call.get()
        results[m] = res
        print(f"--> Finished {m}: Best Test Acc = {res['best_test_acc']:.2f}% | Final = {res['final_test_acc']:.2f}% ({res['total_time_s']:.1f}s)", flush=True)

    total_wall_clock = time.time() - t0

    print("\n" + "=" * 115)
    print("  STUDY S4-002 FINAL SCORECARD (CIFAR-100, d=128, L=256, T=8 HOPS, 20 EPOCHS)")
    print("=" * 115)
    print(f"{'Architecture':<34} | {'Parameters':<12} | {'Best Test Acc':<14} | {'Final Test Acc':<14} | {'Time (s)':<10}")
    print("-" * 115)
    for m in models_to_test:
        r = results[m]
        print(f"{r['model_type']:<34} | {r['parameters']:<12,d} | {r['best_test_acc']:>12.2f}% | {r['final_test_acc']:>12.2f}% | {r['total_time_s']:<10.1f}")
    print("=" * 115)
    print(f"Total Wall-Clock Parallel Runtime: {total_wall_clock:.1f}s ({total_wall_clock / 60.0:.1f} min)")

    os.makedirs("season4/results", exist_ok=True)
    out_json = "season4/results/s4_002_cifar100_spatiotemporal_gru.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved S4-002 results to {out_json}]")
