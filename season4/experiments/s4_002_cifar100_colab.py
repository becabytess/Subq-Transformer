# ==============================================================================
# STUDY S4-002: FAST CIFAR-100 SPATIOTEMPORAL LATTICE (GOOGLE COLAB OPTIMIZED)
# Architecture: Native Fused GRUCell + Direct Tensor Slicing (d=128, L=256, T=8)
# ==============================================================================
# In Colab: Runtime -> Change runtime type -> T4 GPU or A100 GPU
# !pip install -q datasets

import os
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# ------------------------------------------------------------------------------
# 1. Environment & Hyperparameters
# ------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

seq_len = 256            # 16x16 = 256 patches (2x2 stride 2 on 32x32 CIFAR image)
d_model = 128            # Hidden feature dimension (~218k params)
n_heads = 4              # Attention heads
d_k = d_model // n_heads # 32
d_mlp = d_model * 4      # 512 (4x expansion)
n_hops = 8               # T=8 recurrent thought hops
p_peaks = 4              # P=4 crest pairs -> K = 1 + 2*4 = 9 bilateral candidates
k_total = 1 + 2 * p_peaks
num_waves = 12           # Continuous Fourier carriers
num_classes = 100        # CIFAR-100 classes
batch_size = 128
epochs = 20

# ------------------------------------------------------------------------------
# 2. CIFAR-100 Dataset (Pre-cached in RAM for instant 0-overhead GPU streaming)
# ------------------------------------------------------------------------------
print("Loading HuggingFace dataset 'uoft-cs/cifar100'...")
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

# num_workers=2 with pin_memory for fast async host-to-device transfer
train_loader = DataLoader(CifarDataset("train", train_transform), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(CifarDataset("test", test_transform), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train set: {len(raw['train']):,} images | Test set: {len(raw['test']):,} images")

# ------------------------------------------------------------------------------
# 3. High-Performance Bilateral Sensory Antenna (Direct Slicing, Zero Gather)
# ------------------------------------------------------------------------------
class FastBilateralWaveRouter(nn.Module):
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

    def forward(self, wave_latent, device):
        curr_params = wave_latent.view(1, self.num_waves, 4)
        amp = torch.tanh(curr_params[..., 0]).view(1, 1, 1, self.num_waves)
        omega = (F.softplus(curr_params[..., 1]).view(1, 1, 1, self.num_waves) * self.base_freqs)
        phi = (curr_params[..., 2] * math.pi).view(1, 1, 1, self.num_waves)
        decay = (F.softplus(curr_params[..., 3]) * 0.05).view(1, 1, 1, self.num_waves)

        wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
        wave_1d = wave_comps.sum(dim=-1).squeeze() # [max_d - 1]

        topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.p_peaks, dim=-1)
        past_peak_offsets = past_peak_offsets + 1 # [P]

        zero_offset = torch.zeros(1, dtype=torch.long, device=device)
        zero_val = torch.zeros(1, dtype=torch.float, device=device)

        # 1D offset vector: [anchor 0, +offsets (left), -offsets (right)]
        offsets_1d = torch.cat([zero_offset, past_peak_offsets, -past_peak_offsets]) # [K]
        peak_vals_1d = torch.cat([zero_val, topk_vals, topk_vals]) # [K]

        next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
        return offsets_1d, peak_vals_1d, next_wave_latent


class FastBilateralAttentionAntenna(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.router = FastBilateralWaveRouter(num_waves=num_waves, p_peaks=p_peaks, max_d=seq_len)
        self.scale = 1.0 / math.sqrt(d_k)
        self.register_buffer("pos_idx", torch.arange(seq_len).unsqueeze(1)) # [L, 1]

    def forward(self, s, wave_latent):
        B, L, D = s.shape
        device = s.device

        offsets_1d, peak_vals_1d, next_wave = self.router(wave_latent, device)

        # 1. Direct 2D Index Matrix [L, K] - Fast fused slice, ZERO 5D gather!
        raw_targets = self.pos_idx - offsets_1d.unsqueeze(0) # [L, K]
        valid_mask = (raw_targets >= 0) & (raw_targets < L) # [L, K]
        clamped_targets = torch.clamp(raw_targets, 0, L - 1) # [L, K]

        # 2. Q, K, V Projections
        Q = self.q_proj(s).view(B, L, n_heads, d_k).transpose(1, 2) # [B, H, L, d_k]
        K = self.k_proj(s).view(B, L, n_heads, d_k).transpose(1, 2) # [B, H, L, d_k]
        V = self.v_proj(s).view(B, L, n_heads, d_k).transpose(1, 2) # [B, H, L, d_k]

        # 3. Direct Tensor Slicing in 1 single CUDA kernel call:
        K_cand = K[:, :, clamped_targets, :] # [B, H, L, K, d_k]
        V_cand = V[:, :, clamped_targets, :] # [B, H, L, K, d_k]

        # 4. Attention compute
        scores = (Q.unsqueeze(3) * K_cand).sum(dim=-1) * self.scale + peak_vals_1d.view(1, 1, 1, k_total)
        scores = scores.masked_fill(~valid_mask.view(1, 1, L, k_total), -1e4)
        weights = F.softmax(scores, dim=-1)

        context = (weights.unsqueeze(-1) * V_cand).sum(dim=3) # [B, H, L, d_k]
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(context), next_wave


# ------------------------------------------------------------------------------
# 4. Spatiotemporal Vision Architecture (Native Fused cuDNN GRUCell)
# ------------------------------------------------------------------------------
class SpatiotemporalViT(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.antenna = FastBilateralAttentionAntenna()
        self.ln_in = nn.LayerNorm(d_model)

        if mode == 'canonical_subq_final_mlp_vit':
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model),
            )
        elif mode == 'canonical_subq_per_hop_mlp_vit':
            self.ln_attn = nn.LayerNorm(d_model)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model),
            )
        elif mode == 'attention_driven_gru_vit':
            # PyTorch Native cuDNN C++ Fused GRUCell across all tokens in parallel!
            self.gru_cell = nn.GRUCell(d_model, d_model)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch(x).flatten(2).transpose(1, 2).contiguous()
        s = tokens + self.pos

        w = self.antenna.router.init_wave_latent
        inv_sqrt_T = 1.0 / math.sqrt(n_hops)

        # Spatiotemporal Recurrent Loop (T=8 hops)
        for hop in range(1, n_hops + 1):
            z_norm = self.ln_in(s)
            c, w = self.antenna(z_norm, wave_latent=w)

            if self.mode == 'canonical_subq_final_mlp_vit':
                s = s + inv_sqrt_T * c
            elif self.mode == 'canonical_subq_per_hop_mlp_vit':
                s = s + inv_sqrt_T * c
                s = self.ln_attn(s)
                s = s + inv_sqrt_T * self.mlp(self.ln_mlp(s))
            elif self.mode == 'attention_driven_gru_vit':
                # Fully parallel across all B * L tokens in single cuDNN kernel call:
                s = self.gru_cell(c.reshape(-1, d_model), s.reshape(-1, d_model)).reshape(B, seq_len, d_model)

        if self.mode == 'canonical_subq_final_mlp_vit':
            s = s + self.mlp(self.ln_mlp(s))

        pooled = s.mean(dim=1)
        logits = self.head(self.ln_f(pooled))
        return logits


# ------------------------------------------------------------------------------
# 5. Training Engine
# ------------------------------------------------------------------------------
def train_single_model(model_type: str):
    print("\n" + "=" * 95)
    print(f"  STARTING TRAINING: {model_type}")
    print("=" * 95)

    torch.manual_seed(seed)
    model = SpatiotemporalViT(model_type).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_type}] Trainable Parameters: {param_count:,}")

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

        for batch_idx, (bx, by) in enumerate(train_loader, 1):
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == by).sum().item()
            train_total += bx.size(0)
            global_step += 1

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                batch_acc = 100.0 * train_correct / train_total
                cur_time = time.time() - ep_t0
                print(f"  [{model_type}] Ep {epoch:2d}/{epochs} [{batch_idx:3d}/{len(train_loader)}] | Loss: {loss.item():.4f} | Acc: {batch_acc:5.2f}% | Time: {cur_time:4.1f}s", flush=True)

        avg_train_loss = train_loss / train_total
        avg_train_acc = 100.0 * train_correct / train_total

        # Evaluation on Test Split
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for vx, vy in test_loader:
                vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                v_logits = model(vx)
                preds = v_logits.argmax(dim=-1)
                test_correct += (preds == vy).sum().item()
                test_total += vx.size(0)

        avg_test_acc = 100.0 * test_correct / test_total
        ep_time = time.time() - ep_t0
        total_time = time.time() - t0

        print(f">>> [{model_type}] EPOCH {epoch:2d}/{epochs} COMPLETE | Train Acc: {avg_train_acc:5.2f}% | TEST ACC: {avg_test_acc:5.2f}% | Ep Time: {ep_time:.1f}s (Total: {total_time:.1f}s)\n", flush=True)

        epoch_records.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'test_acc': avg_test_acc,
            'ep_time_s': ep_time,
        })

    best_test_acc = max(r['test_acc'] for r in epoch_records)
    return {
        'model_type': model_type,
        'parameters': param_count,
        'final_test_acc': epoch_records[-1]['test_acc'],
        'best_test_acc': best_test_acc,
        'total_time_s': time.time() - t0,
        'history': epoch_records,
    }

# ------------------------------------------------------------------------------
# 6. Run Models & Print Scorecard
# ------------------------------------------------------------------------------
models_to_test = [
    'attention_driven_gru_vit',       # Season 4: GRU cell, zero MLPs (~218k params)
    'canonical_subq_final_mlp_vit',   # Season 2: 1 final MLP (~251k params)
    'canonical_subq_per_hop_mlp_vit', # Season 2: per-hop MLP (~251k params)
]

all_results = {}
for m in models_to_test:
    res = train_single_model(m)
    all_results[m] = res

print("\n" + "=" * 105)
print("  STUDY S4-002 FINAL SCORECARD (CIFAR-100, d=128, L=256, T=8 HOPS, 20 EPOCHS)")
print("=" * 105)
print(f"{'Architecture':<34} | {'Parameters':<12} | {'Best Test Acc':<14} | {'Final Test Acc':<14} | {'Total Time':<10}")
print("-" * 105)
for m in models_to_test:
    r = all_results[m]
    mins = r['total_time_s'] / 60.0
    print(f"{r['model_type']:<34} | {r['parameters']:<12,d} | {r['best_test_acc']:>12.2f}% | {r['final_test_acc']:>12.2f}% | {mins:>8.1f} min")
print("=" * 105)

with open('s4_002_cifar100_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("Saved results to s4_002_cifar100_results.json")
