"""
==========================================================================================
  STUDY S4-009: FAIR CIFAR-100 SHOOTOUT (FULLY VECTORIZED & PRE-CACHED)
  MULTI-STREAM CONSENSUS VIT (M=3, K=9, CRIPPLED MLP) VS. COMPENSATED BASELINE (K=27, 4X MLP)
==========================================================================================
Experimental Design (Fairness Normalization):
1. Candidate Parity:
   - Multi-Stream (Model 1): M=3 streams, each has p_peaks=4 (K=9 candidates: [0, ±Δ_1..4]).
     Total candidates per token per hop: 3 * 9 = 27 candidates.
   - Compensated Baseline (Model 2): M=1 stream, p_peaks=13 (K=27 candidates: [0, ±Δ_1..13]).
     Total candidates per token per hop: 1 * 27 = 27 candidates.
   -> EXACT SAME TOTAL ATTENTION VIEW PER TOKEN (27 candidates/hop, 216 across 8 hops)!

2. Parameter Counter-Balancing:
   - Multi-Stream is crippled to d_mlp = 128 (1x ratio) -> 380,100 parameters.
   - Compensated Baseline gets full standard d_mlp = 512 (4x ratio) -> 235,268 parameters.

3. Complete GPU Vectorization & Zero-CPU DataLoader:
   - BatchedMultiStreamBilateralViT: All M streams execute in parallel within unified
     tensor operations ([M, B, L, D]). Zero sequential stream loops!
   - RAM Pre-caching: CIFAR-100 (~150 MB) is loaded into a contiguous uint8 tensor at startup.
     Completely eliminates Colab CPU bottlenecks from PIL decoding and HuggingFace Arrow tables.

4. Execution Order:
   - Job 1: Multi-Stream Consensus ViT trains first (20 epochs).
   - Job 2: Compensated Single-Stream ViT trains second (20 epochs).
   - Direct side-by-side scorecard at completion!
==========================================================================================
"""

import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# ------------------------------------------------------------------------------
# 1. Architecture 1: Fully Vectorized Batched Multi-Stream Consensus ViT
# ------------------------------------------------------------------------------
class BatchedMultiStreamBilateralViT(nn.Module):
    """Fully vectorized Multi-Stream SubQ ViT.
    All M streams execute in parallel within unified tensor operations ([M, B, L, D]).
    Zero sequential Python loops across streams!
    """
    def __init__(self, num_classes=100, seq_len=256, d_model=128, n_heads=4, d_mlp=128, num_streams=3, hops_per_phase=4, p_peaks=4, num_waves=12):
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_mlp = d_mlp
        self.M = num_streams
        self.hops_per_phase = hops_per_phase
        self.p_peaks = p_peaks
        self.k_total = 1 + 2 * p_peaks
        self.num_waves = num_waves
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Patch Tokenizer (32x32 -> 16x16 = 256 tokens)
        self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Batched Q, K, V for all M streams: [M, D, D]
        self.W_q = nn.Parameter(torch.randn(self.M, d_model, d_model) * (1.0 / math.sqrt(d_model)))
        self.W_k = nn.Parameter(torch.randn(self.M, d_model, d_model) * (1.0 / math.sqrt(d_model)))
        self.W_v = nn.Parameter(torch.randn(self.M, d_model, d_model) * (1.0 / math.sqrt(d_model)))
        self.ln_attn = nn.LayerNorm(d_model)

        # Batched Wave Router parameters for all M streams
        self.init_wave_latent = nn.Parameter(torch.randn(self.M, num_waves * 4) * 0.1)
        self.wave_trans_w1 = nn.Parameter(torch.randn(self.M, num_waves * 4, 64) * 0.1)
        self.wave_trans_b1 = nn.Parameter(torch.zeros(self.M, 1, 64))
        self.wave_trans_w2 = nn.Parameter(torch.randn(self.M, 64, num_waves * 4) * 0.1)
        self.wave_trans_b2 = nn.Parameter(torch.zeros(self.M, 1, num_waves * 4))

        log_freqs = torch.linspace(0.0, -2.0, num_waves)
        self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
        self.register_buffer("d_grid", torch.arange(1, seq_len).float().view(1, 1, seq_len - 1, 1))
        self.register_buffer("pos_idx", torch.arange(seq_len).unsqueeze(1))  # [L, 1]

        # Batched Stream MLPs: [M, D, d_mlp] and [M, d_mlp, D]
        self.ln_mlp = nn.LayerNorm(d_model)
        self.W_mlp1 = nn.Parameter(torch.randn(self.M, d_model, d_mlp) * (1.0 / math.sqrt(d_model)))
        self.b_mlp1 = nn.Parameter(torch.zeros(self.M, 1, 1, d_mlp))
        self.W_mlp2 = nn.Parameter(torch.randn(self.M, d_mlp, d_model) * (1.0 / math.sqrt(d_mlp)))
        self.b_mlp2 = nn.Parameter(torch.zeros(self.M, 1, 1, d_model))

        # Cross-Stream Mixer MLP: takes [B, L, M * D] -> [B, L, D]
        self.ln_mix = nn.LayerNorm(self.M * d_model)
        self.mixer = nn.Sequential(
            nn.Linear(self.M * d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        nn.init.zeros_(self.mixer[-1].weight)
        nn.init.zeros_(self.mixer[-1].bias)

        # Classification Head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def _batched_router(self, wave_latent, dev):
        curr = wave_latent.view(self.M, self.num_waves, 4)
        amp = torch.tanh(curr[..., 0]).view(self.M, 1, 1, self.num_waves)
        omega = F.softplus(curr[..., 1]).view(self.M, 1, 1, self.num_waves) * self.base_freqs
        phi = (curr[..., 2] * math.pi).view(self.M, 1, 1, self.num_waves)
        decay = (F.softplus(curr[..., 3]) * 0.05).view(self.M, 1, 1, self.num_waves)

        wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
        wave_1d = wave_comps.sum(dim=-1).squeeze(1)  # [M, max_d - 1]

        topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.p_peaks, dim=-1)
        past_peak_offsets = past_peak_offsets + 1  # [M, p_peaks]

        zero_off = torch.zeros(self.M, 1, dtype=torch.long, device=dev)
        zero_val = torch.zeros(self.M, 1, dtype=torch.float, device=dev)

        offsets = torch.cat([zero_off, past_peak_offsets, -past_peak_offsets], dim=-1)  # [M, K]
        vals = torch.cat([zero_val, topk_vals, topk_vals], dim=-1)  # [M, K]

        # Batched wave transition update
        wl = wave_latent.unsqueeze(1)
        h = F.gelu(torch.bmm(wl, self.wave_trans_w1) + self.wave_trans_b1)
        next_latent = wave_latent + 0.1 * (torch.bmm(h, self.wave_trans_w2) + self.wave_trans_b2).squeeze(1)
        return offsets, vals, next_latent

    def _run_batched_hops(self, state, wave_latent, n_hops=4):
        M, B, L, D = state.shape
        dev = state.device

        for _ in range(n_hops):
            z = self.ln_attn(state)  # [M, B, L, D]
            offsets, vals, wave_latent = self._batched_router(wave_latent, dev)  # [M, K]

            # Vectorized candidate targets for all M streams simultaneously
            raw_targets = self.pos_idx.unsqueeze(0) - offsets.unsqueeze(1)  # [M, L, K]
            valid_mask = (raw_targets >= 0) & (raw_targets < L)  # [M, L, K]
            clamped = torch.clamp(raw_targets, 0, L - 1)  # [M, L, K]

            # Batched QKV projections: [M, B, L, D] @ [M, D, D]
            Q = torch.einsum('mbld,mde->mble', z, self.W_q).view(M, B, L, self.n_heads, self.d_k).transpose(2, 3)
            K = torch.einsum('mbld,mde->mble', z, self.W_k).view(M, B, L, self.n_heads, self.d_k).transpose(2, 3)
            V = torch.einsum('mbld,mde->mble', z, self.W_v).view(M, B, L, self.n_heads, self.d_k).transpose(2, 3)

            # Batched candidate gathering across M streams
            clamped_exp = clamped.unsqueeze(1).unsqueeze(2).expand(M, B, self.n_heads, L, self.k_total)
            K_cand = torch.gather(K.unsqueeze(4).expand(M, B, self.n_heads, L, self.k_total, self.d_k), dim=3, index=clamped_exp.unsqueeze(-1).expand(M, B, self.n_heads, L, self.k_total, self.d_k))
            V_cand = torch.gather(V.unsqueeze(4).expand(M, B, self.n_heads, L, self.k_total, self.d_k), dim=3, index=clamped_exp.unsqueeze(-1).expand(M, B, self.n_heads, L, self.k_total, self.d_k))

            scores = (Q.unsqueeze(4) * K_cand).sum(dim=-1) * self.scale + vals.view(M, 1, 1, 1, self.k_total)
            scores = scores.masked_fill(~valid_mask.view(M, 1, 1, L, self.k_total), -1e4)
            weights = F.softmax(scores, dim=-1)

            context = (weights.unsqueeze(-1) * V_cand).sum(dim=4)  # [M, B, H, L, d_k]
            state = context.transpose(2, 3).contiguous().view(M, B, L, D)

        # Batched MLPs for all M streams
        z_mlp = self.ln_mlp(state)
        h = F.gelu(torch.einsum('mbld,mde->mble', z_mlp, self.W_mlp1) + self.b_mlp1)
        mlp_out = torch.einsum('mble,med->mbld', h, self.W_mlp2) + self.b_mlp2
        state = state + mlp_out
        return state, wave_latent

    def forward(self, img, return_individual=False):
        B = img.shape[0]
        patches = self.patch(img).flatten(2).transpose(1, 2)
        x = patches + self.pos

        # Expand across all M streams: [M, B, L, D]
        state = x.unsqueeze(0).expand(self.M, B, self.seq_len, self.d_model).contiguous()

        # Phase 1: 4 optical hops in parallel across all M streams
        state, wave_latent = self._run_batched_hops(state, self.init_wave_latent, n_hops=self.hops_per_phase)

        # Mixer Consensus at T=4
        cat = state.permute(1, 2, 0, 3).reshape(B, self.seq_len, self.M * self.d_model)
        mean = state.mean(dim=0)
        s_common = mean + self.mixer(self.ln_mix(cat))

        # Phase 2: 4 optical hops in parallel across all M streams
        state = s_common.unsqueeze(0).expand(self.M, B, self.seq_len, self.d_model).contiguous()
        state, _ = self._run_batched_hops(state, wave_latent, n_hops=self.hops_per_phase)

        # Mixer Consensus at T=8
        cat_final = state.permute(1, 2, 0, 3).reshape(B, self.seq_len, self.M * self.d_model)
        mean_final = state.mean(dim=0)
        s_final = mean_final + self.mixer(self.ln_mix(cat_final))

        # Classification Head
        pooled = self.ln_f(s_final).mean(dim=1)
        logits = self.head(pooled)

        if return_individual:
            individual_logits = [self.head(self.ln_f(state[m]).mean(dim=1)) for m in range(self.M)]
            return logits, individual_logits

        return logits

# ------------------------------------------------------------------------------
# 2. Architecture 2: Compensated Single-Stream Baseline (Full 4x MLP, K=27 Peaks)
# ------------------------------------------------------------------------------
class FastBilateralWaveRouter(nn.Module):
    def __init__(self, num_waves=12, p_peaks=13, max_d=256):
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

    def forward(self, wave_latent, dev):
        curr_params = wave_latent.view(1, self.num_waves, 4)
        amp = torch.tanh(curr_params[..., 0]).view(1, 1, 1, self.num_waves)
        omega = F.softplus(curr_params[..., 1]).view(1, 1, 1, self.num_waves) * self.base_freqs
        phi = (curr_params[..., 2] * math.pi).view(1, 1, 1, self.num_waves)
        decay = (F.softplus(curr_params[..., 3]) * 0.05).view(1, 1, 1, self.num_waves)

        wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
        wave_1d = wave_comps.sum(dim=-1).squeeze()

        topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.p_peaks, dim=-1)
        past_peak_offsets = past_peak_offsets + 1

        zero_offset = torch.zeros(1, dtype=torch.long, device=dev)
        zero_val = torch.zeros(1, dtype=torch.float, device=dev)

        offsets_1d = torch.cat([zero_offset, past_peak_offsets, -past_peak_offsets])
        peak_vals_1d = torch.cat([zero_val, topk_vals, topk_vals])

        next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
        return offsets_1d, peak_vals_1d, next_wave_latent

class CompensatedSingleStreamViT(nn.Module):
    def __init__(self, num_classes=100, seq_len=256, d_model=128, n_heads=4, d_mlp=512, hops_per_phase=4, p_peaks=13):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.p_peaks = p_peaks
        self.k_total = 1 + 2 * p_peaks
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.hops_per_phase = hops_per_phase
        self.seq_len = seq_len

        self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.ln_attn = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        self.router = FastBilateralWaveRouter(num_waves=12, p_peaks=p_peaks, max_d=seq_len)
        self.register_buffer("pos_idx", torch.arange(seq_len).unsqueeze(1))

        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def run_optical_hops(self, state, wave_latent, n_hops=4):
        B, L, D = state.shape
        dev = state.device

        for _ in range(n_hops):
            z = self.ln_attn(state)
            offsets_1d, peak_vals_1d, wave_latent = self.router(wave_latent, dev)

            raw_targets = self.pos_idx - offsets_1d.unsqueeze(0)
            valid_mask = (raw_targets >= 0) & (raw_targets < L)
            clamped_targets = torch.clamp(raw_targets, 0, L - 1)

            Q = self.q(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            clamped_exp = clamped_targets.unsqueeze(0).unsqueeze(1).expand(B, self.n_heads, L, self.k_total)
            K_cand = torch.gather(K.unsqueeze(3).expand(B, self.n_heads, L, self.k_total, self.d_k), dim=2, index=clamped_exp.unsqueeze(-1).expand(B, self.n_heads, L, self.k_total, self.d_k))
            V_cand = torch.gather(V.unsqueeze(3).expand(B, self.n_heads, L, self.k_total, self.d_k), dim=2, index=clamped_exp.unsqueeze(-1).expand(B, self.n_heads, L, self.k_total, self.d_k))

            scores = (Q.unsqueeze(3) * K_cand).sum(dim=-1) * self.scale + peak_vals_1d.view(1, 1, 1, self.k_total)
            scores = scores.masked_fill(~valid_mask.view(1, 1, L, self.k_total), -1e4)
            weights = F.softmax(scores, dim=-1)

            context = (weights.unsqueeze(-1) * V_cand).sum(dim=3)
            state = context.transpose(1, 2).contiguous().view(B, L, D)

        state = state + self.mlp(self.ln_mlp(state))
        return state, wave_latent

    def forward(self, img):
        B = img.shape[0]
        patches = self.patch(img).flatten(2).transpose(1, 2)
        x = patches + self.pos

        s, w = self.run_optical_hops(x, self.router.init_wave_latent, n_hops=self.hops_per_phase)
        s, _ = self.run_optical_hops(s, w, n_hops=self.hops_per_phase)

        pooled = self.ln_f(s).mean(dim=1)
        return self.head(pooled)

# ------------------------------------------------------------------------------
# 3. High-Throughput RAM-Cached Dataset (Zero-CPU Bottleneck)
# ------------------------------------------------------------------------------
class CachedCifar100(Dataset):
    def __init__(self, X, Y, is_train=True):
        self.X = X  # Contiguous uint8 tensor: [N, 3, 32, 32]
        self.Y = Y  # Long tensor: [N]
        self.is_train = is_train
        self.mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        self.std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].float() / 255.0
        if self.is_train:
            # Fast tensor reflect pad + random crop
            img = F.pad(img, (4, 4, 4, 4), mode='reflect')
            top = torch.randint(0, 9, (1,)).item()
            left = torch.randint(0, 9, (1,)).item()
            img = img[:, top:top+32, left:left+32]
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, [2])

        img = (img - self.mean) / self.std
        return img, self.Y[idx]

# ------------------------------------------------------------------------------
# 4. Evaluation Routine
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, dev, is_multistream=False, num_streams=3):
    model.eval()
    correct_main = 0
    correct_streams = [0 for _ in range(num_streams)] if is_multistream else []
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
        if is_multistream:
            logits_c, indiv_logits = model(imgs, return_individual=True)
            preds_c = logits_c.argmax(dim=-1)
            correct_main += (preds_c == labels).sum().item()
            for s_idx, s_logits in enumerate(indiv_logits):
                preds_s = s_logits.argmax(dim=-1)
                correct_streams[s_idx] += (preds_s == labels).sum().item()
        else:
            logits = model(imgs)
            preds = logits.argmax(dim=-1)
            correct_main += (preds == labels).sum().item()

        total += labels.size(0)

    acc_main = 100.0 * correct_main / total
    acc_streams = [100.0 * c / total for c in correct_streams] if is_multistream else []
    return acc_main, acc_streams

# ------------------------------------------------------------------------------
# 5. Main Training Routine (Model 1 First, Then Model 2)
# ------------------------------------------------------------------------------
def main():
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("  STUDY S4-009: FAIR CIFAR-100 BENCHMARK (EXACT 27-CANDIDATE EQUIVALENCE)")
    print(f"  Compute Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 100)

    # 1. Dataset Pre-Caching (Eliminate CPU Bottleneck)
    print("Loading HuggingFace dataset 'uoft-cs/cifar100'...")
    raw = load_dataset("uoft-cs/cifar100")

    print("Pre-caching dataset into contiguous RAM tensors (zero-CPU DataLoader)...")
    t0_cache = time.time()
    def cache_split(split):
        imgs, labels = [], []
        for item in raw[split]:
            img = item.get("img", item.get("image"))
            imgs.append(TF.pil_to_tensor(img))
            labels.append(item.get("fine_label", item.get("label")))
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)

    X_train, Y_train = cache_split("train")
    X_test, Y_test = cache_split("test")
    print(f"Pre-caching completed in {time.time() - t0_cache:.2f}s! Train: {X_train.shape} | Test: {X_test.shape}\n")

    batch_size = 128
    epochs = 20
    train_loader = DataLoader(CachedCifar100(X_train, Y_train, is_train=True), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(CachedCifar100(X_test, Y_test, is_train=False), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # =========================================================================
    # JOB 1: Multi-Stream Consensus ViT (Runs First!)
    # =========================================================================
    print("=" * 100)
    print("  [JOB 1/2] TRAINING: Multi-Stream Consensus ViT (M=3, p_peaks=4 -> 3x9 = 27 candidates, Crippled d_mlp=128)")
    print("=" * 100)

    torch.manual_seed(seed)
    model_ms = BatchedMultiStreamBilateralViT(
        num_classes=100, seq_len=256, d_model=128, n_heads=4,
        d_mlp=128,          # Crippled 1x MLP
        num_streams=3,
        hops_per_phase=4,
        p_peaks=4,          # K = 1 + 2*4 = 9 candidates per stream (Total = 27)
    ).to(device)

    params_ms = sum(p.numel() for p in model_ms.parameters() if p.requires_grad)
    print(f"Multi-Stream ViT Trainable Parameters: {params_ms:,}")

    opt_ms = torch.optim.AdamW(model_ms.parameters(), lr=1e-3, weight_decay=0.05)
    sched_ms = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ms, T_max=epochs, eta_min=1e-4)

    t0 = time.time()
    history_ms = []

    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        model_ms.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt_ms.zero_grad(set_to_none=True)
            logits = model_ms(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_ms.parameters(), 1.0)
            opt_ms.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += labels.size(0)

        sched_ms.step()
        ep_loss = train_loss / train_total
        ep_acc = 100.0 * train_correct / train_total
        test_acc, stream_accs = evaluate(model_ms, test_loader, device, is_multistream=True, num_streams=3)
        ep_duration = time.time() - t_ep_start

        streams_str = " | ".join([f"S{i+1}: {a:5.2f}%" for i, a in enumerate(stream_accs)])
        print(
            f"  [Multi-Stream] Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {ep_loss:.4f} | Train Acc: {ep_acc:5.2f}% | "
            f"CONSENSUS TEST ACC: {test_acc:5.2f}% | [{streams_str}] | "
            f"Epoch Time: {ep_duration:4.1f}s"
        )
        history_ms.append({"epoch": epoch, "loss": round(ep_loss, 4), "train_acc": round(ep_acc, 2), "test_acc": round(test_acc, 2), "streams": [round(a, 2) for a in stream_accs], "epoch_sec": round(ep_duration, 1)})

    time_ms = time.time() - t0
    final_acc_ms = test_acc
    print(f"--> [JOB 1 FINISHED] Multi-Stream Final Test Accuracy: {final_acc_ms:.2f}% (Total Time: {time_ms:.1f}s)\n")

    # =========================================================================
    # JOB 2: Compensated Single-Stream Baseline (Runs Second!)
    # =========================================================================
    print("=" * 100)
    print("  [JOB 2/2] TRAINING: Compensated Single-Stream ViT (M=1, p_peaks=13 -> 1x27 = 27 candidates, Full d_mlp=512)")
    print("=" * 100)

    torch.manual_seed(seed)
    model_base = CompensatedSingleStreamViT(
        num_classes=100, seq_len=256, d_model=128, n_heads=4,
        d_mlp=512,          # Full 4x MLP
        hops_per_phase=4,
        p_peaks=13,         # K = 1 + 2*13 = 27 candidates (Total = 27)
    ).to(device)

    params_base = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
    print(f"Compensated Baseline Trainable Parameters: {params_base:,}")

    opt_base = torch.optim.AdamW(model_base.parameters(), lr=1e-3, weight_decay=0.05)
    sched_base = torch.optim.lr_scheduler.CosineAnnealingLR(opt_base, T_max=epochs, eta_min=1e-4)

    t0_base = time.time()
    history_base = []

    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        model_base.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt_base.zero_grad(set_to_none=True)
            logits = model_base(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_base.parameters(), 1.0)
            opt_base.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += labels.size(0)

        sched_base.step()
        ep_loss = train_loss / train_total
        ep_acc = 100.0 * train_correct / train_total
        test_acc, _ = evaluate(model_base, test_loader, device, is_multistream=False)
        ep_duration = time.time() - t_ep_start

        print(
            f"  [Compensated Baseline] Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {ep_loss:.4f} | Train Acc: {ep_acc:5.2f}% | "
            f"BASELINE TEST ACC: {test_acc:5.2f}% | "
            f"Epoch Time: {ep_duration:4.1f}s"
        )
        history_base.append({"epoch": epoch, "loss": round(ep_loss, 4), "train_acc": round(ep_acc, 2), "test_acc": round(test_acc, 2), "epoch_sec": round(ep_duration, 1)})

    time_base = time.time() - t0_base
    final_acc_base = test_acc
    print(f"--> [JOB 2 FINISHED] Baseline Final Test Accuracy: {final_acc_base:.2f}% (Total Time: {time_base:.1f}s)\n")

    # =========================================================================
    # DEFINITIVE HEAD-TO-HEAD SCORECARD
    # =========================================================================
    print("=" * 100)
    print("  DEFINITIVE HEAD-TO-HEAD COMPARISON SCORECARD")
    print("=" * 100)
    print(f"  {'Model Name':<35} | {'Candidates/Hop':<15} | {'MLP Hidden':<12} | {'Params':<10} | {'Test Acc':<10} | {'Time (s)':<10}")
    print(f"  {'-'*35}-+-{'-'*15}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'Multi-Stream Consensus ViT (M=3)':<35} | {'27 (3 x 9)':<15} | {'128 (1x)':<12} | {params_ms:<10,} | {final_acc_ms:5.2f}%    | {time_ms:5.1f}s")
    print(f"  {'Compensated Single-Stream Baseline':<35} | {'27 (1 x 27)':<15} | {'512 (4x)':<12} | {params_base:<10,} | {final_acc_base:5.2f}%    | {time_base:5.1f}s")
    print("=" * 100)

    gain = final_acc_ms - final_acc_base
    print(f"  NET MULTI-STREAM EXPERT ADVANTAGE: {gain:+.2f}%")
    if gain > 0:
        print("  VERDICT: Parallel Multi-Stream Consensus wins, even with crippled MLPs!")
    else:
        print("  VERDICT: Single-stream attention scales effectively when given equal candidates.")
    print("=" * 100)

    # Save Results
    results = {
        "experiment": "S4-009_fair_cifar100",
        "multistream_model": {
            "name": "MultiStreamConsensusViT",
            "parameters": params_ms,
            "candidates_per_hop": 27,
            "mlp_hidden": 128,
            "final_test_acc": round(final_acc_ms, 2),
            "runtime_seconds": round(time_ms, 1),
            "history": history_ms,
        },
        "baseline_model": {
            "name": "CompensatedSingleStreamViT",
            "parameters": params_base,
            "candidates_per_hop": 27,
            "mlp_hidden": 512,
            "final_test_acc": round(final_acc_base, 2),
            "runtime_seconds": round(time_base, 1),
            "history": history_base,
        },
        "net_advantage": round(gain, 2),
    }

    with open("s4_009_fair_cifar100_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to 's4_009_fair_cifar100_results.json'")

if __name__ == "__main__":
    main()
