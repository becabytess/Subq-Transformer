"""
==========================================================================================
  EXPERIMENT S4-010: ZERO-MLP CIFAR-100 SHOOTOUT
  1-LAYER SUBQ VIT (T=4, K=9, ZERO MLP) VS. 1-LAYER DENSE VIT (ZERO MLP)
==========================================================================================
Core Scientific Hypothesis:
- Does recurrent attention state accumulation alone provide expressive power on vision 
  (CIFAR-100) when there are ZERO MLPs anywhere in the architecture?
- Neither model has an MLP:
    1. SubQ ViT: 1 physical layer unrolled across T=4 recurrent hops with K=9 sparse 
       bilateral wave attention (NO MLP at hops, NO MLP at the end).
    2. Dense ViT: 1 standard physical layer of full L x L attention (NO MLP).
- Both models are parameter-matched (~103k vs ~113k parameters).
- Pre-cached RAM dataset (zero-CPU DataLoader) for instant, high-throughput execution.
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
# 1. Bilateral Harmonic Wave Router (Direct Spatial Slicing)
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

# ------------------------------------------------------------------------------
# 2. Architecture 1: SubQ ViT (1-Layer Recurrent, T=4, K=9, ZERO MLP ANYWHERE)
# ------------------------------------------------------------------------------
class SubQViT_NoMLP(nn.Module):
    def __init__(self, num_classes=100, seq_len=256, d_model=128, n_heads=4, T=4, p_peaks=4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.T = T
        self.p_peaks = p_peaks
        self.k_total = 1 + 2 * p_peaks
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.inv_sqrt_T = 1.0 / math.sqrt(T)

        # Patch Tokenizer (32x32 -> 16x16 = 256 tokens)
        self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Pure Attention Projections (Tied across T hops, NO MLP!)
        self.ln_attn = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        # Wave Router
        self.router = FastBilateralWaveRouter(num_waves=12, p_peaks=p_peaks, max_d=seq_len)
        self.register_buffer("pos_idx", torch.arange(seq_len).unsqueeze(1))

        # Classification Head (NO MLP!)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, img):
        B = img.shape[0]
        patches = self.patch(img).flatten(2).transpose(1, 2)
        s = patches + self.pos
        dev = img.device
        L = self.seq_len

        wave_latent = self.router.init_wave_latent

        # Recurrent Attention Loop (T=4 hops)
        for _ in range(self.T):
            z = self.ln_attn(s)
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
            attn_out = context.transpose(1, 2).contiguous().view(B, L, self.d_model)

            # Euler-discretized Recurrent State Update (NO MLP!)
            s = s + self.inv_sqrt_T * attn_out

        # Global Average Pooling -> Head
        pooled = self.ln_f(s).mean(dim=1)
        return self.head(pooled)

# ------------------------------------------------------------------------------
# 3. Architecture 2: Dense ViT (1-Layer Standard Full Attention, ZERO MLP ANYWHERE)
# ------------------------------------------------------------------------------
class DenseViT_NoMLP(nn.Module):
    def __init__(self, num_classes=100, seq_len=256, d_model=128, n_heads=4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Patch Tokenizer (32x32 -> 16x16 = 256 tokens)
        self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Dense Attention (NO MLP!)
        self.ln_attn = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model)

        # Classification Head (NO MLP!)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, img):
        B = img.shape[0]
        patches = self.patch(img).flatten(2).transpose(1, 2)
        x = patches + self.pos
        L = self.seq_len

        z = self.ln_attn(x)
        Q = self.q(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Full L x L Dense Attention Matrix
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        attn_out = (weights @ V).transpose(1, 2).contiguous().view(B, L, self.d_model)

        # Standard Attention Residual Connection (NO MLP!)
        x = x + self.c_proj(attn_out)

        # Global Average Pooling -> Head
        pooled = self.ln_f(x).mean(dim=1)
        return self.head(pooled)

# ------------------------------------------------------------------------------
# 4. High-Throughput RAM-Cached Dataset (Zero-CPU Bottleneck)
# ------------------------------------------------------------------------------
class CachedCifar100(Dataset):
    def __init__(self, X, Y, is_train=True):
        self.X = X
        self.Y = Y
        self.is_train = is_train
        self.mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        self.std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].float() / 255.0
        if self.is_train:
            img = F.pad(img, (4, 4, 4, 4), mode='reflect')
            top = torch.randint(0, 9, (1,)).item()
            left = torch.randint(0, 9, (1,)).item()
            img = img[:, top:top+32, left:left+32]
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, [2])

        img = (img - self.mean) / self.std
        return img, self.Y[idx]

# ------------------------------------------------------------------------------
# 5. Evaluation Routine
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

# ------------------------------------------------------------------------------
# 6. Main Training Routine
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
    print("  EXPERIMENT S4-010: ZERO-MLP CIFAR-100 SHOOTOUT")
    print("  1-LAYER SUBQ (T=4, K=9, NO MLP) VS. 1-LAYER DENSE (NO MLP)")
    print(f"  Compute Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 100)

    # 1. Dataset Pre-Caching
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
    # MODEL 1: SubQ ViT (T=4, K=9, ZERO MLP)
    # =========================================================================
    print("=" * 100)
    print("  [MODEL 1/2] TRAINING: SubQ ViT (1-Layer Recurrent, T=4 hops, K=9 candidates, ZERO MLP)")
    print("=" * 100)

    torch.manual_seed(seed)
    model_subq = SubQViT_NoMLP(
        num_classes=100, seq_len=256, d_model=128, n_heads=4,
        T=4, p_peaks=4
    ).to(device)

    params_subq = sum(p.numel() for p in model_subq.parameters() if p.requires_grad)
    print(f"SubQ (No-MLP) Trainable Parameters: {params_subq:,}")

    opt_subq = torch.optim.AdamW(model_subq.parameters(), lr=1e-3, weight_decay=0.05)
    sched_subq = torch.optim.lr_scheduler.CosineAnnealingLR(opt_subq, T_max=epochs, eta_min=1e-4)

    t0_subq = time.time()
    history_subq = []

    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        model_subq.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt_subq.zero_grad(set_to_none=True)
            logits = model_subq(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_subq.parameters(), 1.0)
            opt_subq.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += labels.size(0)

        sched_subq.step()
        ep_loss = train_loss / train_total
        ep_acc = 100.0 * train_correct / train_total
        test_acc = evaluate(model_subq, test_loader, device)
        ep_duration = time.time() - t_ep_start

        print(
            f"  [SubQ No-MLP] Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {ep_loss:.4f} | Train Acc: {ep_acc:5.2f}% | "
            f"TEST ACC: {test_acc:5.2f}% | "
            f"Epoch Time: {ep_duration:4.1f}s"
        )
        history_subq.append({"epoch": epoch, "loss": round(ep_loss, 4), "train_acc": round(ep_acc, 2), "test_acc": round(test_acc, 2), "epoch_sec": round(ep_duration, 1)})

    time_subq = time.time() - t0_subq
    final_acc_subq = test_acc
    print(f"--> [MODEL 1 FINISHED] SubQ (No-MLP) Final Test Accuracy: {final_acc_subq:.2f}% (Total Time: {time_subq:.1f}s)\n")

    # =========================================================================
    # MODEL 2: Dense ViT (1-Layer Standard Full Attention, ZERO MLP)
    # =========================================================================
    print("=" * 100)
    print("  [MODEL 2/2] TRAINING: Dense ViT (1-Layer Standard Full Attention, ZERO MLP)")
    print("=" * 100)

    torch.manual_seed(seed)
    model_dense = DenseViT_NoMLP(
        num_classes=100, seq_len=256, d_model=128, n_heads=4
    ).to(device)

    params_dense = sum(p.numel() for p in model_dense.parameters() if p.requires_grad)
    print(f"Dense (No-MLP) Trainable Parameters: {params_dense:,}")

    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=1e-3, weight_decay=0.05)
    sched_dense = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dense, T_max=epochs, eta_min=1e-4)

    t0_dense = time.time()
    history_dense = []

    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        model_dense.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt_dense.zero_grad(set_to_none=True)
            logits = model_dense(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_dense.parameters(), 1.0)
            opt_dense.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += labels.size(0)

        sched_dense.step()
        ep_loss = train_loss / train_total
        ep_acc = 100.0 * train_correct / train_total
        test_acc = evaluate(model_dense, test_loader, device)
        ep_duration = time.time() - t_ep_start

        print(
            f"  [Dense No-MLP] Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {ep_loss:.4f} | Train Acc: {ep_acc:5.2f}% | "
            f"TEST ACC: {test_acc:5.2f}% | "
            f"Epoch Time: {ep_duration:4.1f}s"
        )
        history_dense.append({"epoch": epoch, "loss": round(ep_loss, 4), "train_acc": round(ep_acc, 2), "test_acc": round(test_acc, 2), "epoch_sec": round(ep_duration, 1)})

    time_dense = time.time() - t0_dense
    final_acc_dense = test_acc
    print(f"--> [MODEL 2 FINISHED] Dense (No-MLP) Final Test Accuracy: {final_acc_dense:.2f}% (Total Time: {time_dense:.1f}s)\n")

    # =========================================================================
    # DEFINITIVE HEAD-TO-HEAD SCORECARD
    # =========================================================================
    print("=" * 100)
    print("  DEFINITIVE HEAD-TO-HEAD SCORECARD: ZERO-MLP ATTENTION SHOOTOUT")
    print("=" * 100)
    print(f"  {'Model Name':<35} | {'Lookups/Token':<15} | {'Parameters':<12} | {'Test Acc':<10} | {'Time (s)':<10}")
    print(f"  {'-'*35}-+-{'-'*15}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'SubQ ViT (T=4, K=9, Zero-MLP)':<35} | {'36 (4 x 9)':<15} | {params_subq:<12,} | {final_acc_subq:5.2f}%    | {time_subq:5.1f}s")
    print(f"  {'Dense ViT (1-Layer, Zero-MLP)':<35} | {'256 (1 x 256)':<15} | {params_dense:<12,} | {final_acc_dense:5.2f}%    | {time_dense:5.1f}s")
    print("=" * 100)

    delta = final_acc_subq - final_acc_dense
    print(f"  NET RECURRENT SUBQ ADVANTAGE (ZERO-MLP): {delta:+.2f}%")
    if delta > 0:
        print("  VERDICT: Multi-hop Recurrence delivers strong representation even without ANY MLP!")
    else:
        print("  VERDICT: Single dense projection is competitive without non-linear channel expansion.")
    print("=" * 100)

    # Save results to JSON
    results = {
        "experiment": "S4-010_zero_mlp_cifar100",
        "subq_no_mlp": {
            "name": "SubQViT_NoMLP",
            "T": 4,
            "K": 9,
            "parameters": params_subq,
            "final_test_acc": round(final_acc_subq, 2),
            "total_time_s": round(time_subq, 1),
            "history": history_subq,
        },
        "dense_no_mlp": {
            "name": "DenseViT_NoMLP",
            "layers": 1,
            "parameters": params_dense,
            "final_test_acc": round(final_acc_dense, 2),
            "total_time_s": round(time_dense, 1),
            "history": history_dense,
        },
        "recurrent_advantage": round(delta, 2),
    }

    with open("s4_010_zero_mlp_cifar100_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to 's4_010_zero_mlp_cifar100_results.json'")

if __name__ == "__main__":
    main()
