"""
==========================================================================================
  STUDY S4-008: MULTI-STREAM CONSENSUS SUBQ VIT (M=3, K=3, T=8) ON CIFAR-100
==========================================================================================
Architectural Paradigm:
- Dataset: HuggingFace 'uoft-cs/cifar100' (Pre-cached in RAM, 0-overhead streaming).
- 2D Vision Grid: 16x16 = 256 spatial patch tokens (2x2 patches from 32x32 image).
- M = 3 Completely Autonomous, Parallel Vision Streams (Untied QKV, WaveRouter, MLPs).
- Bilateral Wave Routing: Each stream selects p_peaks=1 (K=3 bilateral candidates: [0, +Δ, -Δ]).
  Total attention lookups across all streams: M * K * T = 3 * 3 * 8 = 72 (only 28% of Dense!).
- Direct Tensor Slicing: K[:, :, clamped_targets, :] in 1 CUDA kernel call (zero 5D gather).
- Pure Optical Transport (S4-005 Law): state = context during 4-hop runway (no W_o, no additive residual).
- Cross-Stream Mixer MLP:
    s_common = mean(s^(m)) + MixerMLP(LayerNorm(cat([s^(1), s^(2), s^(3)])))
  Fuses multi-scale spatial perspectives non-linearly at T=4 and T=8.
- Real-time logging of Consensus Accuracy vs Individual Stream Accuracies at every epoch!
==========================================================================================
"""

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
# 1. Bilateral Harmonic Wave Router (Direct Spatial Slicing, Zero Gather)
# ------------------------------------------------------------------------------
class FastBilateralWaveRouter(nn.Module):
    def __init__(self, num_waves=12, p_peaks=1, max_d=256):
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
        omega = F.softplus(curr_params[..., 1]).view(1, 1, 1, self.num_waves) * self.base_freqs
        phi = curr_params[..., 2] * math.pi
        phi = phi.view(1, 1, 1, self.num_waves)
        decay = F.softplus(curr_params[..., 3]) * 0.05
        decay = decay.view(1, 1, 1, self.num_waves)

        # Synthesize continuous spatial wave interference pattern
        wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
        wave_1d = wave_comps.sum(dim=-1).squeeze()  # [max_d - 1]

        topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.p_peaks, dim=-1)
        past_peak_offsets = past_peak_offsets + 1

        zero_offset = torch.zeros(1, dtype=torch.long, device=device)
        zero_val = torch.zeros(1, dtype=torch.float, device=device)

        # Bilateral candidates: [anchor 0, backward -Delta, forward +Delta]
        offsets_1d = torch.cat([zero_offset, past_peak_offsets, -past_peak_offsets])
        peak_vals_1d = torch.cat([zero_val, topk_vals, topk_vals])

        next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
        return offsets_1d, peak_vals_1d, next_wave_latent

# ------------------------------------------------------------------------------
# 2. Autonomous Bilateral SubQ Stream (Clean Optical Transport, NO W_o)
# ------------------------------------------------------------------------------
class AutonomousBilateralSubQStream(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_mlp=256, num_waves=12, p_peaks=1, seq_len=256):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.p_peaks = p_peaks
        self.k_total = 1 + 2 * p_peaks
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.seq_len = seq_len

        self.ln_attn = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        self.router = FastBilateralWaveRouter(num_waves=num_waves, p_peaks=p_peaks, max_d=seq_len)
        self.register_buffer("pos_idx", torch.arange(seq_len).unsqueeze(1))

        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

    def run_optical_hops(self, state, wave_latent, n_hops=4):
        B, L, D = state.shape
        dev = state.device

        for _ in range(n_hops):
            z = self.ln_attn(state)
            offsets_1d, peak_vals_1d, wave_latent = self.router(wave_latent, dev)

            # Direct 2D Index Matrix [L, K]
            raw_targets = self.pos_idx - offsets_1d.unsqueeze(0)
            valid_mask = (raw_targets >= 0) & (raw_targets < L)
            clamped_targets = torch.clamp(raw_targets, 0, L - 1)

            Q = self.q(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            # Direct Fast Tensor Slicing in 1 single CUDA kernel call
            K_cand = K[:, :, clamped_targets, :]
            V_cand = V[:, :, clamped_targets, :]

            scores = (Q.unsqueeze(3) * K_cand).sum(dim=-1) * self.scale + peak_vals_1d.view(1, 1, 1, self.k_total)
            scores = scores.masked_fill(~valid_mask.view(1, 1, L, self.k_total), -1e4)
            weights = F.softmax(scores, dim=-1)

            context = (weights.unsqueeze(-1) * V_cand).sum(dim=3)
            # Pure optical transport: NO W_o projection, NO additive residual inside runway!
            state = context.transpose(1, 2).contiguous().view(B, L, D)

        # Delayed MLP at end of the 4-hop runway
        state = state + self.mlp(self.ln_mlp(state))
        return state, wave_latent

# ------------------------------------------------------------------------------
# 3. Cross-Stream Mixer MLP (Learns Inter-Stream Dialect Translation)
# ------------------------------------------------------------------------------
class StreamMixerMLP(nn.Module):
    def __init__(self, d_model=128, num_streams=3, d_mix=256):
        super().__init__()
        self.ln = nn.LayerNorm(num_streams * d_model)
        self.mlp = nn.Sequential(
            nn.Linear(num_streams * d_model, d_mix),
            nn.GELU(),
            nn.Linear(d_mix, d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, states):
        cat = torch.cat(states, dim=-1)
        mean = torch.stack(states, dim=0).mean(dim=0)
        return mean + self.mlp(self.ln(cat))

# ------------------------------------------------------------------------------
# 4. Multi-Stream Consensus Vision Transformer (M=3, K=3, T=8)
# ------------------------------------------------------------------------------
class MultiStreamConsensusViT(nn.Module):
    def __init__(self, num_classes=100, seq_len=256, d_model=128, n_heads=4, d_mlp=256, num_streams=3, hops_per_phase=4, p_peaks=1):
        super().__init__()
        self.num_streams = num_streams
        self.hops_per_phase = hops_per_phase
        self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.streams = nn.ModuleList([
            AutonomousBilateralSubQStream(d_model=d_model, n_heads=n_heads, d_mlp=d_mlp, p_peaks=p_peaks, seq_len=seq_len)
            for _ in range(num_streams)
        ])

        self.mixer = StreamMixerMLP(d_model=d_model, num_streams=num_streams, d_mix=d_mlp)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, img, return_individual=False):
        B = img.shape[0]
        # 32x32 image -> 16x16 tokens = 256 patches
        patches = self.patch(img).flatten(2).transpose(1, 2)
        x = patches + self.pos

        # PHASE 1: Hops 1 to 4 -> Dedicated MLPs
        states_p1 = []
        latents_p1 = []
        for stream in self.streams:
            s_out, w_out = stream.run_optical_hops(x, stream.router.init_wave_latent, n_hops=self.hops_per_phase)
            states_p1.append(s_out)
            latents_p1.append(w_out)

        # Midpoint Consensus Checkpoint at T=4
        s_common = self.mixer(states_p1)

        # PHASE 2: Hops 5 to 8 -> Dedicated MLPs
        states_p2 = []
        for i, stream in enumerate(self.streams):
            s_out, _ = stream.run_optical_hops(s_common, latents_p1[i], n_hops=self.hops_per_phase)
            states_p2.append(s_out)

        # Final Consensus at T=8
        s_final = self.mixer(states_p2)

        # Global Average Pooling across 256 spatial tokens
        pooled = self.ln_f(s_final).mean(dim=1)
        logits = self.head(pooled)

        if return_individual:
            individual_logits = [self.head(self.ln_f(s).mean(dim=1)) for s in states_p2]
            return logits, individual_logits

        return logits

# ------------------------------------------------------------------------------
# 5. Evaluation Routine (Consensus Accuracy vs Stream Accuracies)
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_cifar(model, loader, dev, num_streams=3):
    model.eval()
    correct_consensus = 0
    correct_streams = [0 for _ in range(num_streams)]
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
        logits_c, indiv_logits = model(imgs, return_individual=True)

        preds_c = logits_c.argmax(dim=-1)
        correct_consensus += (preds_c == labels).sum().item()

        for s_idx, s_logits in enumerate(indiv_logits):
            preds_s = s_logits.argmax(dim=-1)
            correct_streams[s_idx] += (preds_s == labels).sum().item()

        total += labels.size(0)

    acc_consensus = 100.0 * correct_consensus / total
    acc_streams = [100.0 * c / total for c in correct_streams]
    return acc_consensus, acc_streams

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
    print("=" * 95)
    print("  EXPERIMENT S4-008: MULTI-STREAM CONSENSUS SUBQ VIT ON CIFAR-100")
    print(f"  Compute Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)

    # 1. Dataset Setup via HuggingFace datasets (Official S3/S4 protocol)
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

    class Cifar(Dataset):
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

    batch_size = 128
    train_loader = DataLoader(Cifar("train", train_transform), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(Cifar("test", test_transform), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Dataset Loaded: Train: {len(raw['train']):,} images | Test: {len(raw['test']):,} images | Classes: 100")

    # Hyperparameters
    seq_len = 256
    d_model = 128
    n_heads = 4
    d_mlp = 256
    num_streams = 3
    p_peaks = 1        # K = 1 + 2*1 = 3 bilateral candidates per stream
    hops_per_phase = 4 # 4 hops -> mixer -> 4 hops -> mixer (Total T=8)
    epochs = 20

    model = MultiStreamConsensusViT(
        num_classes=100,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_mlp=d_mlp,
        num_streams=num_streams,
        hops_per_phase=hops_per_phase,
        p_peaks=p_peaks,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Multi-Stream Consensus ViT M={num_streams}, K=3] Trainable Parameters: {n_params:,}")
    print("Architectural Breakdown:")
    print(f"  • {num_streams} Independent Vision Streams x (QKV + BilateralRouter(K=3) + {d_mlp}-wide MLP)")
    print(f"  • Cross-Stream Mixer MLP: LN(3D) -> Linear(3D, {d_mlp}) -> GELU -> Linear({d_mlp}, D)")
    print(f"  • Lookups per patch token: M*K*T = {num_streams} * 3 * {hops_per_phase*2} = {num_streams*3*hops_per_phase*2} (only {round(100*num_streams*3*hops_per_phase*2/seq_len)}% of Dense!)")
    print("-" * 95)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    # Initial Pre-Training Accuracy (Random Guessing ~ 1.0%)
    init_acc, _ = evaluate_cifar(model, test_loader, device, num_streams=num_streams)
    print(f"Epoch  0/{epochs} | Initial Test Accuracy: {init_acc:.2f}% (Chance: 1.00%)")
    print("=" * 95)

    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        epoch_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # Evaluate on Test Set
        test_acc, stream_accs = evaluate_cifar(model, test_loader, device, num_streams=num_streams)
        elapsed = time.time() - t0

        streams_str = " | ".join([f"S{i+1}: {acc:5.2f}%" for i, acc in enumerate(stream_accs)])

        print(
            f"  Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:5.2f}% | "
            f"CONSENSUS TEST ACC: {test_acc:5.2f}% | "
            f"[{streams_str}] | "
            f"Time: {elapsed:5.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "train_acc": round(epoch_train_acc, 2),
            "consensus_test_acc": round(test_acc, 2),
            "stream_test_accs": [round(a, 2) for a in stream_accs],
            "elapsed_seconds": round(elapsed, 1),
        })

    print("=" * 95)
    print("TRAINING COMPLETE!")
    print(f"Final CONSENSUS Test Accuracy: {test_acc:.2f}%")
    for i, acc in enumerate(stream_accs):
        gain = test_acc - acc
        print(f"  Stream {i+1} Test Accuracy: {acc:.2f}% (Consensus Boost: +{gain:.2f}%)")
    print("=" * 95)

    result_data = {
        "experiment": "S4-008_cifar100",
        "architecture": "MultiStreamConsensusViT",
        "num_streams": num_streams,
        "parameters": n_params,
        "final_train_acc": round(epoch_train_acc, 2),
        "final_consensus_test_acc": round(test_acc, 2),
        "final_stream_test_accs": [round(a, 2) for a in stream_accs],
        "history": history,
    }

    output_path = "s4_008_cifar100_results.json"
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Saved full experiment history to '{output_path}'")

if __name__ == "__main__":
    main()
