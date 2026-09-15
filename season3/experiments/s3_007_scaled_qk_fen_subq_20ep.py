"""S2-048: Scaled FEN-SubQ ViT with Q.K Dot-Product Attention (2x Parameters, 20 Epochs).

Configuration:
- d_model = 192 (Parameters: 372,032 - exact ~2x scaling over S2-047's 183,552)
- Epochs = 20
- Sequence Length L = 257 (2x2 patches on 32x32 CIFAR-100 + 1 CLS token)
- Phase 1: Causal cuDNN GRU Scanner + FEN Feature Gating + Pure Bag Escrow Vault (torch.cumsum)
- Phase 2: Content-Dependent Q.K Dot-Product Attention:
  * Query: LN_q(state)
  * Keys: LN_k(E_cand)
  * Values: E_cand
  * Logit Bias: Continuous 12-harmonic carrier wave peak values directly in attention softmax
  * Recurrent Residual Accumulation: state = state + (1/sqrt(n_hops)) * context
  * Autonomous Wave Annealing: w_{t+1} = w_t + 0.1 * MLP(w_t)
- Persistence: Checkpoint (/models/s2_048_checkpoint.pt) and progress (/models/s2_048_progress.json)
  saved and committed to Modal volume after every single epoch. Automatic resume enabled.
"""

import json
import math
import os
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "torchvision>=0.17.0", "datasets>=2.18.0", "numpy", "pillow"
)
app = modal.App("season2-s2-048-fen-subq-scaled-20ep")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_scaled_fen_subq_experiment(
    seed: int = 42,
    epochs: int = 20,
    n_hops: int = 4,
    k_peaks: int = 8,
    num_waves: int = 12,
    d_model: int = 192,
    resume: bool = True,
):
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

    batch_size = 128
    seq_len = 257  # 256 patches (2x2, stride 2 on 32x32) + 1 CLS token
    num_classes = 100

    print("=" * 90)
    print(f"  S2-048: Scaled FEN-SubQ ViT (Q.K Dot-Product Attention) on High-Res CIFAR-100")
    print(f"  Seq Len L = {seq_len}, Dim = {d_model} (2x Scale), Epochs = {epochs}, Hops = {n_hops}, K = {k_peaks} Peaks")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 90)

    # 1. Dataset
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
            self.items, self.transform = raw[split_name], transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            item = self.items[index]
            return self.transform(item.get("img", item.get("image"))), item.get(
                "fine_label", item.get("label")
            )

    train_loader = DataLoader(
        Cifar("train", train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Cifar("test", test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Scaled FEN-SubQ with Content-Dependent Q.K Attention Readout
    class ScaledFENSubQViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.k_peaks = k_peaks
            self.num_waves = num_waves
            self.n_hops = n_hops
            self.scale = 1.0 / math.sqrt(n_hops)

            # Patch projection
            self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

            # Phase 1: Fused Causal GRU Scanner + FEN Feature Extraction
            self.scanner = nn.GRU(d_model, d_model, batch_first=True)
            self.gate = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)

            # Harmonic Wave Router (1 shared bank of 12 carriers)
            self.init_wave_latent = nn.Parameter(torch.randn(num_waves * 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4),
            )

            # Frequency spectrum covering short to long distances (Study 61 canon)
            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, num_waves))
            self.max_d = seq_len
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, self.max_d - 1, 1))

            # Phase 2: Q.K Attention Normalization
            self.ln_q = nn.LayerNorm(d_model)
            self.ln_k = nn.LayerNorm(d_model)

            # Output head
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes, bias=False)

        def forward(self, x, return_diagnostics=False):
            bsz = x.shape[0]
            tokens = self.patch(x).flatten(2).transpose(1, 2)  # [B, 256, D]
            tokens = torch.cat([self.cls.expand(bsz, -1, -1), tokens], dim=1) + self.pos  # [B, 257, D]

            # -------------------------------------------------------------
            # PHASE 1: Fused Causal GRU Scan + Pure Bag Escrow Vault (cumsum)
            # -------------------------------------------------------------
            H, _ = self.scanner(tokens)                # [B, L, D]
            G = torch.sigmoid(self.gate(H))            # [B, L, D]
            V = self.v_proj(G * H)                     # [B, L, D]
            E_all = torch.cumsum(V, dim=1)             # [B, L, D] (immutable feature escrow vault)

            # -------------------------------------------------------------
            # PHASE 2: Content-Dependent Q.K Dot-Product Attention Hops
            # -------------------------------------------------------------
            state = H
            curr_wave = self.init_wave_latent
            offsets_log = []
            attn_entropy_log = []
            pos_grid = torch.arange(seq_len, device=x.device).unsqueeze(1)  # [L, 1]

            for hop in range(self.n_hops):
                # 1. Harmonic Carrier Wave Evaluation
                params = curr_wave.view(self.num_waves, 4)
                amp = torch.tanh(params[:, 0]).view(1, 1, self.num_waves)
                omega = (F.softplus(params[:, 1]).view(1, 1, self.num_waves) * self.base_freqs)
                phi = (params[:, 2] * math.pi).view(1, 1, self.num_waves)
                decay = (F.softplus(params[:, 3]) * 0.05).view(1, 1, self.num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).squeeze(0)  # [max_d - 1]

                # 2. Extract Top-K Peak Offsets + continuous peak_vals
                topk_vals, past_offsets = torch.topk(wave_1d, k=self.k_peaks - 1, dim=-1)
                past_offsets = past_offsets + 1
                zero_off = torch.zeros(1, dtype=torch.long, device=x.device)
                zero_val = torch.zeros(1, dtype=torch.float, device=x.device)

                active_offsets = torch.cat([zero_off, past_offsets])  # [K]
                peak_vals = torch.cat([zero_val, topk_vals])           # [K]

                # 3. Dynamic candidate escrow vectors: [B, L, K, D]
                targets = (pos_grid - active_offsets.unsqueeze(0)) % seq_len
                E_cand = E_all[:, targets, :]  # [B, L, K, D]

                # 4. True Content-Dependent Q.K Attention:
                # Query = Current token state s (evolving across hops)
                # Keys = Escrow representations E_cand at candidate offsets
                # Values = Escrow representations E_cand
                q = self.ln_q(state)           # [B, L, D]
                k = self.ln_k(E_cand)          # [B, L, K, D]
                v = E_cand                     # [B, L, K, D]

                # Dot-product similarity scaled by sqrt(D)
                scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(d_model)  # [B, L, K]

                # Add harmonic wave carrier values directly as logit bias!
                scores = scores + peak_vals.view(1, 1, self.k_peaks)

                # Standard softmax attention over candidate escrow offsets
                weights = F.softmax(scores, dim=-1)  # [B, L, K]

                # Aggregate attended escrow values
                context = (weights.unsqueeze(-1) * v).sum(dim=2)  # [B, L, D]

                # Recurrent residual accumulation into token state
                state = state + self.scale * context

                if return_diagnostics:
                    offsets_log.append(active_offsets.detach().cpu().tolist())
                    entropy = -(weights * (weights + 1e-9).log()).sum(dim=-1).mean().item()
                    attn_entropy_log.append(round(entropy, 3))

                # 5. Autonomous spatial annealing of wave router across hops
                if hop < self.n_hops - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            logits = self.head(self.ln_f(state)[:, 0])
            if return_diagnostics:
                return logits, offsets_log, attn_entropy_log
            return logits

    model = ScaledFENSubQViT().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params:,} (~2x scaled over S2-047)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    ckpt_path = "/models/s2_048_checkpoint.pt"
    progress_path = "/models/s2_048_progress.json"
    start_epoch = 1
    epoch_records = []

    # Resume from checkpoint if present
    if resume and os.path.exists(ckpt_path) and os.path.exists(progress_path):
        try:
            print(f"\n[RESUME] Loading checkpoint from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1

            with open(progress_path, "r") as f:
                epoch_records = json.load(f)

            print(f"[RESUME] Resumed successfully from Epoch {ckpt['epoch']}! Starting Epoch {start_epoch}.\n")
        except Exception as e:
            print(f"[RESUME WARNING] Failed to load checkpoint: {e}. Starting fresh.")
            start_epoch = 1
            epoch_records = []

    t0_train = time.time()

    for ep in range(start_epoch, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0_ep = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += imgs.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = total_loss / total

        # Evaluation & Diagnostics
        model.eval()
        test_correct, test_total = 0, 0
        last_offsets, last_entropy = None, None
        with torch.no_grad():
            for idx, (imgs, labels) in enumerate(test_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                if idx == 0:
                    logits, last_offsets, last_entropy = model(imgs, return_diagnostics=True)
                else:
                    logits = model(imgs)
                pred = logits.argmax(dim=-1)
                test_correct += (pred == labels).sum().item()
                test_total += imgs.size(0)

        test_acc = 100.0 * test_correct / test_total
        ep_time = time.time() - t0_ep

        print(
            f"  Epoch {ep:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}% | Time: {ep_time:.1f}s",
            flush=True
        )
        if last_offsets:
            for h in range(n_hops):
                print(f"    Hop {h+1} Offsets: {last_offsets[h]} | Attn Entropy: {last_entropy[h]} (max={math.log(k_peaks):.2f})", flush=True)

        record = {
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_acc": round(test_acc, 2),
            "offsets_per_hop": last_offsets,
            "attn_entropy_per_hop": last_entropy,
            "time_s": round(ep_time, 2),
        }
        epoch_records.append(record)

        # SAVE CHECKPOINT & PROGRESS EVERY SINGLE EPOCH
        try:
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "test_acc": test_acc,
            }, ckpt_path)

            with open(progress_path, "w") as f:
                json.dump(epoch_records, f, indent=2)

            volume.commit()
            print(f"  [SAVED & COMMITTED] Epoch {ep} checkpoint & progress saved to persistent volume.", flush=True)
        except Exception as e:
            print(f"  [SAVE WARNING] Failed to commit to volume: {e}", flush=True)

    total_time = time.time() - t0_train
    print("-" * 90)
    print(f"Final Test Accuracy: {epoch_records[-1]['test_acc']:.2f}% in {total_time:.1f}s")
    print("-" * 90)

    return {
        "model": "scaled_fen_subq_dot_product_vit",
        "seed": seed,
        "parameters": num_params,
        "d_model": d_model,
        "epochs": epochs,
        "n_hops": n_hops,
        "k_peaks": k_peaks,
        "epoch_records": epoch_records,
        "final_test_acc": epoch_records[-1]["test_acc"],
        "total_time_s": round(total_time, 2),
    }


@app.local_entrypoint()
def main():
    print("\n>>> Launching S2-048 Scaled FEN-SubQ ViT (d_model=192, 20 Epochs) on Modal A10G...")
    res = run_scaled_fen_subq_experiment.remote(
        seed=42,
        epochs=20,
        n_hops=4,
        k_peaks=8,
        num_waves=12,
        d_model=192,
        resume=True,
    )
    print(f"\n>>> Completed S2-048: Final Accuracy = {res['final_test_acc']}%\n")

    print("=" * 95)
    print("  S2-048 SCALED FEN-SUBQ (2X PARAMS, 20 EPOCHS) FINAL RESULTS")
    print("=" * 95)
    print(f"{'Architecture':<35} | {'Params':<10} | {'Epochs':<8} | {'Ep 1 Acc':<10} | {'Final Acc':<10} | {'Time (s)':<10}")
    print("-" * 95)
    print(f"{'S2-030 Canonical SubQ':<35} | {'229,120':<10} | {'10':<8} | {'14.5%':<10} | {'32.05%':<10} | {'325s':<10}")
    print(f"{'S2-047 FEN-SubQ (1x Scale)':<35} | {'183,552':<10} | {'10':<8} | {'14.07%':<10} | {'37.36%':<10} | {'238.8s':<10}")
    print(f"{'S2-048 Scaled FEN-SubQ (2x Scale)':<35} | {res['parameters']:<10,} | {res['epochs']:<8} | {res['epoch_records'][0]['test_acc']:<10.2f}% | {res['final_test_acc']:<10.2f}% | {res['total_time_s']:<10.1f}s")
    print("=" * 95)

    os.makedirs("season2/results", exist_ok=True)
    with open("season2/results/s2_048_fen_subq_scaled_20ep.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved results to season2/results/s2_048_fen_subq_scaled_20ep.json")
