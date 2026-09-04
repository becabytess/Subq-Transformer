import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "datasets>=2.18.0",
        "numpy",
        "pillow"
    )
)

app = modal.App("study75-fen-subq-vit", image=image)
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)

@app.function(gpu="A10G", timeout=3600, volumes={"/models": volume})
def train_fen_subq_vit():
    import math, time, json, torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 75: FEN-SUBQ VISION TRANSFORMER (CHANNEL-ROLL ESCROW + SPARSE HARMONIC WAVES)")
    print("  High-Res CIFAR-100 (L = 257 Patches, K = 8 Wave Peaks, T = 4 Thought Hops)")
    print("  Dual-Pathway: Active Harmonic Recurrence (s) + Speculative Channel-Roll Escrow Vault (E)")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Dataset
    print("\n[1/3] Loading CIFAR-100 Dataset...")
    hf_dataset = load_dataset("uoft-cs/cifar100")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    class Cifar100Wrapper(Dataset):
        def __init__(self, split_data, transform=None):
            self.data = split_data
            self.transform = transform
            self.has_img_key = 'img' in self.data.column_names
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            img = item['img'] if self.has_img_key else item['image']
            label = item['fine_label'] if 'fine_label' in item else item['label']
            if self.transform: img = self.transform(img)
            return img, label

    train_loader = DataLoader(Cifar100Wrapper(hf_dataset['train'], transform=transform_train), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(Cifar100Wrapper(hf_dataset['test'], transform=transform_test), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    d_model, n_heads = 192, 6
    head_dim = d_model // n_heads # 32
    patch_size = 2 # 2x2 patches -> 16x16 = 256 patches
    num_patches = (32 // patch_size) ** 2 # 256
    seq_len = num_patches + 1 # 257 tokens (with [CLS])
    num_classes = 100
    K_peaks = 8
    num_waves = 12
    epochs = 20
    T_train = 4

    print(f"High-Res ViT Config: Patch Size {patch_size}x{patch_size} -> {num_patches} patches (Seq Len L = {seq_len}), Dim={d_model}, Heads={n_heads}, T={T_train}")

    # -------------------------------------------------------------------------
    # 2. Patch Embedding
    # -------------------------------------------------------------------------
    class HighResPatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        def forward(self, x):
            B = x.shape[0]
            x = self.proj(x).flatten(2).transpose(1, 2)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            return torch.cat((cls_tokens, x), dim=1) + self.pos_embed

    # -------------------------------------------------------------------------
    # 3. FEN-SubQ ViT Model (Channel-Roll Escrow Vault + Harmonic Waves)
    # -------------------------------------------------------------------------
    class FENSubQViT(nn.Module):
        def __init__(self, T=4):
            super().__init__()
            self.T = T
            self.patch_embed = HighResPatchEmbed()

            # SubQ Attention & MLP Core
            self.ln_1 = nn.LayerNorm(d_model)
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)

            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )

            # Continuous Spatial Wave Parameters
            init_latents = torch.zeros(n_heads, num_waves, 4)
            init_latents[..., 0] = 0.5
            init_latents[..., 3] = 0.1
            self.init_wave_latent = nn.Parameter(init_latents.view(n_heads, num_waves * 4))
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4)
            )

            log_freqs = torch.linspace(math.log10(math.pi / 1.0), math.log10(math.pi / 64.0), num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.max_d = seq_len
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, 1, self.max_d - 1, 1))

            # FEN Escrow Extraction & Channel-Roll Vault
            self.fen_gate = nn.Linear(d_model, d_model)
            self.fen_v_proj = nn.Linear(d_model, d_model)
            self.fen_roll_gate = nn.Linear(d_model, 1)

            # Final Joint Decision Head: Active State [s] + Escrow Vault [E]
            self.ln_f_s = nn.LayerNorm(d_model)
            self.ln_f_e = nn.LayerNorm(d_model)
            self.head = nn.Linear(2 * d_model, num_classes)

        def forward(self, x):
            B = x.shape[0]
            s = self.patch_embed(x) # [B, L, d_model]
            L = s.shape[1]

            # Initialize Escrow Vault
            E = s.new_zeros(B, L, d_model)

            curr_wave = self.init_wave_latent
            inv_sqrt_T = 1.0 / math.sqrt(self.T)
            q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)

            for t in range(self.T):
                s_norm = self.ln_1(s)

                # 1. Full Evolving Q, K, V
                qkv = self.c_attn(s_norm).chunk(3, dim=-1)
                q, k, v = [t_tensor.view(B, L, n_heads, head_dim).transpose(1, 2) for t_tensor in qkv]

                # 2. Continuous Spatial Wave Carrier
                params = curr_wave.view(n_heads, num_waves, 4)
                amp = torch.tanh(params[..., 0]).view(1, n_heads, 1, num_waves)
                omega = (F.softplus(params[..., 1]).view(1, n_heads, 1, num_waves) * self.base_freqs)
                phi = (params[..., 2] * math.pi).view(1, n_heads, 1, num_waves)
                decay = (F.softplus(params[..., 3]) * 0.05).view(1, n_heads, 1, num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

                # 3. Extract Top-K Peak Offsets
                topk_vals, past_peak_offsets = torch.topk(wave_1d, k=K_peaks - 1, dim=-1)
                past_peak_offsets = past_peak_offsets + 1
                zero_offset = torch.zeros((B, n_heads, 1), dtype=torch.long, device=x.device)
                zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=x.device)
                peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
                peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

                # 4. Gather Keys & Values
                target_indices = (q_pos - peak_offsets.unsqueeze(2)) % L
                idx_exp = target_indices.unsqueeze(-1).expand(B, n_heads, L, K_peaks, head_dim)

                K_g = torch.gather(k.unsqueeze(3).expand(B, n_heads, L, K_peaks, head_dim), dim=2, index=idx_exp)
                V_g = torch.gather(v.unsqueeze(3).expand(B, n_heads, L, K_peaks, head_dim), dim=2, index=idx_exp)

                # 5. Attention with Harmonic Logit Bias
                scores = (q.unsqueeze(3) * K_g).sum(dim=-1) / math.sqrt(head_dim) + peak_vals.unsqueeze(2)
                attn = F.softmax(scores, dim=-1)
                attn_out = (attn.unsqueeze(-1) * V_g).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
                attn_out = self.c_proj(attn_out)

                # 6. Recurrent Proposal
                s_prop = s + inv_sqrt_T * attn_out
                s_prop = s_prop + inv_sqrt_T * self.mlp(self.ln_2(s_prop))

                # 7. FEN Feature Escrow Absorption (Deplete OFF: roll_nodep)
                g_escrow = torch.sigmoid(self.fen_gate(s_prop))
                D_escrow = g_escrow * s_prop
                v_escrow = self.fen_v_proj(D_escrow)

                # Active state continues unconstrained (no depletion)
                s = s_prop

                # Channel-Roll Escrow Vault Update
                gamma = torch.sigmoid(self.fen_roll_gate(s_prop)) # [B, L, 1]
                E = (1.0 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + v_escrow

                # 8. Wave Transition
                if t < self.T - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            # Joint Decision Head: evaluates active final state [s] and escrow vault [E]
            s_cls = self.ln_f_s(s[:, 0])
            e_cls = self.ln_f_e(E[:, 0])
            joint_cls = torch.cat([s_cls, e_cls], dim=-1)
            return self.head(joint_cls)

    model = FENSubQViT(T=T_train).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2/3] Model Initialized: {param_count:,} Parameters (1 Layer FEN-SubQ, T={T_train} Hops, Epochs={epochs})")

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    ckpt_path = "/models/study75_fen_subq_checkpoint.pt"
    start_epoch = 1
    total_train_time = 0.0

    if os.path.exists(ckpt_path):
        print(f"--> Found existing checkpoint at {ckpt_path}! Resuming training...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        total_train_time = ckpt.get('total_time', 0.0)
        print(f"--> Resumed from Epoch {start_epoch} (Previous Time: {total_train_time:.1f}s)!")

    print(f"\n[3/3] Training FEN-SubQ ViT (Epochs {start_epoch} to {epochs})...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        ep_start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += images.size(0)

        ep_time = time.time() - ep_start
        total_train_time += ep_time
        train_acc = (train_correct / train_total) * 100.0
        print(f"  Epoch {epoch:>2d}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss/train_total:.4f}, Train Acc: {train_acc:>5.2f}%")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_acc': train_acc,
            'total_time': total_train_time
        }, ckpt_path)
        volume.commit()

    # Final Test Set Evaluation
    model.eval()
    test_loss, test_correct_top1, test_correct_top5, test_total = 0.0, 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, top1 = outputs.max(1)
            test_correct_top1 += top1.eq(labels).sum().item()
            _, top5 = outputs.topk(5, dim=1)
            test_correct_top5 += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            test_total += images.size(0)

    top1_acc = (test_correct_top1 / test_total) * 100.0
    top5_acc = (test_correct_top5 / test_total) * 100.0
    avg_loss = test_loss / test_total

    print("\n" + "=" * 115)
    print("  STUDY 75 FINAL RESULTS: FEN-SUBQ VISION TRANSFORMER (T=4 HOPS)")
    print("=" * 115)
    print(f"  Top-1 Test Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Test Accuracy: {top5_acc:.2f}%")
    print(f"  Test Cross-Entropy:  {avg_loss:.4f}")
    print(f"  Total Training Time: {total_train_time:.1f}s ({total_train_time/epochs:.1f}s/epoch)")
    print(f"  Parameters:          {param_count:,}")
    print("=" * 115)

    res = {
        "name": "FEN-SubQ ViT (T=4, K=8, Channel-Roll Escrow)",
        "params": param_count,
        "top1": top1_acc,
        "top5": top5_acc,
        "loss": avg_loss,
        "time": total_train_time
    }
    with open("/models/study75_fen_subq_results.json", "w") as f:
        json.dump(res, f)
    volume.commit()
    return res

@app.local_entrypoint()
def main():
    train_fen_subq_vit.remote()
