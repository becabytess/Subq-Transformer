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

app = modal.App("study78-param-matched-unidir-gru-subq-t8", image=image)
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)

@app.function(gpu="A10G", timeout=3600, volumes={"/models": volume})
def train_unidir_matched_gru_subq_t8():
    import math, time, json, torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 78: EXACT PARAMETER-MATCHED UNIDIRECTIONAL GRU + SUBQ WAVE ATTENTION (T = 8 HOPS)")
    print("  High-Res CIFAR-100 (L = 257 Patches, K = 8 Wave Peaks, T = 8 Thought Hops)")
    print("  Strictly Parameter-Matched to ~520k (MLP dim = 192, Unidirectional GRU dim = 192)")
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
    mlp_dim = 192 # Parameter matching: 1.0x d_model so total params ~ 520,980
    patch_size = 2 # 2x2 patches -> 16x16 = 256 patches
    num_patches = (32 // patch_size) ** 2 # 256
    seq_len = num_patches + 1 # 257 tokens (with [CLS])
    num_classes = 100
    K_peaks = 8
    num_waves = 12
    epochs = 20
    T_train = 8

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
    # 3. Parameter-Matched Unidirectional GRU + SubQ ViT (T=8)
    # -------------------------------------------------------------------------
    class ParamMatchedUnidirGRUSubQViT(nn.Module):
        def __init__(self, T=8):
            super().__init__()
            self.T = T
            self.patch_embed = HighResPatchEmbed()

            # Phase 1: Strictly Unidirectional Forward CuDNN GRU Scan
            self.gru_scan = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
            self.ln_gru = nn.LayerNorm(d_model)

            # Phase 2: 1-Layer SubQ Attention & Parameter-Matched MLP
            self.ln_1 = nn.LayerNorm(d_model)
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)

            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, d_model)
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

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            B = x.shape[0]
            s = self.patch_embed(x) # [B, L, d_model]
            L = s.shape[1]

            # Phase 1: Fast O(L) Strictly Unidirectional Forward Scan
            gru_out, _ = self.gru_scan(s)
            s = self.ln_gru(s + gru_out)

            # Phase 2: SubQ Multi-Hop Wave Relaxation (T Hops)
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

                # 6. Recurrent Residual Update
                s = s + inv_sqrt_T * attn_out
                s = s + inv_sqrt_T * self.mlp(self.ln_2(s))

                # 7. Wave Transition
                if t < self.T - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            s_final = self.ln_f(s)
            return self.head(s_final[:, 0])

    model = ParamMatchedUnidirGRUSubQViT(T=T_train).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2/3] Model Initialized: {param_count:,} Parameters (Target: ~520,000 Matched, T={T_train} Hops, Epochs={epochs})")

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    ckpt_path = "/models/study78_unidir_matched_checkpoint.pt"
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

    print(f"\n[3/3] Training Parameter-Matched Unidir-GRU-SubQ ViT (Epochs {start_epoch} to {epochs})...")
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
    print(f"  STUDY 78 FINAL RESULTS: EXACT PARAMETER-MATCHED UNIDIR GRU + SUBQ (T={T_train} HOPS)")
    print("=" * 115)
    print(f"  Top-1 Test Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Test Accuracy: {top5_acc:.2f}%")
    print(f"  Test Cross-Entropy:  {avg_loss:.4f}")
    print(f"  Total Training Time: {total_train_time:.1f}s ({total_train_time/epochs:.1f}s/epoch)")
    print(f"  Parameters:          {param_count:,}")
    print("=" * 115)

    res = {
        "name": f"Param-Matched Unidir-GRU SubQ ViT (T={T_train}, K=8, M={mlp_dim})",
        "params": param_count,
        "top1": top1_acc,
        "top5": top5_acc,
        "loss": avg_loss,
        "time": total_train_time
    }
    with open("/models/study78_unidir_matched_results.json", "w") as f:
        json.dump(res, f)
    volume.commit()
    return res

@app.local_entrypoint()
def main():
    train_unidir_matched_gru_subq_t8.remote()
