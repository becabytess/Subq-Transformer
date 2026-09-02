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

app = modal.App("study73-2-2layer-interleaved", image=image)
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)

@app.function(gpu="A10G", timeout=3600, volumes={"/models": volume})
def train_2layer_interleaved():
    import math, time, json, torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 73 - VARIANT 2: 2-LAYER INTERLEAVED HARMONIC SUBQ ViT (Deep Thought Block unrolled T=4 Hops)")
    print("  High-Res CIFAR-100 (L = 257 Patches, K = 8 Peak Offsets, 20 Epochs, Per-Epoch Auto-Resume Checkpointing)")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Dataset
    print("\n[1/3] Loading CIFAR-100 Dataset via Fast Cloud CDN...")
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
    patch_size = 2
    num_patches = (32 // patch_size) ** 2 # 256
    seq_len = num_patches + 1 # 257
    num_classes = 100
    num_waves = 12
    epochs = 20
    T_train = 4
    K_points = 8

    # -------------------------------------------------------------------------
    # 2. Patch Embedding Layer
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
    # 3. Single Sub-Step Layer
    # -------------------------------------------------------------------------
    class SubQStepLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1 = nn.LayerNorm(d_model)
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
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

        def step(self, s, curr_wave, step_mult):
            B, L, _ = s.shape
            device = s.device

            s_norm = self.ln_1(s)
            qkv = self.c_attn(s_norm).chunk(3, dim=-1)
            q, k, v = [t_tensor.view(B, L, n_heads, head_dim).transpose(1, 2) for t_tensor in qkv]

            params = curr_wave.view(n_heads, num_waves, 4)
            amp = torch.tanh(params[..., 0]).view(1, n_heads, 1, num_waves)
            omega = (F.softplus(params[..., 1]).view(1, n_heads, 1, num_waves) * self.base_freqs)
            phi = (params[..., 2] * math.pi).view(1, n_heads, 1, num_waves)
            decay = (F.softplus(params[..., 3]) * 0.05).view(1, n_heads, 1, num_waves)

            wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
            wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

            topk_vals, past_peak_offsets = torch.topk(wave_1d, k=K_points - 1, dim=-1)
            past_peak_offsets = past_peak_offsets + 1
            zero_offset = torch.zeros((B, n_heads, 1), dtype=torch.long, device=device)
            zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=device)
            peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
            peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

            q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
            target_indices = (q_pos - peak_offsets.unsqueeze(2)) % L
            idx_exp = target_indices.unsqueeze(-1).expand(B, n_heads, L, K_points, head_dim)

            K_g = torch.gather(k.unsqueeze(3).expand(B, n_heads, L, K_points, head_dim), dim=2, index=idx_exp)
            V_g = torch.gather(v.unsqueeze(3).expand(B, n_heads, L, K_points, head_dim), dim=2, index=idx_exp)

            scores = (q.unsqueeze(3) * K_g).sum(dim=-1) / math.sqrt(head_dim) + peak_vals.unsqueeze(2)
            attn = F.softmax(scores, dim=-1)
            attn_out = (attn.unsqueeze(-1) * V_g).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
            attn_out = self.c_proj(attn_out)

            s = s + step_mult * attn_out
            s = s + step_mult * self.mlp(self.ln_2(s))
            next_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)
            return s, next_wave

    class TwoLayerInterleavedSubQViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = HighResPatchEmbed()
            self.layer1 = SubQStepLayer()
            self.layer2 = SubQStepLayer()
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            s = self.patch_embed(x)
            step_mult = 1.0 / math.sqrt(T_train)
            curr_wave1 = self.layer1.init_wave_latent
            curr_wave2 = self.layer2.init_wave_latent

            for t in range(T_train):
                s, curr_wave1 = self.layer1.step(s, curr_wave1, step_mult)
                s, curr_wave2 = self.layer2.step(s, curr_wave2, step_mult)

            s_final = self.ln_f(s)
            return self.head(s_final[:, 0])

    model = TwoLayerInterleavedSubQViT().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2/3] Model Initialized: {param_count:,} Parameters (2 Interleaved Layers, T={T_train} Hops, Epochs={epochs})")

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    ckpt_path = "/models/study73_2_interleaved_checkpoint.pt"
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
        print(f"--> Resumed successfully from Epoch {start_epoch} (Previous Time: {total_train_time:.1f}s)!")

    print(f"\n[3/3] Training 2-Layer Interleaved Harmonic SubQ ViT (Epochs {start_epoch} to {epochs})...")
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

        # Save Checkpoint Every Single Epoch & Commit Volume
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
    print("  STUDY 73 VARIANT 2 FINAL RESULTS: 2-LAYER INTERLEAVED HARMONIC SUBQ ViT")
    print("=" * 115)
    print(f"  Top-1 Test Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Test Accuracy: {top5_acc:.2f}%")
    print(f"  Test Cross-Entropy:  {avg_loss:.4f}")
    print(f"  Total Training Time: {total_train_time:.1f}s ({total_train_time/epochs:.1f}s/epoch)")
    print(f"  Parameters:          {param_count:,}")
    print("=" * 115)

    res = {
        "name": "2-Layer Interleaved Harmonic SubQ ViT (T=4)",
        "params": param_count,
        "top1": top1_acc,
        "top5": top5_acc,
        "loss": avg_loss,
        "time": total_train_time
    }
    with open("/models/study73_2_results.json", "w") as f:
        json.dump(res, f)
    volume.commit()
    return res

@app.local_entrypoint()
def main():
    train_2layer_interleaved.remote()
