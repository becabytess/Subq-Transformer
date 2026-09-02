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

app = modal.App("exp-subq-vit-harmonic-filterbank", image=image)

@app.function(gpu="A10G", timeout=7200)
def run_harmonic_filterbank_experiment():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 135)
    print("  STUDY 71: CONTINUOUS HARMONIC FILTERBANK (HARMONIC WAVELET POOLING) SUBQ ViT")
    print("  Evaluating Harmonic Super-Token Neighborhood Pooling on High-Res CIFAR-100 (L = 257 Patches, K = 8 Super-Tokens, T = 4 Hops)")
    print("=" * 135)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Dataset & Transforms (L = 257 Patches)
    print("\n[1/3] Loading CIFAR-100 Dataset via Fast Cloud CDN...")
    t_d0 = time.time()
    hf_dataset = load_dataset("uoft-cs/cifar100")
    print(f"--> Loaded CIFAR-100 in {time.time()-t_d0:.2f}s! Train: {len(hf_dataset['train'])}, Test: {len(hf_dataset['test'])}")

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
            if self.transform:
                img = self.transform(img)
            return img, label

    train_dataset = Cifar100Wrapper(hf_dataset['train'], transform=transform_train)
    test_dataset = Cifar100Wrapper(hf_dataset['test'], transform=transform_test)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    d_model = 192
    n_heads = 6
    head_dim = d_model // n_heads # 32
    patch_size = 2
    num_patches = (32 // patch_size) ** 2 # 256 patches
    seq_len = num_patches + 1 # 257 tokens (with [CLS])
    num_classes = 100
    num_waves = 12
    epochs = 20
    T_train = 4
    K_super_tokens = 8

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
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            return x

    # -------------------------------------------------------------------------
    # 3. Continuous Harmonic Filterbank (Wavelet Super-Token Pooling) ViT
    # -------------------------------------------------------------------------
    class HarmonicFilterbankSubQViT(nn.Module):
        def __init__(self, filter_radius=2):
            """
            filter_radius: Neighborhood radius delta around each peak center.
            Window size W = 2 * filter_radius + 1 (e.g. radius=2 -> 5-token window: [-2, -1, 0, 1, 2]).
            """
            super().__init__()
            self.filter_radius = filter_radius
            self.window_size = 2 * filter_radius + 1
            self.patch_embed = HighResPatchEmbed()

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

            # Fixed / Learnable Triangular/Gaussian Window Prior
            window_offsets = torch.arange(-filter_radius, filter_radius + 1).float()
            # Triangular weighting prior: 1 - |delta| / (radius + 1)
            tri_weights = 1.0 - torch.abs(window_offsets) / (filter_radius + 1.0)
            self.register_buffer("window_offsets", window_offsets.long().view(1, 1, 1, self.window_size))
            self.register_buffer("tri_prior", tri_weights.view(1, 1, 1, self.window_size))

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            B = x.shape[0]
            s = self.patch_embed(x)
            L = s.shape[1]
            device = x.device

            curr_wave = self.init_wave_latent
            step_mult = 1.0 / math.sqrt(T_train)
            D_len = self.max_d - 1 # 256

            for t in range(T_train):
                s_norm = self.ln_1(s)

                # 1. Full Evolving Q, K, V
                qkv = self.c_attn(s_norm).chunk(3, dim=-1)
                q, k, v = [t_tensor.view(B, L, n_heads, head_dim).transpose(1, 2) for t_tensor in qkv]
                # q, k, v: [B, n_heads, L, head_dim]

                # 2. Continuous Spatial Wave Carrier
                params = curr_wave.view(n_heads, num_waves, 4)
                amp = torch.tanh(params[..., 0]).view(1, n_heads, 1, num_waves)
                omega = (F.softplus(params[..., 1]).view(1, n_heads, 1, num_waves) * self.base_freqs)
                phi = (params[..., 2] * math.pi).view(1, n_heads, 1, num_waves)
                decay = (F.softplus(params[..., 3]) * 0.05).view(1, n_heads, 1, num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1) # [B, n_heads, 256]

                # 3. Peak Finding via 1D Non-Maximum Suppression (NMS)
                pad = self.filter_radius
                padded = F.pad(wave_1d, (pad, pad), mode='replicate')
                pooled = F.max_pool1d(padded, kernel_size=self.window_size, stride=1)
                is_max = (wave_1d == pooled)
                suppressed = torch.where(is_max, wave_1d, torch.tensor(-1e4, device=device))

                # Top (K_super_tokens - 1) distinct macro peaks
                top_vals, top_indices = torch.topk(suppressed, k=K_super_tokens - 1, dim=-1)
                peak_centers = top_indices + 1 # [B, n_heads, 7]

                # Prepend self-token center (d=0)
                zero_center = torch.zeros((B, n_heads, 1), dtype=torch.long, device=device)
                zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=device)
                all_centers = torch.cat([zero_center, peak_centers], dim=-1) # [B, n_heads, 8]
                all_peak_vals = torch.cat([zero_val, top_vals], dim=-1)      # [B, n_heads, 8]

                # 4. Continuous Harmonic Filterbank Super-Token Neighborhood Pooling
                # Expand window offsets around each peak center: [B, n_heads, 8, W]
                # delta in [-radius, ..., +radius]
                neighbor_offsets = (all_centers.unsqueeze(-1) + self.window_offsets) % L # [B, n_heads, 8, W]

                # Gather wave amplitudes at each neighbor to compute filterbank weights
                # Clamp for 1D wave index [0..255]
                wave_lookup_idx = (neighbor_offsets - 1).clamp(0, D_len - 1)
                # For self-token (center 0), use triangular center prior
                local_wave_energy = torch.gather(wave_1d.unsqueeze(2).expand(B, n_heads, K_super_tokens, D_len), dim=-1, index=wave_lookup_idx)
                # Combine wave amplitude with triangular bandpass window prior
                filter_logits = local_wave_energy * 0.5 + torch.log(self.tri_prior.clamp(min=1e-4))
                filter_weights = F.softmax(filter_logits, dim=-1) # [B, n_heads, 8, W]

                # 5. Vectorized Gather Keys & Values across Window and Compute Super-Tokens
                q_pos = torch.arange(L, device=device).view(1, 1, 1, L, 1, 1) # [1, 1, 1, L, 1, 1]
                # Target token index in sequence: (q - offset) % L
                # neighbor_offsets: [B, n_heads, 1, 1, 8, W]
                target_tokens = (q_pos - neighbor_offsets.unsqueeze(2).unsqueeze(3)) % L # [B, n_heads, 1, L, 8, W]
                target_tokens = target_tokens.squeeze(2) # [B, n_heads, L, 8, W]

                # Expand K, V: [B, n_heads, L, 8, W, head_dim]
                idx_exp = target_tokens.unsqueeze(-1).expand(B, n_heads, L, K_super_tokens, self.window_size, head_dim)
                k_exp = k.unsqueeze(3).unsqueeze(4).expand(B, n_heads, L, K_super_tokens, self.window_size, head_dim)
                v_exp = v.unsqueeze(3).unsqueeze(4).expand(B, n_heads, L, K_super_tokens, self.window_size, head_dim)

                K_neighbors = torch.gather(k_exp, dim=2, index=idx_exp) # [B, n_heads, L, 8, W, head_dim]
                V_neighbors = torch.gather(v_exp, dim=2, index=idx_exp) # [B, n_heads, L, 8, W, head_dim]

                # Weighted Super-Token Pooling: sum_W (w_p * K_neighbor)
                w_exp = filter_weights.unsqueeze(2).unsqueeze(-1) # [B, n_heads, 1, 8, W, 1]
                K_super = (w_exp * K_neighbors).sum(dim=4) # [B, n_heads, L, 8, head_dim]
                V_super = (w_exp * V_neighbors).sum(dim=4) # [B, n_heads, L, 8, head_dim]

                # 6. Attention against 8 Super-Tokens with Harmonic Bias
                scores = (q.unsqueeze(3) * K_super).sum(dim=-1) / math.sqrt(head_dim) + all_peak_vals.unsqueeze(2)
                attn = F.softmax(scores, dim=-1) # [B, n_heads, L, 8]
                attn_out = (attn.unsqueeze(-1) * V_super).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
                attn_out = self.c_proj(attn_out)

                # 7. Recurrent Residual Update
                s = s + step_mult * attn_out
                s = s + step_mult * self.mlp(self.ln_2(s))

                # 8. Wave Transition
                if t < T_train - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            s_final = self.ln_f(s)
            cls_out = s_final[:, 0]
            logits = self.head(cls_out)
            return logits

    # -------------------------------------------------------------------------
    # 4. Train & Evaluate Harmonic Filterbank ViT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"  TRAINING CONTINUOUS HARMONIC FILTERBANK ViT (Radius = 2 -> 5-Patch Window per Super-Token)")
    print("=" * 115)
    model = HarmonicFilterbankSubQViT(filter_radius=2).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,} | Super-Tokens: {K_super_tokens} | Window: 5 tokens/super-token | Thought Hops: {T_train} | Epochs: {epochs}")

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    t_start = time.time()
    for epoch in range(1, epochs + 1):
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

        train_acc = (train_correct / train_total) * 100.0
        ep_time = time.time() - ep_start
        if epoch % 2 == 0 or epoch == epochs or epoch == 1:
            print(f"  Epoch {epoch:>2d}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss/train_total:.4f}, Train Acc: {train_acc:>5.2f}%")

    total_train_time = time.time() - t_start

    # Evaluate on Test Set
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

    print("\n" + "=" * 125)
    print("  STUDY 71 FINAL RESULTS: CONTINUOUS HARMONIC FILTERBANK (HARMONIC WAVELET POOLING) SUBQ ViT")
    print("=" * 125)
    print(f"  Top-1 Test Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Test Accuracy: {top5_acc:.2f}%")
    print(f"  Test Cross-Entropy:  {avg_loss:.4f}")
    print(f"  Total Training Time: {total_train_time:.1f}s ({total_train_time/epochs:.1f}s/epoch)")
    print(f"  Physical Parameters: {param_count:,}")
    print("=" * 125)

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "loss": avg_loss,
        "time": total_train_time
    }

@app.local_entrypoint()
def main():
    run_harmonic_filterbank_experiment.remote()
