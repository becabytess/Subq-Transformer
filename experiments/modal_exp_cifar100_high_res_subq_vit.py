import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "datasets>=2.18.0",
        "triton>=2.2.0",
        "numpy",
        "pillow"
    )
)

app = modal.App("exp-cifar100-high-res-subq-vit", image=image)

@app.function(gpu="A10G", timeout=7200)
def benchmark_high_res_cifar100_vit():
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
    print("  STUDY 68: HIGH-RESOLUTION CIFAR-100 SHOOTOUT (L = 257 TOKENS) — HARMONIC SUBQ ViT VS DENSE ViT")
    print("  Evaluating 1-Layer Harmonic SubQ ViT (T = 4, 8, 12 Hops) vs 1-Layer & 4-Layer Dense Vision Transformers")
    print("=" * 135)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Dataset & Transforms (L = 257 Patches)
    print("\n[1/4] Loading CIFAR-100 Dataset via Fast Cloud CDN...")
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
    patch_size = 2 # 2x2 patches -> (32/2)*(32/2) = 16x16 = 256 patches
    num_patches = (32 // patch_size) ** 2 # 256 patches
    seq_len = num_patches + 1 # 257 tokens (with [CLS])
    num_classes = 100
    K_peaks = 8
    num_waves = 12
    epochs = 20

    print(f"High-Res ViT Config: Patch Size {patch_size}x{patch_size} -> {num_patches} patches (Seq Len L = {seq_len}), Dim={d_model}, Heads={n_heads}")

    # -------------------------------------------------------------------------
    # 2. Patch Embedding Layer (2x2 Patches)
    # -------------------------------------------------------------------------
    class HighResPatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        def forward(self, x):
            B = x.shape[0]
            # [B, 3, 32, 32] -> [B, d_model, 16, 16] -> [B, d_model, 256] -> [B, 256, d_model]
            x = self.proj(x).flatten(2).transpose(1, 2)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1) # [B, 257, d_model]
            x = x + self.pos_embed
            return x

    # -------------------------------------------------------------------------
    # 3. Model 1 & 2: Dense Vision Transformer (1L & 4L)
    # -------------------------------------------------------------------------
    class DenseViTBlock(nn.Module):
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

        def forward(self, x):
            B, L, D = x.shape
            x_norm = self.ln_1(x)
            qkv = self.c_attn(x_norm).chunk(3, dim=-1)
            q, k, v = [t.view(B, L, n_heads, head_dim).transpose(1, 2) for t in qkv]

            scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores, dim=-1)
            attn_out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
            x = x + self.c_proj(attn_out)

            x = x + self.mlp(self.ln_2(x))
            return x

    class DenseViT(nn.Module):
        def __init__(self, num_layers=1):
            super().__init__()
            self.patch_embed = HighResPatchEmbed()
            self.blocks = nn.ModuleList([DenseViTBlock() for _ in range(num_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            x = self.patch_embed(x)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            cls_out = x[:, 0]
            return self.head(cls_out)

    # -------------------------------------------------------------------------
    # 4. Model 3, 4, 5: Fused High-Speed Harmonic SubQ Vision Transformer (T = 4, 8, 12)
    # -------------------------------------------------------------------------
    class OptimizedHarmonicSubQViT(nn.Module):
        def __init__(self, T=4):
            super().__init__()
            self.T = T
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

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            B = x.shape[0]
            s = self.patch_embed(x) # [B, 257, d_model]
            L = s.shape[1]

            curr_wave = self.init_wave_latent
            inv_sqrt_T = 1.0 / math.sqrt(self.T)

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

                # 4. Vectorized Gather Keys & Values for peak offsets
                q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
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
            cls_out = s_final[:, 0]
            return self.head(cls_out)

    # -------------------------------------------------------------------------
    # 5. Training & Evaluation Engine
    # -------------------------------------------------------------------------
    def train_and_eval(model, model_name):
        print(f"\n" + "=" * 110)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  TRAINING {model_name} (Parameters: {param_count:,}, Epochs: {epochs}, L = {seq_len})")
        print("=" * 110)

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

            # Test evaluation
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

            train_acc = (train_correct / train_total) * 100.0
            test_acc1 = (test_correct_top1 / test_total) * 100.0
            test_acc5 = (test_correct_top5 / test_total) * 100.0
            avg_test_loss = test_loss / test_total
            ep_time = time.time() - ep_start

            if epoch % 5 == 0 or epoch == epochs or epoch == 1:
                print(f"  Epoch {epoch:>2d}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss/train_total:.4f}, Train Acc: {train_acc:>5.2f}% | Test Loss: {avg_test_loss:.4f}, Top-1: {test_acc1:>5.2f}%, Top-5: {test_acc5:>5.2f}%")

        total_time = time.time() - t_start
        print(f"  --> Final Test Top-1: {test_acc1:.2f}% | Top-5: {test_acc5:.2f}% | Total Time: {total_time:.1f}s")
        return {
            "name": model_name,
            "params": param_count,
            "test_top1": test_acc1,
            "test_top5": test_acc5,
            "test_loss": avg_test_loss,
            "total_time": total_time
        }

    # -------------------------------------------------------------------------
    # 6. Benchmark Shootout Across Target Models
    # -------------------------------------------------------------------------
    models = [
        ("1. Standard Dense 1L-ViT (O(L^2), 1 Physical Layer)", DenseViT(num_layers=1).to(device)),
        ("2. Standard Dense 4L-ViT (O(L^2), 4 Physical Layers)", DenseViT(num_layers=4).to(device)),
        ("3. Harmonic SubQ ViT (1L, T=4 Hops, Evolving QKV)", OptimizedHarmonicSubQViT(T=4).to(device)),
        ("4. Harmonic SubQ ViT (1L, T=8 Hops, Dynamic Waves)", OptimizedHarmonicSubQViT(T=8).to(device)),
        ("5. Harmonic SubQ ViT (1L, T=12 Hops, Deep Thought)", OptimizedHarmonicSubQViT(T=12).to(device)),
    ]

    all_results = []
    for name, model in models:
        torch.manual_seed(42)
        res = train_and_eval(model, name)
        all_results.append(res)

    # -------------------------------------------------------------------------
    # 7. Final Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 145)
    print("  STUDY 68 HIGH-RES CIFAR-100 (L = 257 TOKENS) BENCHMARK SUMMARY: HARMONIC SUBQ ViT VS DENSE ViT")
    print("=" * 145)
    print(f"{'Model Architecture':<54} | {'Complexity':<12} | {'Parameters':<14} | {'Top-1 Acc':<12} | {'Top-5 Acc':<12} | {'Test Loss':<12} | {'Train Time':<10}")
    print("-" * 145)
    for r in all_results:
        print(f"{r['name']:<54} | {'O(L*K)' if 'SubQ' in r['name'] else 'O(L^2)':<12} | {r['params']:<14,d} | {r['test_top1']:<10.2f}% | {r['test_top5']:<10.2f}% | {r['test_loss']:<12.4f} | {r['total_time']:>6.1f}s")
    print("=" * 145)

    return all_results

@app.local_entrypoint()
def main():
    benchmark_high_res_cifar100_vit.remote()
