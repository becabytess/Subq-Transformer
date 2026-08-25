"""
Modal Experiment: 2D Vision Benchmark on CIFAR-10
Head-to-Head Comparison:
1. Standard Modern CNN (ResNet-style ConvNet)
2. Standard 4-Layer Vision Transformer (ViT-4L, Full Dense Attention)
3. 1-Layer 2D SubQ-ViT (2D Fibonacci Radial Surfer, T=3)
4. 2-Layer 2D SubQ-ViT (2D Fibonacci Radial Surfer, T=3)

Evaluates: Test Accuracy (%), Training Speed (img/sec), Peak VRAM (MB), Convergence Rate
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "torchvision", "numpy", "datasets", "tqdm")
)

app = modal.App("subq-vision-benchmark", image=image)


@app.function(gpu="T4", timeout=3600)
def run_vision_benchmark():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from datasets import load_dataset
    import numpy as np

    print("=" * 85)
    print("  2D VISION BENCHMARK: CNN vs ViT-4L vs 1L/2L 2D SUBQ-ViT ON CIFAR-10")
    print("=" * 85)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Container: {torch.cuda.get_device_name(0)}")

    # 1. Fast Dataset Loading via Hugging Face CDN (downloads in 2s)
    print("Loading CIFAR-10 from Hugging Face mirror...")
    hf_dataset = load_dataset("uoft-cs/cifar10")
    
    train_imgs = np.array(hf_dataset['train']['img'])
    train_labels = torch.tensor(hf_dataset['train']['label'], dtype=torch.long)
    test_imgs = np.array(hf_dataset['test']['img'])
    test_labels = torch.tensor(hf_dataset['test']['label'], dtype=torch.long)

    # Normalize to tensors [N, 3, 32, 32]
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

    train_x = torch.from_numpy(train_imgs).permute(0, 3, 1, 2).float() / 255.0
    train_x = (train_x - mean) / std
    test_x = torch.from_numpy(test_imgs).permute(0, 3, 1, 2).float() / 255.0
    test_x = (test_x - mean) / std

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, x, y, is_train=False):
            self.x = x
            self.y = y
            self.is_train = is_train

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            img = self.x[idx]
            if self.is_train and torch.rand(1).item() > 0.5:
                img = torch.flip(img, [2])  # Horizontal flip
            return img, self.y[idx]

    trainset = SimpleDataset(train_x, train_labels, is_train=True)
    testset = SimpleDataset(test_x, test_labels, is_train=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    epochs = 12
    d_model = 128
    d_mlp = 512
    n_heads = 4
    head_dim = d_model // n_heads
    patch_size = 2  # 32x32 image with 2x2 patches -> 16x16 grid = 256 patches!
    grid_size = 32 // patch_size  # 16
    n_patches = grid_size * grid_size  # 256
    T_hops = 3

    print(f"Image Resolution: 32x32 | Patch Size: {patch_size}x{patch_size} | Total Tokens: {n_patches} patches")
    print(f"Epochs: {epochs} | Batch Size: 128 | Optimizer: AdamW (lr=1e-3, weight_decay=1e-2)\n")

    # -------------------------------------------------------------------------
    # Build 2D Multi-Scale Fibonacci Grid Candidates
    # -------------------------------------------------------------------------
    fib_strides = [2, 3, 5, 8, 13]
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),      # Cardinal: N, S, W, E
        (-1, -1), (-1, 1), (1, -1), (1, 1)     # Diagonal: NW, NE, SW, SE
    ]

    # Pre-calculate candidate list for each patch (r, c)
    cand_map = []
    max_k = 0
    for r in range(grid_size):
        for c in range(grid_size):
            p_list = []
            # 1. Local 3x3 Fovea (9 patches)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        p_list.append(nr * grid_size + nc)
            
            # 2. Radial Fibonacci Strides in 8 directions
            for dr, dc in directions:
                for stride in fib_strides:
                    nr, nc = r + dr * stride, c + dc * stride
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        p_list.append(nr * grid_size + nc)

            p_list = sorted(list(set(p_list)))
            cand_map.append(p_list)
            if len(p_list) > max_k:
                max_k = len(p_list)

    K_2d = max_k
    cand_indices_2d = torch.zeros((n_patches, K_2d), dtype=torch.long, device=device)
    cand_mask_2d = torch.zeros((n_patches, K_2d), dtype=torch.bool, device=device)

    for i in range(n_patches):
        p_list = cand_map[i]
        for k_idx, p in enumerate(p_list):
            cand_indices_2d[i, k_idx] = p
            cand_mask_2d[i, k_idx] = True

    print(f"2D Multi-Scale Grid Config: K = {K_2d} candidates per patch (out of {n_patches} total patches)")

    # -------------------------------------------------------------------------
    # Model 1: Modern ResNet-style ConvNet (CNN)
    # -------------------------------------------------------------------------
    class ConvNetBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                # Block 1 (32x32 -> 16x16)
                nn.Conv2d(3, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.MaxPool2d(2),
                
                # Block 2 (16x16 -> 8x8)
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.MaxPool2d(2),

                # Block 3 (8x8 -> 4x4)
                nn.Conv2d(128, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.MaxPool2d(2),

                # Classifier
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 10)
            )

        def forward(self, x):
            return self.net(x)

    # -------------------------------------------------------------------------
    # Model 2: Standard 4-Layer Vision Transformer (ViT-4L)
    # -------------------------------------------------------------------------
    class ViT4LBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_mlp,
                dropout=0.0, activation='gelu', batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 10)

        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, 256, d_model]
            x = x + self.pos_embed
            x = self.transformer(x)
            x = self.norm(x.mean(dim=1))
            return self.head(x)

    # -------------------------------------------------------------------------
    # Model 3 & 4: 2D SubQ-Surfer Layer
    # -------------------------------------------------------------------------
    class SubQSurfer2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)
            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=False)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x_norm):
            B, L, D = x_norm.shape
            q = self.q_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)
            k = self.k_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)
            v = self.v_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)

            k_cand = k[:, :, cand_indices_2d, :]
            v_cand = v[:, :, cand_indices_2d, :]

            q_exp = q.unsqueeze(3)
            scores = (q_exp * k_cand).sum(dim=-1) / math.sqrt(head_dim)
            scores = scores.masked_fill(~cand_mask_2d.unsqueeze(0).unsqueeze(0), -1e9)
            pi = F.softmax(scores, dim=-1)

            ctx = (pi.unsqueeze(-1) * v_cand).sum(dim=3).transpose(1, 2).contiguous().view(B, L, D)
            gates_ctx = self.w_ih(ctx)
            r_ctx, z_ctx, n_ctx = gates_ctx.chunk(3, dim=-1)

            s = x_norm
            for _ in range(T_hops):
                gates_h = self.w_gate_h(s)
                r_h, z_h = gates_h.chunk(2, dim=-1)
                r = torch.sigmoid(r_ctx + r_h)
                z = torch.sigmoid(z_ctx + z_h)
                n = torch.tanh(n_ctx + self.w_cand_h(r * s))
                s = (1.0 - z) * n + z * s

            return self.out_proj(s)

    class SubQViT_1Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
            self.ln1 = nn.LayerNorm(d_model)
            self.surfer = SubQSurfer2D()
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp, bias=False),
                nn.GELU(),
                nn.Linear(d_mlp, d_model, bias=False)
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 10)

        def forward(self, x):
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x = x + self.pos_embed
            x = x + self.surfer(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            x = self.norm(x.mean(dim=1))
            return self.head(x)

    class SubQViT_2Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
            # Layer 1
            self.ln1_1 = nn.LayerNorm(d_model)
            self.surfer1 = SubQSurfer2D()
            self.ln2_1 = nn.LayerNorm(d_model)
            self.mlp1 = nn.Sequential(
                nn.Linear(d_model, d_mlp, bias=False),
                nn.GELU(),
                nn.Linear(d_mlp, d_model, bias=False)
            )
            # Layer 2
            self.ln1_2 = nn.LayerNorm(d_model)
            self.surfer2 = SubQSurfer2D()
            self.ln2_2 = nn.LayerNorm(d_model)
            self.mlp2 = nn.Sequential(
                nn.Linear(d_model, d_mlp, bias=False),
                nn.GELU(),
                nn.Linear(d_mlp, d_model, bias=False)
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 10)

    # -------------------------------------------------------------------------
    # Iso-Parameter Comparison:
    # 1. Standard ViT-1L scaled to ~331k params (d_mlp=896) to match SubQ-1L (331k)
    # 2. SubQ-1L scaled down to ~234k params (d_mlp=128) to match standard ViT-1L (234k)
    # -------------------------------------------------------------------------
    class ViT1L_Iso331k(nn.Module):
        def __init__(self):
            super().__init__()
            d_mlp_scaled = 896
            self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_mlp_scaled,
                dropout=0.0, activation='gelu', batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 10)

        def forward(self, x):
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x = x + self.pos_embed
            x = self.transformer(x)
            x = self.norm(x.mean(dim=1))
            return self.head(x)

    class SubQViT_1Layer_Iso234k(nn.Module):
        def __init__(self):
            super().__init__()
            d_mlp_scaled = 128
            self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
            self.ln1 = nn.LayerNorm(d_model)
            self.surfer = SubQSurfer2D()
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp_scaled, bias=False),
                nn.GELU(),
                nn.Linear(d_mlp_scaled, d_model, bias=False)
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 10)

        def forward(self, x):
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x = x + self.pos_embed
            x = x + self.surfer(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            x = self.norm(x.mean(dim=1))
            return self.head(x)

    models_to_test = [
        ("1. Standard ViT-1L (Iso-331k params)", ViT1L_Iso331k),
        ("2. SubQ-ViT-1L (Iso-234k params)", SubQViT_1Layer_Iso234k),
    ]

    benchmark_results = []

    for name, model_cls in models_to_test:
        model = model_cls().to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scaler = torch.cuda.amp.GradScaler()

        print("-" * 80)
        print(f"Training {name} ({n_params:,} parameters) for {epochs} epochs...")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        total_images_processed = 0

        for ep in range(1, epochs + 1):
            model.train()
            correct, total, total_loss = 0, 0, 0.0
            t_ep = time.time()

            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                total_images_processed += inputs.size(0)

                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                total_loss += loss.item() * targets.size(0)
                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()

            train_acc = 100.0 * correct / total
            if ep % 4 == 0 or ep == epochs:
                print(f"  Epoch {ep:2d}/{epochs} | Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.2f}% | Epoch Time: {time.time()-t_ep:.1f}s")

        total_time = time.time() - t0
        img_per_sec = total_images_processed / max(total_time, 1e-5)
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Evaluation on Test Set
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, pred = outputs.max(1)
                test_total += targets.size(0)
                test_correct += pred.eq(targets).sum().item()

        test_acc = 100.0 * test_correct / test_total
        print(f"  => TEST ACCURACY: {test_acc:.2f}% | Throughput: {img_per_sec:.0f} img/s | Peak VRAM: {peak_vram:.1f} MB\n")

        benchmark_results.append({
            "name": name,
            "params": n_params,
            "test_acc": test_acc,
            "img_per_sec": img_per_sec,
            "peak_vram_mb": peak_vram,
            "time_s": total_time
        })

    print("=" * 95)
    print("  HEAD-TO-HEAD VISION BENCHMARK RESULTS (CIFAR-10)")
    print("=" * 95)
    print(f"{'Model Architecture':<42} | {'Params':<10} | {'Test Acc':<10} | {'Throughput':<14} | {'VRAM':<10}")
    print("-" * 95)
    for r in benchmark_results:
        print(f"{r['name']:<42} | {r['params']:<10,d} | {r['test_acc']:<9.2f}% | {r['img_per_sec']:<10.0f} img/s | {r['peak_vram_mb']:<6.1f} MB")
    print("=" * 95)

    return benchmark_results


@app.local_entrypoint()
def main():
    print("Launching 2D Vision Benchmark on Modal GPU...")
    res = run_vision_benchmark.remote()
    print("\nBenchmark finished successfully!")
