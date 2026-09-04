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

app = modal.App("study81-scaled-dense-vit", image=image)
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)

@app.function(gpu="A10G", timeout=3600, volumes={"/models": volume})
def train_scaled_dense_vit():
    import math, time, json, torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 81: SCALED 1-LAYER DENSE ViT (742k PARAMETERS)")
    print("  High-Res CIFAR-100 (L = 257 Patches, Dense All-to-All Softmax Attention)")
    print("  Scaled Parameter Parity with Study 77 GRU-SubQ: MLP dim = 1356 -> Total Params ~ 742,960")
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
    mlp_dim = 1356 # Scaled to match ~742k total parameters
    patch_size = 2 # 2x2 patches -> 16x16 = 256 patches
    num_patches = (32 // patch_size) ** 2 # 256
    seq_len = num_patches + 1 # 257 tokens (with [CLS])
    num_classes = 100
    epochs = 20

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
    # 3. Scaled 1-Layer Dense ViT Block
    # -------------------------------------------------------------------------
    class ScaledDenseViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = HighResPatchEmbed()
            self.ln_1 = nn.LayerNorm(d_model)
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)

            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, d_model)
            )

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x)
            L, D = x.shape[1], x.shape[2]

            # Dense All-to-All Attention
            x_norm = self.ln_1(x)
            qkv = self.c_attn(x_norm).chunk(3, dim=-1)
            q, k, v = [t.view(B, L, n_heads, head_dim).transpose(1, 2) for t in qkv]

            scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores, dim=-1)
            attn_out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
            x = x + self.c_proj(attn_out)

            # Scaled MLP
            x = x + self.mlp(self.ln_2(x))

            x = self.ln_f(x)
            return self.head(x[:, 0])

    model = ScaledDenseViT().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2/3] Model Initialized: {param_count:,} Parameters (Scaled Dense ViT, Target: ~742k, Epochs={epochs})")

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    ckpt_path = "/models/study81_scaled_dense_checkpoint.pt"
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

    print(f"\n[3/3] Training Scaled Dense ViT (Epochs {start_epoch} to {epochs})...")
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
    print(f"  STUDY 81 FINAL RESULTS: SCALED 1L DENSE ViT (742k PARAMS)")
    print("=" * 115)
    print(f"  Top-1 Test Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Test Accuracy: {top5_acc:.2f}%")
    print(f"  Test Cross-Entropy:  {avg_loss:.4f}")
    print(f"  Total Training Time: {total_train_time:.1f}s ({total_train_time/epochs:.1f}s/epoch)")
    print(f"  Parameters:          {param_count:,}")
    print("=" * 115)

    res = {
        "name": f"Scaled 1L Dense ViT (M={mlp_dim})",
        "params": param_count,
        "top1": top1_acc,
        "top5": top5_acc,
        "loss": avg_loss,
        "time": total_train_time
    }
    with open("/models/study81_scaled_dense_results.json", "w") as f:
        json.dump(res, f)
    volume.commit()
    return res

@app.local_entrypoint()
def main():
    train_scaled_dense_vit.remote()
