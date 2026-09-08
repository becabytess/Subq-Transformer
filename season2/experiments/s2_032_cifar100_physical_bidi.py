"""S2-032: CIFAR-100 Vision Shootout with Physical Depth and Bidirectional 2D Offsets (L=257).

Compares:
1. dense_4layer: 4 physical dense layers with all-to-all bidirectional attention + MLP (838,784 params).
2. subq_4layer_physical_bidi: 4 physical SubQ layers, each 2 linear hops -> MLP (773,248 params, 8 total hops, 4 MLPs).
3. subq_1layer_final_mlp_bidi: 1 physical SubQ layer, 8 linear hops -> 1 final MLP (229,120 params).

Both SubQ models use symmetric bidirectional 2D spatial grid offsets (H: +-1, +-2, +-4, +-8; V: +-1, +-2, +-4, +-8; self; CLS hub; K=18).
SubQ has no parameter advantage over Dense 4-layer (773k vs 838k).
"""

import json
import pathlib
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "torchvision>=0.17.0", "datasets>=2.18.0", "numpy", "pillow"
)
app = modal.App("season2-s2-032-cifar100-physical-bidi")


def build_2d_grid_targets(seq_len=257, K=18):
    import torch
    targets = torch.zeros((seq_len, K), dtype=torch.long)
    valid = torch.zeros((seq_len, K), dtype=torch.bool)

    # Token 0: CLS hub
    targets[0, 0] = 0
    valid[0, 0] = True
    anchors = []
    for r in [1, 5, 9, 13]:
        for c in [1, 5, 9, 13]:
            anchors.append(1 + r * 16 + c)
    for i, a in enumerate(anchors):
        targets[0, 1 + i] = a
        valid[0, 1 + i] = True
    targets[0, 17] = 1 + 7 * 16 + 7
    valid[0, 17] = True

    # Patches 1..256
    h_offsets = [-8, -4, -2, -1, 1, 2, 4, 8]
    v_offsets = [-8, -4, -2, -1, 1, 2, 4, 8]

    for pos in range(1, seq_len):
        p = pos - 1
        r = p // 16
        c = p % 16

        # Slot 0: Self
        targets[pos, 0] = pos
        valid[pos, 0] = True

        # Slot 1: CLS
        targets[pos, 1] = 0
        valid[pos, 1] = True

        # Slots 2..9: Horizontal
        for idx, dc in enumerate(h_offsets):
            slot = 2 + idx
            c_new = c + dc
            if 0 <= c_new < 16:
                targets[pos, slot] = 1 + r * 16 + c_new
                valid[pos, slot] = True
            else:
                targets[pos, slot] = 0
                valid[pos, slot] = False

        # Slots 10..17: Vertical
        for idx, dr in enumerate(v_offsets):
            slot = 10 + idx
            r_new = r + dr
            if 0 <= r_new < 16:
                targets[pos, slot] = 1 + r_new * 16 + c
                valid[pos, slot] = True
            else:
                targets[pos, slot] = 0
                valid[pos, slot] = False

    return targets, valid


@app.function(image=image, gpu="A10G", timeout=3600)
def train_one(
    model_name: str,
    architecture: str,
    seed: int,
    epochs: int = 10,
):
    import math
    import time
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

    d_model, n_heads, d_mlp = 128, 4, 512
    batch_size = 128
    seq_len = 257  # 256 patches (2x2, stride 2 on 32x32) + 1 CLS token
    num_classes = 100
    K = 18

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

    targets_2d, valid_2d = build_2d_grid_targets(seq_len=seq_len, K=K)

    # 1. Dense 4-Layer ViT
    class Dense4LayerViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "ln_attn": nn.LayerNorm(d_model),
                    "q": nn.Linear(d_model, d_model, bias=False),
                    "k": nn.Linear(d_model, d_model, bias=False),
                    "v": nn.Linear(d_model, d_model, bias=False),
                    "o": nn.Linear(d_model, d_model, bias=False),
                    "ln_mlp": nn.LayerNorm(d_model),
                    "mlp": nn.Sequential(
                        nn.Linear(d_model, d_mlp),
                        nn.GELU(),
                        nn.Linear(d_mlp, d_model),
                    ),
                }) for _ in range(4)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes, bias=False)

        def forward(self, x):
            bsz = x.shape[0]
            state = self.patch(x).flatten(2).transpose(1, 2)
            state = torch.cat([self.cls.expand(bsz, -1, -1), state], dim=1) + self.pos
            scale = 1.0 / math.sqrt(d_model // n_heads)
            for layer in self.layers:
                z = layer["ln_attn"](state)
                q = layer["q"](z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                k = layer["k"](z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                v = layer["v"](z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                scores = (q @ k.transpose(-2, -1)) * scale
                weights = F.softmax(scores, dim=-1)
                msg = layer["o"]((weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, d_model))
                state = state + msg
                state = state + layer["mlp"](layer["ln_mlp"](state))
            return self.head(self.ln_f(state))[:, 0]

    # 2. SubQ 4-Layer Physical ViT (4 physical layers, each [2 hops -> MLP], 2D grid offsets)
    class SubQ4LayerPhysicalViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "ln_attn": nn.LayerNorm(d_model),
                    "q": nn.Linear(d_model, d_model, bias=False),
                    "k": nn.Linear(d_model, d_model, bias=False),
                    "v": nn.Linear(d_model, d_model, bias=False),
                    "ln_mlp": nn.LayerNorm(d_model),
                    "mlp": nn.Sequential(
                        nn.Linear(d_model, d_mlp),
                        nn.GELU(),
                        nn.Linear(d_mlp, d_model),
                    ),
                }) for _ in range(4)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes, bias=False)
            self.register_buffer("targets", targets_2d)
            self.register_buffer("valid", valid_2d)

        def forward(self, x):
            bsz = x.shape[0]
            state = self.patch(x).flatten(2).transpose(1, 2)
            state = torch.cat([self.cls.expand(bsz, -1, -1), state], dim=1) + self.pos
            scale = 1.0 / math.sqrt(d_model // n_heads)
            valid = self.valid.view(1, 1, seq_len, K)
            for layer in self.layers:
                for _ in range(8):  # 8 linear hops per layer (full reachability in every layer)
                    z = layer["ln_attn"](state)
                    q = layer["q"](z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                    k = layer["k"](z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                    v = layer["v"](z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                    scores = (q.unsqueeze(3) * k[:, :, self.targets, :]).sum(-1) * scale
                    weights = F.softmax(scores.masked_fill(~valid, -1e4), dim=-1)
                    context = (weights.unsqueeze(-1) * v[:, :, self.targets, :]).sum(3)
                    state = context.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
                state = state + layer["mlp"](layer["ln_mlp"](state))
            return self.head(self.ln_f(state))[:, 0]

    # 3. SubQ 1-Layer Final MLP ViT (8 linear hops -> 1 final MLP, 2D grid offsets)
    class SubQ1LayerFinalMLPViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            self.ln_attn = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model),
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes, bias=False)
            self.register_buffer("targets", targets_2d)
            self.register_buffer("valid", valid_2d)

        def forward(self, x):
            bsz = x.shape[0]
            state = self.patch(x).flatten(2).transpose(1, 2)
            state = torch.cat([self.cls.expand(bsz, -1, -1), state], dim=1) + self.pos
            scale = 1.0 / math.sqrt(d_model // n_heads)
            valid = self.valid.view(1, 1, seq_len, K)
            for _ in range(8):  # 8 continuous linear hops
                z = self.ln_attn(state)
                q = self.q(z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                k = self.k(z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                v = self.v(z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                scores = (q.unsqueeze(3) * k[:, :, self.targets, :]).sum(-1) * scale
                weights = F.softmax(scores.masked_fill(~valid, -1e4), dim=-1)
                context = (weights.unsqueeze(-1) * v[:, :, self.targets, :]).sum(3)
                state = context.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
            state = state + self.mlp(self.ln_mlp(state))
            return self.head(self.ln_f(state))[:, 0]

    if architecture == "dense_4layer":
        model = Dense4LayerViT().to(device)
    elif architecture == "subq_4layer_physical_bidi":
        model = SubQ4LayerPhysicalViT().to(device)
    elif architecture == "subq_1layer_final_mlp_bidi":
        model = SubQ1LayerFinalMLPViT().to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader), eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start = time.time()

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                preds = model(images).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = round(100.0 * correct / total, 2)
    elapsed = round(time.time() - start, 3)

    return {
        "model": model_name,
        "architecture": architecture,
        "seed": seed,
        "parameters": parameters,
        "epochs": epochs,
        "top1_accuracy": accuracy,
        "total_test_images": total,
        "elapsed_s": elapsed,
    }


@app.local_entrypoint()
def main():
    seeds = [42, 1337, 2026]
    architectures = [
        "dense_4layer",
        "subq_4layer_physical_bidi",
        "subq_1layer_final_mlp_bidi",
    ]
    configs = [
        (f"{arch}_s{seed}", arch, seed)
        for arch in architectures
        for seed in seeds
    ]

    print(f"Launching {len(configs)} S2-032 CIFAR-100 Vision jobs on Modal in parallel...", flush=True)
    calls = [
        train_one.spawn(name, arch, seed)
        for name, arch, seed in configs
    ]

    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config[0]}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "experiment": "S2-032",
        "description": "CIFAR-100 High-Res Vision Shootout: Dense 4-layer vs SubQ 4-layer physical vs SubQ 1-layer final MLP with 2D bidirectional spatial offsets (L=257 tokens)",
        "task": "CIFAR-100 classification with patch size 2x2 stride 2 (257 tokens)",
        "seeds": seeds,
        "epochs": 10,
        "conditions": configs,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    path = pathlib.Path("season2/results/s2_032_cifar100_physical_bidi.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {path}", flush=True)


if __name__ == "__main__":
    main()
