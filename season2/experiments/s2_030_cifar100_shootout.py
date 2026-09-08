"""S2-030: CIFAR-100 High-Res Vision Shootout (L=257 tokens).

Compares:
1. dense_4layer: 4 physical dense layers with all-to-all bidirectional attention (~830k params).
2. subq_2macro_recurrent: 1 parameter-tied block, 2 macro-iterations of [4 linear hops -> MLP] (~220k params).
3. subq_1layer_final_mlp: 1 parameter-tied block, 8 linear hops -> 1 final MLP (~220k params).
All SubQ models use the complete K=9 offset basis [0, 1, 2, 4, 8, 16, 63, 127, 128].
"""

import json
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "torchvision>=0.17.0", "datasets>=2.18.0", "numpy", "pillow"
)
app = modal.App("season2-s2-030-cifar100-shootout")


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
    offsets = [0, 1, 2, 4, 8, 16, 63, 127, 128]
    batch_size = 128
    seq_len = 257  # 256 patches (2x2, stride 2 on 32x32) + 1 CLS token
    num_classes = 100

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

    # 2. SubQ 2-Macro Recurrent ViT
    class SubQ2MacroViT(nn.Module):
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
            targets = torch.zeros((seq_len, len(offsets)), dtype=torch.long)
            for pos in range(seq_len):
                for j, offset in enumerate(offsets):
                    targets[pos, j] = (pos - offset) % seq_len
            self.register_buffer("targets", targets)

        def forward(self, x):
            bsz = x.shape[0]
            state = self.patch(x).flatten(2).transpose(1, 2)
            state = torch.cat([self.cls.expand(bsz, -1, -1), state], dim=1) + self.pos
            scale = 1.0 / math.sqrt(d_model // n_heads)
            for _ in range(2):  # 2 macro iterations
                for _ in range(4):  # 4 linear transport hops
                    z = self.ln_attn(state)
                    q = self.q(z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                    k = self.k(z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                    v = self.v(z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                    scores = (q.unsqueeze(3) * k[:, :, self.targets, :]).sum(-1) * scale
                    weights = F.softmax(scores, dim=-1)
                    context = (weights.unsqueeze(-1) * v[:, :, self.targets, :]).sum(3)
                    state = context.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
                state = state + self.mlp(self.ln_mlp(state))
            return self.head(self.ln_f(state))[:, 0]

    # 3. SubQ 1-Layer Final MLP ViT
    class SubQ1LayerViT(nn.Module):
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
            targets = torch.zeros((seq_len, len(offsets)), dtype=torch.long)
            for pos in range(seq_len):
                for j, offset in enumerate(offsets):
                    targets[pos, j] = (pos - offset) % seq_len
            self.register_buffer("targets", targets)

        def forward(self, x):
            bsz = x.shape[0]
            state = self.patch(x).flatten(2).transpose(1, 2)
            state = torch.cat([self.cls.expand(bsz, -1, -1), state], dim=1) + self.pos
            scale = 1.0 / math.sqrt(d_model // n_heads)
            for _ in range(8):  # 8 continuous linear transport hops
                z = self.ln_attn(state)
                q = self.q(z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                k = self.k(z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                v = self.v(z).view(bsz, seq_len, n_heads, -1).transpose(1, 2)
                scores = (q.unsqueeze(3) * k[:, :, self.targets, :]).sum(-1) * scale
                weights = F.softmax(scores, dim=-1)
                context = (weights.unsqueeze(-1) * v[:, :, self.targets, :]).sum(3)
                state = context.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
            state = state + self.mlp(self.ln_mlp(state))
            return self.head(self.ln_f(state))[:, 0]

    if architecture == "dense_4layer":
        model = Dense4LayerViT().to(device)
    elif architecture == "subq_2macro_recurrent":
        model = SubQ2MacroViT().to(device)
    elif architecture == "subq_1layer_final_mlp":
        model = SubQ1LayerViT().to(device)
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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
            correct += logits.argmax(-1).eq(labels).sum().item()
            total += labels.numel()

    return {
        "model": model_name,
        "architecture": architecture,
        "seed": seed,
        "parameters": parameters,
        "epochs": epochs,
        "top1_accuracy": round(100.0 * correct / total, 4),
        "total_test_images": total,
        "elapsed_s": round(time.time() - start, 3),
    }


@app.local_entrypoint()
def main():
    import pathlib
    import time

    seeds = [42, 1337, 2026]
    architectures = [
        "dense_4layer",
        "subq_2macro_recurrent",
        "subq_1layer_final_mlp",
    ]
    configs = [
        (f"{arch}_s{seed}", arch, seed)
        for arch in architectures
        for seed in seeds
    ]

    print(f"Launching {len(configs)} CIFAR-100 L=257 jobs on Modal in parallel...", flush=True)
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
        "experiment": "S2-030",
        "description": "CIFAR-100 High-Res Vision Shootout: Dense 4-layer vs SubQ 2-macro recurrent vs SubQ 1-layer final MLP (L=257 tokens)",
        "task": "CIFAR-100 classification with patch size 2x2 stride 2 (257 tokens)",
        "seeds": seeds,
        "epochs": 10,
        "conditions": configs,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    path = pathlib.Path("season2/results/s2_030_cifar100_shootout.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {path}", flush=True)


if __name__ == "__main__":
    main()
