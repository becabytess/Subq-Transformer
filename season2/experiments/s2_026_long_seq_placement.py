"""S2-026: long-sequence MLP placement on language L=512 and vision L=257."""

import json
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "torchvision>=0.17.0", "datasets>=2.18.0", "numpy", "pillow"
)
app = modal.App("season2-s2-026-long-seq-placement")


@app.function(image=image, gpu="A10G", timeout=7200)
def train_one(domain: str, mode: str, seed: int, steps: int = 2000, epochs: int = 10):
    import math
    import time
    import urllib.request

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_model, n_heads, d_mlp, thought_hops = 128, 4, 512, 8
    offsets = [0, 1, 2, 4, 8, 16, 32, 64]
    batch_size = 32 if domain == "language" else 128
    causal = domain == "language"

    if domain == "language":
        text = (
            urllib.request.urlopen(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            )
            .read()
            .decode()
        )
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
        split = int(0.9 * len(data))
        train_data, val_data = data[:split], data[split:]
        seq_len, vocab_size = 512, len(chars)

        def get_batch(which, generator):
            source = train_data if which == "train" else val_data
            starts = torch.randint(
                len(source) - seq_len - 1, (batch_size,), generator=generator
            )
            x = torch.stack([source[int(i) : int(i) + seq_len] for i in starts]).to(
                device
            )
            y = torch.stack(
                [source[int(i) + 1 : int(i) + seq_len + 1] for i in starts]
            ).to(device)
            return x, y
    else:
        from datasets import load_dataset
        from torch.utils.data import DataLoader, Dataset
        import torchvision.transforms as transforms

        raw = load_dataset("uoft-cs/cifar100")
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

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
        seq_len, vocab_size = 257, 100

    class PlacementModel(nn.Module):
        def __init__(self):
            super().__init__()
            if domain == "language":
                self.tok = nn.Embedding(vocab_size, d_model)
                self.pos = nn.Embedding(seq_len, d_model)
            else:
                self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
                self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
                self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            self.ln_attn = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.o = nn.Linear(d_model, d_model, bias=False)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            targets = torch.zeros((seq_len, len(offsets)), dtype=torch.long)
            valid = torch.zeros((seq_len, len(offsets)), dtype=torch.bool)
            for pos in range(seq_len):
                for j, offset in enumerate(offsets):
                    source = (pos - offset) if causal else ((pos - offset) % seq_len)
                    if causal and source < 0:
                        continue
                    targets[pos, j], valid[pos, j] = source, True
            self.register_buffer("targets", targets)
            self.register_buffer("valid", valid)

        def forward(self, inputs):
            if domain == "language":
                state = self.tok(inputs) + self.pos(
                    torch.arange(seq_len, device=inputs.device).unsqueeze(0)
                )
            else:
                state = self.patch(inputs).flatten(2).transpose(1, 2)
                state = (
                    torch.cat([self.cls.expand(inputs.shape[0], -1, -1), state], dim=1)
                    + self.pos
                )
            bsz, length, _ = state.shape
            targets = self.targets[:length]
            valid = self.valid[:length].view(1, 1, length, len(offsets))
            scale = 1.0 / math.sqrt(d_model // n_heads)
            residual_scale = 1.0 / math.sqrt(thought_hops)
            for _ in range(thought_hops):
                z = self.ln_attn(state)
                q = self.q(z).view(bsz, length, n_heads, -1).transpose(1, 2)
                k = self.k(z).view(bsz, length, n_heads, -1).transpose(1, 2)
                v = self.v(z).view(bsz, length, n_heads, -1).transpose(1, 2)
                scores = (q.unsqueeze(3) * k[:, :, targets, :]).sum(-1) * scale
                weights = F.softmax(scores.masked_fill(~valid, -1e4), dim=-1)
                context = (weights.unsqueeze(-1) * v[:, :, targets, :]).sum(3)
                message = self.o(
                    context.transpose(1, 2).contiguous().view(bsz, length, d_model)
                )
                state = state + residual_scale * message
                if mode == "per_hop_mlp":
                    state = state + residual_scale * self.mlp(self.ln_mlp(state))
            if mode == "final_mlp_only":
                state = state + self.mlp(self.ln_mlp(state))
            logits = self.head(self.ln_f(state))
            return logits[:, 0] if domain == "vision" else logits

    model = PlacementModel().to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4 if domain == "vision" else 1e-3,
        weight_decay=0.05 if domain == "vision" else 1e-2,
    )
    start = time.time()
    if domain == "language":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=1e-4
        )
        generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
        for _ in range(steps):
            model.train()
            x, y = get_batch("train", generator)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(30):
                x, y = get_batch("val", generator)
                losses.append(
                    F.cross_entropy(
                        model(x).reshape(-1, vocab_size), y.reshape(-1)
                    ).item()
                )
        val_loss = sum(losses) / len(losses)
        result = {
            "domain": domain,
            "mode": mode,
            "seed": seed,
            "parameters": parameters,
            "steps": steps,
            "seq_len": seq_len,
            "val_loss": round(val_loss, 6),
            "ppl": round(math.exp(val_loss), 4),
        }
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(train_loader), eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        for _ in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                logits = model(images.to(device))
                labels = labels.to(device)
                correct += logits.argmax(-1).eq(labels).sum().item()
                total += labels.numel()
        result = {
            "domain": domain,
            "mode": mode,
            "seed": seed,
            "parameters": parameters,
            "epochs": epochs,
            "seq_len": seq_len,
            "top1": round(100.0 * correct / total, 4),
        }
    result["elapsed_s"] = round(time.time() - start, 3)
    return result


@app.local_entrypoint()
def main():
    import pathlib
    import time

    offsets = [0, 1, 2, 4, 8, 16, 32, 64]
    configs = []
    for mode in ["per_hop_mlp", "final_mlp_only"]:
        for seed in [42, 1337, 2026]:
            configs.append(("language", mode, seed))
        for seed in [42, 1337]:
            configs.append(("vision", mode, seed))
    calls = [train_one.spawn(domain, mode, seed) for domain, mode, seed in configs]
    results = []
    for config, call in zip(configs, calls):
        print(f"Waiting for {config}...", flush=True)
        result = call.get()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    payload = {
        "experiment": "S2-026",
        "description": "Long-sequence per-hop MLP versus final-MLP-only: language L=512, vision L=257",
        "conditions": configs,
        "offsets": offsets,
        "thought_hops": 8,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    path = pathlib.Path("season2/results/s2_026_long_seq_placement.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}", flush=True)
