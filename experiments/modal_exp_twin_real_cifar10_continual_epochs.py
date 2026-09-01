import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy"
    )
)

app = modal.App("subq-twin-real-cifar10-epochs", image=image)

@app.function(gpu="T4", timeout=1200)
def run_real_cifar10_study():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 105)
    print("  REAL-WORLD BENCHMARK: CONTINUAL MULTI-EPOCH SELF-TRAINING ON CIFAR-10")
    print("  Phase 1: Train on 60% of CIFAR-10 (30,000 Real Images) with Standard Backprop")
    print("  Phase 2: Train for 5 Full Epochs on Remaining 40% (20,000 Images) with ZERO BACKPROP")
    print("  Evaluation: Test Accuracy on 10,000 Completely Held-Out CIFAR-10 Test Images")
    print("=" * 105)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. CIFAR-10 Dataset & 60% / 40% Split
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("Downloading and preparing CIFAR-10 dataset...")
    full_train = CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    n_total = len(full_train) # 50,000
    n_phase1 = int(n_total * 0.60) # 30,000
    n_phase2 = n_total - n_phase1  # 20,000

    torch.manual_seed(42)
    indices = torch.randperm(n_total).tolist()
    phase1_indices = indices[:n_phase1]
    phase2_indices = indices[n_phase1:]

    phase1_dataset = Subset(full_train, phase1_indices)
    phase2_dataset = Subset(full_train, phase2_indices)

    batch_size = 128
    loader_p1 = DataLoader(phase1_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_p2 = DataLoader(phase2_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_test = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    print(f"CIFAR-10 Dataset Ready:")
    print(f"  • Phase 1 (60% Initial Training): {len(phase1_dataset):,} images")
    print(f"  • Phase 2 (40% Zero-Backprop Stream): {len(phase2_dataset):,} images")
    print(f"  • Held-Out Test Set: {len(test_dataset):,} images\n")

    # 2. ConvNet Feature Extractor Backbone
    class ConvBackbone(nn.Module):
        def __init__(self, d_feat=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2), # 16x16
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2), # 8x8
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)) # 128
            )

        def forward(self, x):
            return self.net(x).squeeze(-1).squeeze(-1)

    # 3. SubQ Core Block
    class SubQBlock(nn.Module):
        def __init__(self, d_model=128, n_heads=4):
            super().__init__()
            self.d_model = d_model
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=False)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=False)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)

        def forward(self, h_init, T=4):
            s = h_init
            for _ in range(T):
                Q = self.q_proj(s)
                K = self.k_proj(s)
                V = self.v_proj(s)

                scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)
                attn = F.softmax(scores, dim=-1)
                context = self.out_proj(torch.bmm(attn, V))

                gates_ih = self.w_ih(context)
                r_ih, z_ih, n_ih = gates_ih.chunk(3, dim=-1)

                gates_h = self.w_gate_h(s)
                r_h, z_h = gates_h.chunk(2, dim=-1)

                r = torch.sigmoid(r_ih + r_h)
                z = torch.sigmoid(z_ih + z_h)
                n = torch.tanh(n_ih + self.w_cand_h(r * s))

                s = (1.0 - z) * n + z * s
            return s

    # 4. CIFAR-10 Twin Architecture
    class CIFAR10TwinSystem(nn.Module):
        def __init__(self, d_feat=128, n_classes=10):
            super().__init__()
            self.backbone = ConvBackbone(d_feat=d_feat)
            self.d_feat = d_feat
            self.n_classes = n_classes

            # Forward SubQ
            self.fwd_subq = SubQBlock(d_model=d_feat)
            self.head = nn.Linear(d_feat, n_classes, bias=False)

            # Error Twin
            self.err_proj = nn.Linear(n_classes, d_feat)
            self.err_subq = SubQBlock(d_model=d_feat)
            self.meta_lr = nn.Parameter(torch.tensor(0.1))

        def forward_pass(self, x, T=4, custom_head=None):
            feats = self.backbone(x).unsqueeze(1) # [B, 1, 128]
            h_fwd = self.fwd_subq(feats, T=T).squeeze(1) # [B, 128]

            if custom_head is None:
                logits = self.head(h_fwd)
            else:
                logits = F.linear(h_fwd, custom_head.t())
            return logits, h_fwd

        def compute_twin_delta(self, x, y, T_fwd=4, T_err=4, current_head=None):
            B = x.shape[0]
            logits, h_fwd = self.forward_pass(x, T=T_fwd, custom_head=current_head)

            prob = F.softmax(logits, dim=-1)
            y_onehot = F.one_hot(y, num_classes=self.n_classes).float()
            error = y_onehot - prob # [B, 10]

            h_err_in = self.err_proj(error).unsqueeze(1) # [B, 1, 128]
            h_err = self.err_subq(h_err_in, T=T_err).squeeze(1) # [B, 128]

            # Bilinear outer product weight delta: [128, 10]
            delta_fwd = torch.mm(h_fwd.t(), error) / B
            delta_err = torch.mm(h_err.t(), error) / B
            return (delta_fwd + delta_err), logits

    # 5. Standard Supervised Model (Backprop Only Baseline)
    class StandardCIFAR10(nn.Module):
        def __init__(self, d_feat=128, n_classes=10):
            super().__init__()
            self.backbone = ConvBackbone(d_feat=d_feat)
            self.fwd_subq = SubQBlock(d_model=d_feat)
            self.head = nn.Linear(d_feat, n_classes, bias=False)

        def forward(self, x, T=4):
            feats = self.backbone(x).unsqueeze(1)
            h_fwd = self.fwd_subq(feats, T=T).squeeze(1)
            return self.head(h_fwd)

    def evaluate_test_accuracy(model, custom_head=None, is_twin=False):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in loader_test:
                bx, by = bx.to(device), by.to(device)
                if is_twin:
                    logits, _ = model.forward_pass(bx, T=4, custom_head=custom_head)
                else:
                    logits = model(bx, T=4)
                preds = logits.argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += len(by)
        return (correct / total) * 100.0

    # -------------------------------------------------------------------------
    # PHASE 1: Train Models on 60% CIFAR-10 Data (8 Epochs)
    # -------------------------------------------------------------------------
    p1_epochs = 8
    print("Phase 1: Meta-Training Models on 60% CIFAR-10 Data (8 Epochs)...")

    # 1. Standard Model
    std_model = StandardCIFAR10(d_feat=128, n_classes=10).to(device)
    opt_std = torch.optim.AdamW(std_model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 2. Twin System
    twin_system = CIFAR10TwinSystem(d_feat=128, n_classes=10).to(device)
    opt_twin = torch.optim.AdamW(twin_system.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for ep in range(1, p1_epochs + 1):
        std_model.train()
        twin_system.train()

        for bx, by in loader_p1:
            bx, by = bx.to(device), by.to(device)

            # Train Standard Model
            logits_std = std_model(bx, T=4)
            loss_std = F.cross_entropy(logits_std, by)
            opt_std.zero_grad()
            loss_std.backward()
            opt_std.step()

            # Train Twin System
            delta, logits_twin = twin_system.compute_twin_delta(bx, by, T_fwd=4, T_err=4)
            loss_twin = F.cross_entropy(logits_twin, by)
            opt_twin.zero_grad()
            loss_twin.backward()
            opt_twin.step()

        acc_std = evaluate_test_accuracy(std_model)
        acc_twin = evaluate_test_accuracy(twin_system, is_twin=True)
        print(f"  [Phase 1] Epoch {ep:>2}/{p1_epochs} | Standard Test Acc: {acc_std:.2f}% | Twin Test Acc: {acc_twin:.2f}%")

    print(f"Phase 1 completed in {time.time() - t0:.1f}s\n")

    # -------------------------------------------------------------------------
    # PHASE 2: Continual Multi-Epoch Training on Remaining 40% Data (Zero Backprop!)
    # -------------------------------------------------------------------------
    print("=" * 105)
    print("  PHASE 2: CONTINUAL MULTI-EPOCH LEARNING ON 40% UNSEEN DATA (20,000 IMAGES)")
    print("=" * 105)
    print(f"{'Epoch on 40% Stream':<22} | {'Static (60% Only) Acc':<24} | {'Online SGD (Backprop) Acc':<28} | {'SubQ Twin (Zero Backprop) Acc':<30} | {'Status'}")
    print("-" * 115)

    # Initial Test Accuracies before Phase 2
    base_static_acc = evaluate_test_accuracy(std_model)

    # Model 1: Static Base (frozen)
    # Model 2: Online Backprop SGD (continues training on 40% data via backprop)
    sgd_online_model = StandardCIFAR10(d_feat=128, n_classes=10).to(device)
    sgd_online_model.load_state_dict(std_model.state_dict())
    opt_online_sgd = torch.optim.SGD(sgd_online_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # Model 3: SubQ Twin (ZERO BACKPROP - Only forward relaxation weight accumulation)
    twin_system.eval()
    accumulated_twin_head = twin_system.head.weight.t().clone() # [128, 10]
    twin_lr = 0.02

    init_twin_acc = evaluate_test_accuracy(twin_system, custom_head=accumulated_twin_head, is_twin=True)
    print(f"Epoch 0 (Pre-Stream)  | {base_static_acc:>20.2f}% | {base_static_acc:>24.2f}% | {init_twin_acc:>26.2f}% | Baseline Prior")

    # Train for 5 Full Epochs over the 20,000 new images
    p2_epochs = 5
    for ep in range(1, p2_epochs + 1):
        # 1. Train Online Backprop Model
        sgd_online_model.train()
        for bx, by in loader_p2:
            bx, by = bx.to(device), by.to(device)
            logits_sgd = sgd_online_model(bx, T=4)
            loss_sgd = F.cross_entropy(logits_sgd, by)
            opt_online_sgd.zero_grad()
            loss_sgd.backward()
            opt_online_sgd.step()

        # 2. Train SubQ Twin (ZERO BACKPROP - Forward Error Relaxation Only)
        with torch.no_grad():
            for bx, by in loader_p2:
                bx, by = bx.to(device), by.to(device)
                twin_delta, _ = twin_system.compute_twin_delta(bx, by, T_fwd=4, T_err=4, current_head=accumulated_twin_head)
                accumulated_twin_head = accumulated_twin_head + twin_lr * twin_delta

        # Evaluate all models on 10,000 held-out test images
        acc_sgd_ep = evaluate_test_accuracy(sgd_online_model)
        acc_twin_ep = evaluate_test_accuracy(twin_system, custom_head=accumulated_twin_head, is_twin=True)

        delta_sgd = acc_sgd_ep - base_static_acc
        delta_twin = acc_twin_ep - init_twin_acc

        print(f"Epoch {ep:>2}/{p2_epochs} on 40% Stream | {base_static_acc:>20.2f}% | {acc_sgd_ep:>24.2f}% (+{delta_sgd:.2f}%) | {acc_twin_ep:>26.2f}% (+{delta_twin:.2f}%) | Continual Learning")

    print("=" * 115)

@app.local_entrypoint()
def main():
    run_real_cifar10_study.remote()
