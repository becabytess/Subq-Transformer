import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy"
    )
)

app = modal.App("subq-twin-full-weight-deep-network", image=image)

@app.function(gpu="T4", timeout=600)
def run_full_weight_twin_study():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.datasets import FashionMNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 105)
    print("  FULL-WEIGHT DEEP NETWORK BENCHMARK: ZERO-BACKPROP UPDATES ACROSS ALL LAYERS")
    print("  Updating: Layer 1 MLP (W1, b1) + Layer 2 SubQ Attention (W2, b2) + Layer 3 Head (W3, b3)")
    print("  Phase 1: Train on 60% of Real Dataset (36,000 Images)")
    print("  Phase 2: Train ALL weights for 5 Epochs on Remaining 40% (24,000 Images) with ZERO BACKPROP")
    print("  Evaluation: 10,000 Completely Held-Out Real Test Images After Every Epoch")
    print("=" * 105)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Split (60% Phase 1, 40% Phase 2)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    full_train = FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root="./data", train=False, download=True, transform=transform)

    n_total = len(full_train) # 60,000
    n_phase1 = int(n_total * 0.60) # 36,000

    torch.manual_seed(42)
    indices = torch.randperm(n_total).tolist()
    phase1_indices = indices[:n_phase1]
    phase2_indices = indices[n_phase1:]

    phase1_dataset = Subset(full_train, phase1_indices)
    phase2_dataset = Subset(full_train, phase2_indices)

    batch_size = 128
    loader_p1 = DataLoader(phase1_dataset, batch_size=batch_size, shuffle=True)
    loader_p2 = DataLoader(phase2_dataset, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"Dataset Ready: {len(phase1_dataset):,} Phase 1 | {len(phase2_dataset):,} Phase 2 | {len(test_dataset):,} Test Set\n")

    d_in = 784 # 28x28
    d_hidden = 256
    d_deep = 128
    d_out = 10

    # 2. SubQ Core Block
    class SubQBlock(nn.Module):
        def __init__(self, d_model=128):
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

    # 3. Full-Weight Deep Network with Twin Credit Assignment
    class FullWeightTwinSystem(nn.Module):
        def __init__(self, d_in=784, d_hidden=256, d_deep=128, d_out=10):
            super().__init__()
            # Layer 1: Input MLP
            self.w1 = nn.Parameter(torch.randn(d_in, d_hidden) * (2.0 / d_in)**0.5)
            self.b1 = nn.Parameter(torch.zeros(d_hidden))

            # Layer 2: Deep MLP Projection
            self.w2 = nn.Parameter(torch.randn(d_hidden, d_deep) * (2.0 / d_hidden)**0.5)
            self.b2 = nn.Parameter(torch.zeros(d_deep))

            # SubQ Deep Relaxation Block
            self.fwd_subq = SubQBlock(d_model=d_deep)

            # Layer 3: Readout Head
            self.w3 = nn.Parameter(torch.randn(d_deep, d_out) * (2.0 / d_deep)**0.5)
            self.b3 = nn.Parameter(torch.zeros(d_out))

            # Error Twin Components
            self.err_subq = SubQBlock(d_model=d_deep)
            self.err_proj_in = nn.Linear(d_out, d_deep)

            # Learnable Meta Learning Rates per Layer
            self.lr1 = nn.Parameter(torch.tensor(0.02))
            self.lr2 = nn.Parameter(torch.tensor(0.02))
            self.lr3 = nn.Parameter(torch.tensor(0.02))

        def forward_with_weights(self, x, w1, b1, w2, b2, w3, b3, T=4):
            B = x.shape[0]
            h0 = x.view(B, -1) # [B, 784]

            # Layer 1
            z1 = torch.mm(h0, w1) + b1 # [B, 256]
            h1 = F.gelu(z1)

            # Layer 2
            z2 = torch.mm(h1, w2) + b2 # [B, 128]
            h2_in = z2.unsqueeze(1) # [B, 1, 128]
            h2 = self.fwd_subq(h2_in, T=T).squeeze(1) # [B, 128]

            # Layer 3
            logits = torch.mm(h2, w3) + b3 # [B, 10]
            return logits, (h0, z1, h1, z2, h2)

        def compute_full_weight_deltas(self, x, y, w1, b1, w2, b2, w3, b3, T_fwd=4, T_err=4):
            B = x.shape[0]
            logits, states = self.forward_with_weights(x, w1, b1, w2, b2, w3, b3, T=T_fwd)
            h0, z1, h1, z2, h2 = states

            prob = F.softmax(logits, dim=-1)
            y_onehot = F.one_hot(y, num_classes=10).float()
            e_out = y_onehot - prob # [B, 10] Error at output layer

            # Layer 3 Delta (Output Head)
            dw3 = torch.mm(h2.t(), e_out) / B # [128, 10]
            db3 = e_out.mean(dim=0) # [10]

            # Error Twin Multi-Hop Settling for Deep Layer 2
            e2_in = self.err_proj_in(e_out).unsqueeze(1) # [B, 1, 128]
            e2 = self.err_subq(e2_in, T=T_err).squeeze(1) # [B, 128]

            # Layer 2 Delta (Deep MLP)
            dw2 = torch.mm(h1.t(), e2) / B # [256, 128]
            db2 = e2.mean(dim=0) # [128]

            # Propagate Settled Error to Layer 1
            e1 = torch.mm(e2, w2.t()) * (torch.sigmoid(1.702 * z1) * (1.0 + torch.exp(-1.702 * z1))) # Approximate GELU derivative

            # Layer 1 Delta (Input MLP)
            dw1 = torch.mm(h0.t(), e1) / B # [784, 256]
            db1 = e1.mean(dim=0) # [256]

            return (dw1, db1, dw2, db2, dw3, db3), logits

    # 4. Standard Supervised Model (Full-Weight Backpropagation Baseline)
    class StandardDeepModel(nn.Module):
        def __init__(self, d_in=784, d_hidden=256, d_deep=128, d_out=10):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hidden)
            self.fc2 = nn.Linear(d_hidden, d_deep)
            self.fwd_subq = SubQBlock(d_model=d_deep)
            self.fc3 = nn.Linear(d_deep, d_out)

        def forward(self, x, T=4):
            B = x.shape[0]
            h0 = x.view(B, -1)
            h1 = F.gelu(self.fc1(h0))
            h2_in = self.fc2(h1).unsqueeze(1)
            h2 = self.fwd_subq(h2_in, T=T).squeeze(1)
            return self.fc3(h2)

    def evaluate_twin(twin_sys, weights, loader):
        twin_sys.eval()
        w1, b1, w2, b2, w3, b3 = weights
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                logits, _ = twin_sys.forward_with_weights(bx, w1, b1, w2, b2, w3, b3, T=4)
                preds = logits.argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += len(by)
        return (correct / total) * 100.0

    def evaluate_std(model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx, T=4)
                preds = logits.argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += len(by)
        return (correct / total) * 100.0

    # -------------------------------------------------------------------------
    # Phase 1: Train Models on 60% Data (5 Epochs)
    # -------------------------------------------------------------------------
    p1_epochs = 5
    print("Phase 1: Meta-Training Full-Weight Models on 60% Data (36,000 Images)...")

    std_model = StandardDeepModel(d_in, d_hidden, d_deep, d_out).to(device)
    opt_std = torch.optim.AdamW(std_model.parameters(), lr=1e-3, weight_decay=1e-4)

    twin_sys = FullWeightTwinSystem(d_in, d_hidden, d_deep, d_out).to(device)
    opt_twin = torch.optim.AdamW(twin_sys.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for ep in range(1, p1_epochs + 1):
        std_model.train()
        twin_sys.train()

        for bx, by in loader_p1:
            bx, by = bx.to(device), by.to(device)

            # Train Standard Model
            logits_std = std_model(bx, T=4)
            loss_std = F.cross_entropy(logits_std, by)
            opt_std.zero_grad()
            loss_std.backward()
            opt_std.step()

            # Train Twin System
            w_tuple = (twin_sys.w1, twin_sys.b1, twin_sys.w2, twin_sys.b2, twin_sys.w3, twin_sys.b3)
            deltas, logits_twin = twin_sys.compute_full_weight_deltas(bx, by, *w_tuple, T_fwd=4, T_err=4)
            loss_twin = F.cross_entropy(logits_twin, by)
            opt_twin.zero_grad()
            loss_twin.backward()
            opt_twin.step()

        acc_std = evaluate_std(std_model, loader_test)
        w_tuple = (twin_sys.w1, twin_sys.b1, twin_sys.w2, twin_sys.b2, twin_sys.w3, twin_sys.b3)
        acc_twin = evaluate_twin(twin_sys, w_tuple, loader_test)
        print(f"  [Phase 1] Epoch {ep:>2}/{p1_epochs} | Standard Test Acc: {acc_std:.2f}% | Twin Test Acc: {acc_twin:.2f}%")

    print(f"Phase 1 completed in {time.time() - t0:.1f}s\n")

    # -------------------------------------------------------------------------
    # Phase 2: FULL-WEIGHT Zero-Backprop Continual Learning (5 Epochs on 40% Data)
    # -------------------------------------------------------------------------
    print("=" * 105)
    print("  PHASE 2: FULL-WEIGHT MULTI-EPOCH LEARNING ON 40% UNSEEN REAL DATA (24,000 IMAGES)")
    print("  Updating ALL Weights: W1, b1, W2, b2, W3, b3 with ZERO Backprop")
    print("=" * 105)
    print(f"{'Epoch on 40% Stream':<22} | {'Static (60% Only) Acc':<24} | {'Online SGD (Backprop) Acc':<28} | {'SubQ Twin (Zero Backprop) Acc':<30} | {'Status'}")
    print("-" * 115)

    base_static_acc = evaluate_std(std_model, loader_test)

    # Model 1: Static Base (frozen)
    # Model 2: Online Backprop SGD (continues training all layers via SGD backprop)
    sgd_online_model = StandardDeepModel(d_in, d_hidden, d_deep, d_out).to(device)
    sgd_online_model.load_state_dict(std_model.state_dict())
    opt_online_sgd = torch.optim.SGD(sgd_online_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # Model 3: SubQ Twin (ZERO BACKPROP - Accumulate deltas across ALL layers)
    twin_sys.eval()
    cur_w1 = twin_sys.w1.clone()
    cur_b1 = twin_sys.b1.clone()
    cur_w2 = twin_sys.w2.clone()
    cur_b2 = twin_sys.b2.clone()
    cur_w3 = twin_sys.w3.clone()
    cur_b3 = twin_sys.b3.clone()

    lr1 = twin_sys.lr1.item()
    lr2 = twin_sys.lr2.item()
    lr3 = twin_sys.lr3.item()

    w_tuple_init = (cur_w1, cur_b1, cur_w2, cur_b2, cur_w3, cur_b3)
    init_twin_acc = evaluate_twin(twin_sys, w_tuple_init, loader_test)
    print(f"Epoch 0 (Pre-Stream)  | {base_static_acc:>20.2f}% | {base_static_acc:>24.2f}% | {init_twin_acc:>26.2f}% | Baseline Prior")

    # Train ALL weights across all layers for 5 Full Epochs over the 24,000 new images
    p2_epochs = 5
    for ep in range(1, p2_epochs + 1):
        # 1. Full-Weight Online SGD Update (Backprop)
        sgd_online_model.train()
        for bx, by in loader_p2:
            bx, by = bx.to(device), by.to(device)
            logits_sgd = sgd_online_model(bx, T=4)
            loss_sgd = F.cross_entropy(logits_sgd, by)
            opt_online_sgd.zero_grad()
            loss_sgd.backward()
            opt_online_sgd.step()

        # 2. Full-Weight SubQ Twin Update (ZERO BACKPROP - Updates ALL Layers!)
        with torch.no_grad():
            for bx, by in loader_p2:
                bx, by = bx.to(device), by.to(device)
                w_tuple = (cur_w1, cur_b1, cur_w2, cur_b2, cur_w3, cur_b3)
                deltas, _ = twin_sys.compute_full_weight_deltas(bx, by, *w_tuple, T_fwd=4, T_err=4)
                dw1, db1, dw2, db2, dw3, db3 = deltas

                # Apply weight updates to EVERY layer in the deep network
                cur_w1 = cur_w1 + lr1 * dw1
                cur_b1 = cur_b1 + lr1 * db1
                cur_w2 = cur_w2 + lr2 * dw2
                cur_b2 = cur_b2 + lr2 * db2
                cur_w3 = cur_w3 + lr3 * dw3
                cur_b3 = cur_b3 + lr3 * db3

        # Evaluate on 10,000 held-out test images
        acc_sgd_ep = evaluate_std(sgd_online_model, loader_test)
        w_tuple_current = (cur_w1, cur_b1, cur_w2, cur_b2, cur_w3, cur_b3)
        acc_twin_ep = evaluate_twin(twin_sys, w_tuple_current, loader_test)

        delta_sgd = acc_sgd_ep - base_static_acc
        delta_twin = acc_twin_ep - init_twin_acc

        winner = "🏆 SubQ Twin (Zero Backprop)" if acc_twin_ep >= acc_sgd_ep else "— Backprop"
        print(f"Epoch {ep:>2}/{p2_epochs} on 40% Stream | {base_static_acc:>20.2f}% | {acc_sgd_ep:>24.2f}% (+{delta_sgd:+.2f}%) | {acc_twin_ep:>26.2f}% (+{delta_twin:+.2f}%) | {winner}")

    print("=" * 115)

@app.local_entrypoint()
def main():
    run_full_weight_twin_study.remote()
