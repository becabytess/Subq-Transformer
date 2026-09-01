import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy"
    )
)

app = modal.App("subq-twin-progressive-layerwise", image=image)

@app.function(gpu="T4", timeout=900)
def run_progressive_study():
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
    print("=" * 115)
    print("  PROGRESSIVE TOP-DOWN LAYERWISE TWIN: CALIBRATED MULTI-LAYER ZERO-BACKPROP TRAINING")
    print("  Comparing 4 Methods on Real Fashion-MNIST (60,000 Real Images):")
    print("  1. Static Baseline (60% Data Only)")
    print("  2. Online Backprop SGD (Full Network Gradient Descent on 40% Data)")
    print("  3. Head-Only SubQ Twin (Zero Backprop)")
    print("  4. Progressive Top-Down Layerwise SubQ Twin (Zero Backprop: W3 -> W2 -> W1)")
    print("=" * 115)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Split (60% Phase 1 = 36,000, 40% Phase 2 = 24,000)
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

    d_in = 784
    d_hidden = 128
    d_out = 10

    # 2. SubQ Core Block
    class SubQCore(nn.Module):
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

    # 3. Model with Error Twin
    class ProgressiveTwinSystem(nn.Module):
        def __init__(self, d_in=784, d_hidden=128, d_out=10):
            super().__init__()
            self.d_in = d_in
            self.d_hidden = d_hidden
            self.d_out = d_out

            # Layer 1 Base Weights
            self.w1 = nn.Parameter(torch.randn(d_in, d_hidden) * (2.0 / d_in)**0.5)
            self.b1 = nn.Parameter(torch.zeros(d_hidden))
            self.ln1 = nn.LayerNorm(d_hidden)

            # Layer 2 SubQ Thought Block
            self.w2 = nn.Parameter(torch.randn(d_hidden, d_hidden) * (2.0 / d_hidden)**0.5)
            self.b2 = nn.Parameter(torch.zeros(d_hidden))
            self.ln2 = nn.LayerNorm(d_hidden)
            self.fwd_subq = SubQCore(d_model=d_hidden)

            # Layer 3 Readout Head
            self.w3 = nn.Parameter(torch.randn(d_hidden, d_out) * (2.0 / d_hidden)**0.5)
            self.b3 = nn.Parameter(torch.zeros(d_out))

            # Error Twin
            self.err_subq = SubQCore(d_model=d_hidden)

            # Layerwise Learning Rates
            self.lr1 = nn.Parameter(torch.tensor(0.005))
            self.lr2 = nn.Parameter(torch.tensor(0.01))
            self.lr3 = nn.Parameter(torch.tensor(0.02))

        def forward_with_weights(self, x, w1, b1, w2, b2, w3, b3, T=4):
            B = x.shape[0]
            h0 = x.view(B, -1) # [B, 784]

            # Layer 1
            z1 = torch.mm(h0, w1) + b1
            h1 = F.gelu(self.ln1(z1)) # [B, 128]

            # Layer 2
            z2 = torch.mm(h1, w2) + b2
            h2_in = self.ln2(z2).unsqueeze(1) # [B, 1, 128]
            h2 = self.fwd_subq(h2_in, T=T).squeeze(1) # [B, 128]

            # Layer 3
            logits = torch.mm(h2, w3) + b3 # [B, 10]
            return logits, (h0, h1, h2)

    # 4. Standard Supervised Baseline Model
    class StandardDeepModel(nn.Module):
        def __init__(self, d_in=784, d_hidden=128, d_out=10):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hidden)
            self.ln1 = nn.LayerNorm(d_hidden)
            self.fc2 = nn.Linear(d_hidden, d_hidden)
            self.ln2 = nn.LayerNorm(d_hidden)
            self.fwd_subq = SubQCore(d_model=d_hidden)
            self.fc3 = nn.Linear(d_hidden, d_out)

        def forward(self, x, T=4):
            B = x.shape[0]
            h0 = x.view(B, -1)
            h1 = F.gelu(self.ln1(self.fc1(h0)))
            h2_in = self.ln2(self.fc2(h1)).unsqueeze(1)
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
    print("Phase 1: Meta-Training Models on 60% Data (36,000 Images)...")

    std_model = StandardDeepModel(d_in, d_hidden, d_out).to(device)
    opt_std = torch.optim.AdamW(std_model.parameters(), lr=1e-3, weight_decay=1e-4)

    twin_sys = ProgressiveTwinSystem(d_in, d_hidden, d_out).to(device)
    opt_twin = torch.optim.AdamW(twin_sys.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for ep in range(1, p1_epochs + 1):
        std_model.train()
        twin_sys.train()

        p1_batches = list(loader_p1)
        for i in range(0, len(p1_batches) - 1, 2):
            b_supp_x, b_supp_y = p1_batches[i][0].to(device), p1_batches[i][1].to(device)
            b_query_x, b_query_y = p1_batches[i+1][0].to(device), p1_batches[i+1][1].to(device)

            # 1. Train Standard Model
            logits_std = std_model(b_supp_x, T=4)
            loss_std = F.cross_entropy(logits_std, b_supp_y)
            opt_std.zero_grad()
            loss_std.backward()
            opt_std.step()

            # 2. Meta-Train Twin System
            # Forward pass on support
            w_tuple = (twin_sys.w1, twin_sys.b1, twin_sys.w2, twin_sys.b2, twin_sys.w3, twin_sys.b3)
            logits_s, (h0_s, h1_s, h2_s) = twin_sys.forward_with_weights(b_supp_x, *w_tuple, T=4)

            # Label Error
            e_out_s = F.one_hot(b_supp_y, 10).float() - F.softmax(logits_s, dim=-1)
            B_s = b_supp_x.shape[0]

            # Step 1: Head Delta
            dw3 = torch.mm(h2_s.t(), e_out_s) / (B_s * math.sqrt(d_hidden))
            db3 = e_out_s.mean(dim=0)
            w3_adapt = twin_sys.w3 + twin_sys.lr3 * dw3
            b3_adapt = twin_sys.b3 + twin_sys.lr3 * db3

            # Step 2: Layer 2 Delta using freshly adapted w3
            e2_in = torch.mm(e_out_s, w3_adapt.t()).unsqueeze(1)
            e2 = twin_sys.err_subq(e2_in, T=4).squeeze(1)
            dw2 = torch.mm(h1_s.t(), e2) / (B_s * math.sqrt(d_hidden))
            db2 = e2.mean(dim=0)
            w2_adapt = twin_sys.w2 + twin_sys.lr2 * dw2
            b2_adapt = twin_sys.b2 + twin_sys.lr2 * db2

            # Step 3: Layer 1 Delta using freshly adapted w2
            e1 = torch.mm(e2, w2_adapt.t())
            dw1 = torch.mm(h0_s.t(), e1) / (B_s * math.sqrt(d_in))
            db1 = e1.mean(dim=0)
            w1_adapt = twin_sys.w1 + twin_sys.lr1 * dw1
            b1_adapt = twin_sys.b1 + twin_sys.lr1 * db1

            # Meta-Loss on Query
            w_adapted_all = (w1_adapt, b1_adapt, w2_adapt, b2_adapt, w3_adapt, b3_adapt)
            logits_q, _ = twin_sys.forward_with_weights(b_query_x, *w_adapted_all, T=4)
            loss_meta = F.cross_entropy(logits_q, b_query_y)

            opt_twin.zero_grad()
            loss_meta.backward()
            nn.utils.clip_grad_norm_(twin_sys.parameters(), 1.0)
            opt_twin.step()

        acc_std = evaluate_std(std_model, loader_test)
        w_base = (twin_sys.w1, twin_sys.b1, twin_sys.w2, twin_sys.b2, twin_sys.w3, twin_sys.b3)
        acc_twin = evaluate_twin(twin_sys, w_base, loader_test)
        print(f"  [Phase 1] Epoch {ep:>2}/{p1_epochs} | Standard Test Acc: {acc_std:.2f}% | Twin Base Acc: {acc_twin:.2f}%")

    print(f"Phase 1 completed in {time.time() - t0:.1f}s\n")

    # -------------------------------------------------------------------------
    # Phase 2: 4-WAY CONTINUAL MULTI-EPOCH LEARNING ON 40% DATA (24,000 IMAGES)
    # -------------------------------------------------------------------------
    print("=" * 125)
    print("  PHASE 2: 4-WAY CONTINUAL MULTI-EPOCH COMPARISON (24,000 UNSEEN IMAGES, 5 EPOCHS)")
    print("=" * 125)
    print(f"{'Epoch on 40% Stream':<20} | {'Static Base':<12} | {'Online SGD (Backprop)':<22} | {'Head-Only Twin':<18} | {'Progressive Layerwise Twin (Zero Backprop)':<42}")
    print("-" * 125)

    base_static_acc = evaluate_std(std_model, loader_test)

    # 1. Online SGD Model
    sgd_model = StandardDeepModel(d_in, d_hidden, d_out).to(device)
    sgd_model.load_state_dict(std_model.state_dict())
    opt_sgd = torch.optim.SGD(sgd_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # 2. Head-Only Twin Weights
    twin_sys.eval()
    head_w3 = twin_sys.w3.clone()
    head_b3 = twin_sys.b3.clone()

    # 3. Progressive Layerwise Twin Weights (Full Network: W1, b1, W2, b2, W3, b3)
    prog_w1 = twin_sys.w1.clone()
    prog_b1 = twin_sys.b1.clone()
    prog_w2 = twin_sys.w2.clone()
    prog_b2 = twin_sys.b2.clone()
    prog_w3 = twin_sys.w3.clone()
    prog_b3 = twin_sys.b3.clone()

    lr1 = twin_sys.lr1.item()
    lr2 = twin_sys.lr2.item()
    lr3 = twin_sys.lr3.item()

    w_prog_init = (prog_w1, prog_b1, prog_w2, prog_b2, prog_w3, prog_b3)
    init_prog_acc = evaluate_twin(twin_sys, w_prog_init, loader_test)

    print(f"Epoch 0 (Pre-Stream) | {base_static_acc:>10.2f}% | {base_static_acc:>20.2f}% | {init_prog_acc:>16.2f}% | {init_prog_acc:>40.2f}%")

    p2_epochs = 5
    for ep in range(1, p2_epochs + 1):
        # A. Online SGD Update (Backprop)
        sgd_model.train()
        for bx, by in loader_p2:
            bx, by = bx.to(device), by.to(device)
            logits = sgd_model(bx, T=4)
            loss = F.cross_entropy(logits, by)
            opt_sgd.zero_grad()
            loss.backward()
            opt_sgd.step()

        # B. Head-Only Twin Update (Zero Backprop)
        with torch.no_grad():
            for bx, by in loader_p2:
                bx, by = bx.to(device), by.to(device)
                w_head_tuple = (twin_sys.w1, twin_sys.b1, twin_sys.w2, twin_sys.b2, head_w3, head_b3)
                logits_h, (_, _, h2) = twin_sys.forward_with_weights(bx, *w_head_tuple, T=4)
                e_out = F.one_hot(by, 10).float() - F.softmax(logits_h, dim=-1)
                B = bx.shape[0]
                dw3 = torch.mm(h2.t(), e_out) / (B * math.sqrt(d_hidden))
                db3 = e_out.mean(dim=0)
                head_w3 = (1.0 - 1e-4) * head_w3 + lr3 * dw3
                head_b3 = head_b3 + lr3 * db3

        # C. Progressive Top-Down Layerwise Twin Update (Zero Backprop: W3 -> W2 -> W1)
        with torch.no_grad():
            for bx, by in loader_p2:
                bx, by = bx.to(device), by.to(device)
                w_prog_tuple = (prog_w1, prog_b1, prog_w2, prog_b2, prog_w3, prog_b3)
                logits_p, (h0, h1, h2) = twin_sys.forward_with_weights(bx, *w_prog_tuple, T=4)
                e_out = F.one_hot(by, 10).float() - F.softmax(logits_p, dim=-1)
                B = bx.shape[0]

                # Step 1: Update Head W3
                dw3 = torch.mm(h2.t(), e_out) / (B * math.sqrt(d_hidden))
                db3 = e_out.mean(dim=0)
                prog_w3 = (1.0 - 1e-4) * prog_w3 + lr3 * dw3
                prog_b3 = prog_b3 + lr3 * db3

                # Step 2: Project Error through calibrated W3 to Layer 2 & Update W2
                e2_in = torch.mm(e_out, prog_w3.t()).unsqueeze(1)
                e2 = twin_sys.err_subq(e2_in, T=4).squeeze(1)
                dw2 = torch.mm(h1.t(), e2) / (B * math.sqrt(d_hidden))
                db2 = e2.mean(dim=0)
                prog_w2 = (1.0 - 1e-4) * prog_w2 + lr2 * dw2
                prog_b2 = prog_b2 + lr2 * db2

                # Step 3: Project Error through calibrated W2 to Layer 1 & Update W1
                e1 = torch.mm(e2, prog_w2.t())
                dw1 = torch.mm(h0.t(), e1) / (B * math.sqrt(d_in))
                db1 = e1.mean(dim=0)
                prog_w1 = (1.0 - 1e-4) * prog_w1 + lr1 * dw1
                prog_b1 = prog_b1 + lr1 * db1

        # Evaluate on 10,000 held-out test images
        acc_sgd = evaluate_std(sgd_model, loader_test)
        w_head_eval = (twin_sys.w1, twin_sys.b1, twin_sys.w2, twin_sys.b2, head_w3, head_b3)
        acc_head = evaluate_twin(twin_sys, w_head_eval, loader_test)
        w_prog_eval = (prog_w1, prog_b1, prog_w2, prog_b2, prog_w3, prog_b3)
        acc_prog = evaluate_twin(twin_sys, w_prog_eval, loader_test)

        delta_sgd = acc_sgd - base_static_acc
        delta_head = acc_head - init_prog_acc
        delta_prog = acc_prog - init_prog_acc

        print(f"Epoch {ep:>2}/5 on 40% Stream | {base_static_acc:>10.2f}% | {acc_sgd:>15.2f}% ({delta_sgd:+.2f}%) | {acc_head:>11.2f}% ({delta_head:+.2f}%) | {acc_prog:>30.2f}% ({delta_prog:+.2f}%)")

    print("=" * 125)

@app.local_entrypoint()
def main():
    run_progressive_study.remote()
