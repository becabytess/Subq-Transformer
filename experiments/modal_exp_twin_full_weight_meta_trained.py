import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy"
    )
)

app = modal.App("subq-twin-full-weight-meta-trained", image=image)

@app.function(gpu="T4", timeout=900)
def run_full_weight_meta_study():
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
    print("  TRUE FULL-WEIGHT META-TRAINED TWIN: LEARNING CREDIT ASSIGNMENT ACROSS ALL LAYERS")
    print("  Phase 1: Meta-Train Error Twin to Jointly Update ALL Layers: W1 (Input), W2 (SubQ), W3 (Head)")
    print("  Phase 2: Train ALL Weights for 5 Epochs on 40% Unseen Data (24,000 Images) with ZERO BACKPROP")
    print("  Evaluation: 10,000 Completely Held-Out Real Test Images After Every Single Epoch")
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

    d_in = 784
    d_hidden = 128
    d_out = 10

    # 2. SubQ Core Block (Functional so weights can be adapted)
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

    # 3. Full-Weight Meta-Trained Twin Architecture
    class FullWeightMetaTwinSystem(nn.Module):
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
            self.fwd_subq = SubQCore(d_model=d_hidden)
            self.w2 = nn.Parameter(torch.randn(d_hidden, d_hidden) * (2.0 / d_hidden)**0.5)
            self.b2 = nn.Parameter(torch.zeros(d_hidden))
            self.ln2 = nn.LayerNorm(d_hidden)

            # Layer 3 Readout Head
            self.w3 = nn.Parameter(torch.randn(d_hidden, d_out) * (2.0 / d_hidden)**0.5)
            self.b3 = nn.Parameter(torch.zeros(d_out))

            # Error Twin Components
            self.err_in_proj = nn.Linear(d_out, d_hidden)
            self.err_subq = SubQCore(d_model=d_hidden)
            self.err_back_proj = nn.Linear(d_hidden, d_hidden)

            # Learnable Per-Layer Adaptation Rates
            self.lr1 = nn.Parameter(torch.tensor(0.01))
            self.lr2 = nn.Parameter(torch.tensor(0.01))
            self.lr3 = nn.Parameter(torch.tensor(0.01))

        def forward_pass(self, x, w1, b1, w2, b2, w3, b3, T=4):
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

        def compute_deltas(self, x, y, w1, b1, w2, b2, w3, b3, T_fwd=4, T_err=4):
            B = x.shape[0]
            logits, (h0, h1, h2) = self.forward_pass(x, w1, b1, w2, b2, w3, b3, T=T_fwd)

            # Output Categorical Error
            prob = F.softmax(logits, dim=-1)
            y_onehot = F.one_hot(y, num_classes=self.d_out).float()
            e_out = y_onehot - prob # [B, 10]

            # Layer 3 Delta
            dw3 = torch.mm(h2.t(), e_out) / (B * math.sqrt(self.d_hidden))
            db3 = e_out.mean(dim=0)

            # Error Twin Multi-Hop Settling for Layer 2
            e2_in = self.err_in_proj(e_out).unsqueeze(1) # [B, 1, 128]
            e2 = self.err_subq(e2_in, T=T_err).squeeze(1) # [B, 128]

            # Layer 2 Delta
            dw2 = torch.mm(h1.t(), e2) / (B * math.sqrt(self.d_hidden))
            db2 = e2.mean(dim=0)

            # Propagate Settled Error to Layer 1
            e1 = self.err_back_proj(e2) # [B, 128]

            # Layer 1 Delta (Normalized by fan-in sqrt(d_in))
            dw1 = torch.mm(h0.t(), e1) / (B * math.sqrt(self.d_in))
            db1 = e1.mean(dim=0)

            return (dw1, db1, dw2, db2, dw3, db3), logits

        def meta_adapt_weights(self, dw1, db1, dw2, db2, dw3, db3):
            w1_adapt = self.w1 + self.lr1 * dw1
            b1_adapt = self.b1 + self.lr1 * db1
            w2_adapt = self.w2 + self.lr2 * dw2
            b2_adapt = self.b2 + self.lr2 * db2
            w3_adapt = self.w3 + self.lr3 * dw3
            b3_adapt = self.b3 + self.lr3 * db3
            return (w1_adapt, b1_adapt, w2_adapt, b2_adapt, w3_adapt, b3_adapt)

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
                logits, _ = twin_sys.forward_pass(bx, w1, b1, w2, b2, w3, b3, T=4)
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
    # Phase 1: TRUE Full-Weight Meta-Training on 60% Data (5 Epochs)
    # -------------------------------------------------------------------------
    p1_epochs = 5
    print("Phase 1: Meta-Training Error Twin to Update ALL LAYERS on 60% Data (36,000 Images)...")

    std_model = StandardDeepModel(d_in, d_hidden, d_out).to(device)
    opt_std = torch.optim.AdamW(std_model.parameters(), lr=1e-3, weight_decay=1e-4)

    twin_sys = FullWeightMetaTwinSystem(d_in, d_hidden, d_out).to(device)
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

            # 2. Meta-Train Twin System on Adapted Multi-Layer Weights:
            # a. Compute deltas across ALL layers on support batch
            w_base = (twin_sys.w1, twin_sys.b1, twin_sys.w2, twin_sys.b2, twin_sys.w3, twin_sys.b3)
            deltas, _ = twin_sys.compute_deltas(b_supp_x, b_supp_y, *w_base, T_fwd=4, T_err=4)

            # b. Inner Loop: Adapt ALL layers simultaneously
            w_adapted = twin_sys.meta_adapt_weights(*deltas)

            # c. Outer Loop: Evaluate on query batch using the adapted weights
            logits_q, _ = twin_sys.forward_pass(b_query_x, *w_adapted, T=4)
            loss_meta = F.cross_entropy(logits_q, b_query_y)

            opt_twin.zero_grad()
            loss_meta.backward()
            nn.utils.clip_grad_norm_(twin_sys.parameters(), 1.0)
            opt_twin.step()

        acc_std = evaluate_std(std_model, loader_test)
        w_base = (twin_sys.w1, twin_sys.b1, twin_sys.w2, twin_sys.b2, twin_sys.w3, twin_sys.b3)
        acc_twin = evaluate_twin(twin_sys, w_base, loader_test)
        print(f"  [Phase 1 Meta-Train] Epoch {ep:>2}/{p1_epochs} | Standard Acc: {acc_std:.2f}% | Twin Base Acc: {acc_twin:.2f}%")

    print(f"Phase 1 Meta-Training completed in {time.time() - t0:.1f}s\n")

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
    sgd_online_model = StandardDeepModel(d_in, d_hidden, d_out).to(device)
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
                deltas, _ = twin_sys.compute_deltas(bx, by, *w_tuple, T_fwd=4, T_err=4)
                dw1, db1, dw2, db2, dw3, db3 = deltas

                # Apply weight updates to EVERY layer in the deep network with slight weight decay
                cur_w1 = (1.0 - 1e-4) * cur_w1 + lr1 * dw1
                cur_b1 = cur_b1 + lr1 * db1
                cur_w2 = (1.0 - 1e-4) * cur_w2 + lr2 * dw2
                cur_b2 = cur_b2 + lr2 * db2
                cur_w3 = (1.0 - 1e-4) * cur_w3 + lr3 * dw3
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
    run_full_weight_meta_study.remote()
