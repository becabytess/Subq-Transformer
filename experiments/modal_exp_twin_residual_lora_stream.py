import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy"
    )
)

app = modal.App("subq-twin-residual-lora-stream", image=image)

@app.function(gpu="T4", timeout=900)
def run_residual_lora_stream_study():
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
    print("=" * 125)
    print("  RESIDUAL FAST-WEIGHTS (DYNAMIC LORA) STREAMING BENCHMARK (FASHION-MNIST 60,000 IMAGES)")
    print("  Architecture: Base Weights (W1, W2, W3) are PERMANENTLY FROZEN")
    print("  Zero-Backprop Updates: Accumulated as parallel residual deltas: W_active = W_base + Delta_W")
    print("  Phase 1: Meta-train on 60% stream (36,000 images)")
    print("  Phase 2: Continual streaming on 40% unseen data (24,000 images, 5 Epochs, ZERO BACKPROP)")
    print("  Evaluation: 10,000 Completely Held-Out Real Test Images After Every Single Epoch")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Split
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

    # 3. Residual Fast-Weights Model System
    class ResidualTwinSystem(nn.Module):
        def __init__(self, d_in=784, d_hidden=128, d_out=10):
            super().__init__()
            self.d_in = d_in
            self.d_hidden = d_hidden
            self.d_out = d_out

            # Permanent Base Weights (Frozen in Phase 2)
            self.w1 = nn.Parameter(torch.randn(d_in, d_hidden) * (2.0 / d_in)**0.5)
            self.b1 = nn.Parameter(torch.zeros(d_hidden))
            self.ln1 = nn.LayerNorm(d_hidden)

            self.w2 = nn.Parameter(torch.randn(d_hidden, d_hidden) * (2.0 / d_hidden)**0.5)
            self.b2 = nn.Parameter(torch.zeros(d_hidden))
            self.ln2 = nn.LayerNorm(d_hidden)
            self.fwd_subq = SubQCore(d_model=d_hidden)

            self.w3 = nn.Parameter(torch.randn(d_hidden, d_out) * (2.0 / d_hidden)**0.5)
            self.b3 = nn.Parameter(torch.zeros(d_out))

            # Error Twin
            self.err_subq = SubQCore(d_model=d_hidden)

            # Scaling rates
            self.lr1 = nn.Parameter(torch.tensor(0.005))
            self.lr2 = nn.Parameter(torch.tensor(0.01))
            self.lr3 = nn.Parameter(torch.tensor(0.02))

        def forward_with_residuals(self, x, dw1=None, db1=None, dw2=None, db2=None, dw3=None, db3=None, T=4):
            B = x.shape[0]
            h0 = x.view(B, -1)

            # Active Layer 1 (Base + Residual Delta)
            eff_w1 = self.w1 if dw1 is None else (self.w1 + dw1)
            eff_b1 = self.b1 if db1 is None else (self.b1 + db1)
            z1 = torch.mm(h0, eff_w1) + eff_b1
            h1 = F.gelu(self.ln1(z1))

            # Active Layer 2 (Base + Residual Delta)
            eff_w2 = self.w2 if dw2 is None else (self.w2 + dw2)
            eff_b2 = self.b2 if db2 is None else (self.b2 + db2)
            z2 = torch.mm(h1, eff_w2) + eff_b2
            h2_in = self.ln2(z2).unsqueeze(1)
            h2 = self.fwd_subq(h2_in, T=T).squeeze(1)

            # Active Layer 3 Head (Base + Residual Delta)
            eff_w3 = self.w3 if dw3 is None else (self.w3 + dw3)
            eff_b3 = self.b3 if db3 is None else (self.b3 + db3)
            logits = torch.mm(h2, eff_w3) + eff_b3

            return logits, (h0, h1, h2, eff_w2, eff_w3)

        def compute_residual_step(self, x, y, dw1=None, db1=None, dw2=None, db2=None, dw3=None, db3=None, T=4):
            B = x.shape[0]
            logits, (h0, h1, h2, eff_w2, eff_w3) = self.forward_with_residuals(x, dw1, db1, dw2, db2, dw3, db3, T=T)
            e_out = F.one_hot(y, self.d_out).float() - F.softmax(logits, dim=-1)

            # Residual Head Delta
            step_dw3 = torch.mm(h2.t(), e_out) / (B * math.sqrt(self.d_hidden))
            step_db3 = e_out.mean(dim=0)

            # Residual Layer 2 Delta
            e2_in = torch.mm(e_out, eff_w3.t()).unsqueeze(1)
            e2 = self.err_subq(e2_in, T=T).squeeze(1)
            step_dw2 = torch.mm(h1.t(), e2) / (B * math.sqrt(self.d_hidden))
            step_db2 = e2.mean(dim=0)

            # Residual Layer 1 Delta
            e1 = torch.mm(e2, eff_w2.t())
            step_dw1 = torch.mm(h0.t(), e1) / (B * math.sqrt(self.d_in))
            step_db1 = e1.mean(dim=0)

            return (step_dw1, step_db1, step_dw2, step_db2, step_dw3, step_db3), logits

    # 4. Standard Supervised Model Baseline
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

    def evaluate_twin(twin_sys, residuals, loader):
        twin_sys.eval()
        dw1, db1, dw2, db2, dw3, db3 = residuals
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                logits, _ = twin_sys.forward_with_residuals(bx, dw1, db1, dw2, db2, dw3, db3, T=4)
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
    # Phase 1: Train Base Weights & Error Twin on 60% Data (5 Epochs)
    # -------------------------------------------------------------------------
    p1_epochs = 5
    print("Phase 1: Meta-Training Base Models on 60% Data (36,000 Images)...")

    std_model = StandardDeepModel(d_in, d_hidden, d_out).to(device)
    opt_std = torch.optim.AdamW(std_model.parameters(), lr=1e-3, weight_decay=1e-4)

    twin_sys = ResidualTwinSystem(d_in, d_hidden, d_out).to(device)
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

            # 2. Train Twin Model
            # Support forward
            step_deltas, _ = twin_sys.compute_residual_step(b_supp_x, b_supp_y, T=4)
            sdw1, sdb1, sdw2, sdb2, sdw3, sdb3 = step_deltas

            dw1 = twin_sys.lr1 * sdw1
            db1 = twin_sys.lr1 * sdb1
            dw2 = twin_sys.lr2 * sdw2
            db2 = twin_sys.lr2 * sdb2
            dw3 = twin_sys.lr3 * sdw3
            db3 = twin_sys.lr3 * sdb3

            # Query forward with residual deltas
            logits_q, _ = twin_sys.forward_with_residuals(b_query_x, dw1, db1, dw2, db2, dw3, db3, T=4)
            loss_meta = F.cross_entropy(logits_q, b_query_y)

            opt_twin.zero_grad()
            loss_meta.backward()
            nn.utils.clip_grad_norm_(twin_sys.parameters(), 1.0)
            opt_twin.step()

        acc_std = evaluate_std(std_model, loader_test)
        acc_twin_base = evaluate_twin(twin_sys, (None, None, None, None, None, None), loader_test)
        print(f"  [Phase 1] Epoch {ep:>2}/{p1_epochs} | Standard Test Acc: {acc_std:.2f}% | Twin Base Acc: {acc_twin_base:.2f}%")

    print(f"Phase 1 completed in {time.time() - t0:.1f}s\n")

    # -------------------------------------------------------------------------
    # Phase 2: RESIDUAL FAST-WEIGHT STREAMING ON 40% DATA (24,000 IMAGES, 5 EPOCHS)
    # -------------------------------------------------------------------------
    print("=" * 125)
    print("  PHASE 2: 4-WAY COMPARISON ON 40% UNSEEN DATA (24,000 IMAGES, 5 EPOCHS)")
    print("  Base Weights (W1, W2, W3) Are 100% Frozen. Testing Residual Dynamic LoRA Fast-Weights.")
    print("=" * 125)
    print(f"{'Epoch on 40% Stream':<20} | {'Static Base':<12} | {'Online SGD (Backprop)':<22} | {'Head-Only Twin (Zero BP)':<25} | {'Residual Multi-Layer Twin (Zero BP)':<35}")
    print("-" * 125)

    base_static_acc = evaluate_std(std_model, loader_test)

    # 1. Online SGD Model
    sgd_model = StandardDeepModel(d_in, d_hidden, d_out).to(device)
    sgd_model.load_state_dict(std_model.state_dict())
    opt_sgd = torch.optim.SGD(sgd_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # 2. Freeze Base Twin System
    twin_sys.eval()
    init_twin_acc = evaluate_twin(twin_sys, (None, None, None, None, None, None), loader_test)

    # Residual Fast-Weight State for Head-Only
    head_dw3 = torch.zeros_like(twin_sys.w3)
    head_db3 = torch.zeros_like(twin_sys.b3)

    # Residual Fast-Weight State for Multi-Layer
    res_dw1 = torch.zeros_like(twin_sys.w1)
    res_db1 = torch.zeros_like(twin_sys.b1)
    res_dw2 = torch.zeros_like(twin_sys.w2)
    res_db2 = torch.zeros_like(twin_sys.b2)
    res_dw3 = torch.zeros_like(twin_sys.w3)
    res_db3 = torch.zeros_like(twin_sys.b3)

    lr1 = twin_sys.lr1.item()
    lr2 = twin_sys.lr2.item()
    lr3 = twin_sys.lr3.item()

    print(f"Epoch 0 (Pre-Stream) | {base_static_acc:>10.2f}% | {base_static_acc:>20.2f}% | {init_twin_acc:>23.2f}% | {init_twin_acc:>33.2f}%")

    p2_epochs = 5
    for ep in range(1, p2_epochs + 1):
        # A. Online Backprop SGD Update
        sgd_model.train()
        for bx, by in loader_p2:
            bx, by = bx.to(device), by.to(device)
            logits = sgd_model(bx, T=4)
            loss = F.cross_entropy(logits, by)
            opt_sgd.zero_grad()
            loss.backward()
            opt_sgd.step()

        # B. Head-Only Residual Twin Update (Zero Backprop)
        with torch.no_grad():
            for bx, by in loader_p2:
                bx, by = bx.to(device), by.to(device)
                logits_h, (_, _, h2, _, _) = twin_sys.forward_with_residuals(bx, dw3=head_dw3, db3=head_db3, T=4)
                e_out = F.one_hot(by, 10).float() - F.softmax(logits_h, dim=-1)
                B = bx.shape[0]
                step_dw3 = torch.mm(h2.t(), e_out) / (B * math.sqrt(d_hidden))
                step_db3 = e_out.mean(dim=0)
                head_dw3 = (1.0 - 1e-4) * head_dw3 + lr3 * step_dw3
                head_db3 = head_db3 + lr3 * step_db3

        # C. Residual Multi-Layer Twin Update (Zero Backprop: Base Frozen, Residuals Evolve)
        with torch.no_grad():
            for bx, by in loader_p2:
                bx, by = bx.to(device), by.to(device)
                res_tuple = (res_dw1, res_db1, res_dw2, res_db2, res_dw3, res_db3)
                step_deltas, _ = twin_sys.compute_residual_step(bx, by, *res_tuple, T=4)
                sdw1, sdb1, sdw2, sdb2, sdw3, sdb3 = step_deltas

                # Accumulate into residual branch with weight decay regularizer
                res_dw1 = (1.0 - 1e-4) * res_dw1 + lr1 * sdw1
                res_db1 = res_db1 + lr1 * sdb1
                res_dw2 = (1.0 - 1e-4) * res_dw2 + lr2 * sdw2
                res_db2 = res_db2 + lr2 * sdb2
                res_dw3 = (1.0 - 1e-4) * res_dw3 + lr3 * sdw3
                res_db3 = res_db3 + lr3 * sdb3

        # Evaluate on 10,000 held-out test images
        acc_sgd = evaluate_std(sgd_model, loader_test)
        acc_head = evaluate_twin(twin_sys, (None, None, None, None, head_dw3, head_db3), loader_test)
        acc_res_multi = evaluate_twin(twin_sys, (res_dw1, res_db1, res_dw2, res_db2, res_dw3, res_db3), loader_test)

        delta_sgd = acc_sgd - base_static_acc
        delta_head = acc_head - init_twin_acc
        delta_res = acc_res_multi - init_twin_acc

        print(f"Epoch {ep:>2}/5 on 40% Stream | {base_static_acc:>10.2f}% | {acc_sgd:>15.2f}% ({delta_sgd:+.2f}%) | {acc_head:>18.2f}% ({delta_head:+.2f}%) | {acc_res_multi:>28.2f}% ({delta_res:+.2f}%)")

    print("=" * 125)

@app.local_entrypoint()
def main():
    run_residual_lora_stream_study.remote()
