import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy"
    )
)

app = modal.App("subq-twin-continuous-no-reset", image=image)

@app.function(gpu="T4", timeout=900)
def run_continuous_streaming_study():
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
    print("  PURE CONTINUOUS NO-RESET STREAMING BENCHMARK (REAL FASHION-MNIST 60,000 IMAGES)")
    print("  Protocol: 100% Continuous Online Training (NO RESETS EVER)")
    print("  Phase 1: Stream 36,000 images (60%) -> Weights evolve continuously, Twin learns online dynamics")
    print("  Phase 2: Stream 24,000 images (40%, 5 Epochs) -> ZERO BACKPROP online updates across all layers")
    print("  Evaluation: 10,000 Completely Held-Out Real Test Images After Every Single Epoch")
    print("=" * 115)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Setup
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

    # 3. Continuous Streaming Model + Error Twin System
    class ContinuousTwinSystem(nn.Module):
        def __init__(self, d_in=784, d_hidden=128, d_out=10):
            super().__init__()
            self.d_in = d_in
            self.d_hidden = d_hidden
            self.d_out = d_out

            # Layer 1
            self.w1 = nn.Parameter(torch.randn(d_in, d_hidden) * (2.0 / d_in)**0.5)
            self.b1 = nn.Parameter(torch.zeros(d_hidden))
            self.ln1 = nn.LayerNorm(d_hidden)

            # Layer 2
            self.w2 = nn.Parameter(torch.randn(d_hidden, d_hidden) * (2.0 / d_hidden)**0.5)
            self.b2 = nn.Parameter(torch.zeros(d_hidden))
            self.ln2 = nn.LayerNorm(d_hidden)
            self.fwd_subq = SubQCore(d_model=d_hidden)

            # Layer 3 (Head)
            self.w3 = nn.Parameter(torch.randn(d_hidden, d_out) * (2.0 / d_hidden)**0.5)
            self.b3 = nn.Parameter(torch.zeros(d_out))

            # Error Twin
            self.err_subq = SubQCore(d_model=d_hidden)

            # Learnable layer rates
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

        def compute_step_deltas(self, x, y, w1, b1, w2, b2, w3, b3, T=4):
            B = x.shape[0]
            logits, (h0, h1, h2) = self.forward_with_weights(x, w1, b1, w2, b2, w3, b3, T=T)
            e_out = F.one_hot(y, self.d_out).float() - F.softmax(logits, dim=-1)

            # Head Delta
            dw3 = torch.mm(h2.t(), e_out) / (B * math.sqrt(self.d_hidden))
            db3 = e_out.mean(dim=0)

            # Layer 2 Error & Delta
            e2_in = torch.mm(e_out, w3.t()).unsqueeze(1)
            e2 = self.err_subq(e2_in, T=T).squeeze(1)
            dw2 = torch.mm(h1.t(), e2) / (B * math.sqrt(self.d_hidden))
            db2 = e2.mean(dim=0)

            # Layer 1 Error & Delta
            e1 = torch.mm(e2, w2.t())
            dw1 = torch.mm(h0.t(), e1) / (B * math.sqrt(self.d_in))
            db1 = e1.mean(dim=0)

            return (dw1, db1, dw2, db2, dw3, db3), logits

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
    # Phase 1: Pure Streaming Online Training on 60% Data (NO RESETS)
    # -------------------------------------------------------------------------
    p1_epochs = 5
    print("Phase 1: Pure Continuous Online Training on 60% Data (36,000 Images, NO RESETS)...")

    std_model = StandardDeepModel(d_in, d_hidden, d_out).to(device)
    opt_std = torch.optim.AdamW(std_model.parameters(), lr=1e-3, weight_decay=1e-4)

    twin_sys = ContinuousTwinSystem(d_in, d_hidden, d_out).to(device)
    opt_twin = torch.optim.AdamW(twin_sys.parameters(), lr=1e-3, weight_decay=1e-4)

    # Initial continuous weights (will evolve continuously across ALL batches, never reset)
    cur_w1 = twin_sys.w1.clone()
    cur_b1 = twin_sys.b1.clone()
    cur_w2 = twin_sys.w2.clone()
    cur_b2 = twin_sys.b2.clone()
    cur_w3 = twin_sys.w3.clone()
    cur_b3 = twin_sys.b3.clone()

    t0 = time.time()
    K_chunk = 4 # 4-step unrolled streaming chunk

    for ep in range(1, p1_epochs + 1):
        std_model.train()
        twin_sys.train()

        p1_batches = list(loader_p1)
        for i in range(0, len(p1_batches) - K_chunk, K_chunk):
            # 1. Standard Model updates
            for k in range(K_chunk):
                bx, by = p1_batches[i+k][0].to(device), p1_batches[i+k][1].to(device)
                logits_std = std_model(bx, T=4)
                loss_std = F.cross_entropy(logits_std, by)
                opt_std.zero_grad()
                loss_std.backward()
                opt_std.step()

            # 2. Continuous Online Twin Evolution over K streaming batches (NO RESET)
            step_w1, step_b1 = cur_w1, cur_b1
            step_w2, step_b2 = cur_w2, cur_b2
            step_w3, step_b3 = cur_w3, cur_b3

            total_stream_loss = 0.0
            for k in range(K_chunk):
                bx, by = p1_batches[i+k][0].to(device), p1_batches[i+k][1].to(device)
                w_tuple = (step_w1, step_b1, step_w2, step_b2, step_w3, step_b3)
                deltas, logits_step = twin_sys.compute_step_deltas(bx, by, *w_tuple, T=4)
                dw1, db1, dw2, db2, dw3, db3 = deltas

                total_stream_loss = total_stream_loss + F.cross_entropy(logits_step, by)

                # Continuous forward weight update for next streaming batch
                step_w1 = (1.0 - 1e-4) * step_w1 + twin_sys.lr1 * dw1
                step_b1 = step_b1 + twin_sys.lr1 * db1
                step_w2 = (1.0 - 1e-4) * step_w2 + twin_sys.lr2 * dw2
                step_b2 = step_b2 + twin_sys.lr2 * db2
                step_w3 = (1.0 - 1e-4) * step_w3 + twin_sys.lr3 * dw3
                step_b3 = step_b3 + twin_sys.lr3 * db3

            # Update Error Twin parameters to optimize streaming dynamics
            opt_twin.zero_grad()
            total_stream_loss.backward()
            nn.utils.clip_grad_norm_(twin_sys.parameters(), 1.0)
            opt_twin.step()

            # PERSIST WEIGHTS INTO NEXT STREAM CHUNK (NO RESET!)
            cur_w1 = step_w1.detach()
            cur_b1 = step_b1.detach()
            cur_w2 = step_w2.detach()
            cur_b2 = step_b2.detach()
            cur_w3 = step_w3.detach()
            cur_b3 = step_b3.detach()

        acc_std = evaluate_std(std_model, loader_test)
        w_evolved = (cur_w1, cur_b1, cur_w2, cur_b2, cur_w3, cur_b3)
        acc_twin = evaluate_twin(twin_sys, w_evolved, loader_test)
        print(f"  [Phase 1 Continuous Stream] Epoch {ep:>2}/{p1_epochs} | Standard Test Acc: {acc_std:.2f}% | Twin Evolved Acc: {acc_twin:.2f}%")

    print(f"Phase 1 completed in {time.time() - t0:.1f}s\n")

    # -------------------------------------------------------------------------
    # Phase 2: CONTINUAL STREAMING LEARNING ON 40% DATA (24,000 IMAGES, ZERO BACKPROP, NO RESETS)
    # -------------------------------------------------------------------------
    print("=" * 125)
    print("  PHASE 2: CONTINUAL MULTI-EPOCH LEARNING ON 40% UNSEEN DATA (24,000 IMAGES, 5 EPOCHS)")
    print("  Updating Continuous Stream with ZERO BACKPROP and NO RESETS")
    print("=" * 125)
    print(f"{'Epoch on 40% Stream':<20} | {'Static Base':<12} | {'Online SGD (Backprop)':<22} | {'Head-Only Twin (Zero BP)':<25} | {'Full-Weight Twin (Zero BP)':<28}")
    print("-" * 125)

    base_static_acc = evaluate_std(std_model, loader_test)

    # 1. Online SGD Model
    sgd_model = StandardDeepModel(d_in, d_hidden, d_out).to(device)
    sgd_model.load_state_dict(std_model.state_dict())
    opt_sgd = torch.optim.SGD(sgd_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # 2. Head-Only Twin (Continues evolving head from Phase 1 weights)
    twin_sys.eval()
    head_w1 = cur_w1.clone()
    head_b1 = cur_b1.clone()
    head_w2 = cur_w2.clone()
    head_b2 = cur_b2.clone()
    head_w3 = cur_w3.clone()
    head_b3 = cur_b3.clone()

    # 3. Full-Weight Twin (Continues evolving all weights from Phase 1 weights)
    full_w1 = cur_w1.clone()
    full_b1 = cur_b1.clone()
    full_w2 = cur_w2.clone()
    full_b2 = cur_b2.clone()
    full_w3 = cur_w3.clone()
    full_b3 = cur_b3.clone()

    lr1 = twin_sys.lr1.item()
    lr2 = twin_sys.lr2.item()
    lr3 = twin_sys.lr3.item()

    w_init_eval = (cur_w1, cur_b1, cur_w2, cur_b2, cur_w3, cur_b3)
    init_twin_acc = evaluate_twin(twin_sys, w_init_eval, loader_test)

    print(f"Epoch 0 (Pre-Stream) | {base_static_acc:>10.2f}% | {base_static_acc:>20.2f}% | {init_twin_acc:>23.2f}% | {init_twin_acc:>26.2f}%")

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

        # B. Head-Only Twin Update (Zero Backprop, Continuous Stream)
        with torch.no_grad():
            for bx, by in loader_p2:
                bx, by = bx.to(device), by.to(device)
                w_head_tuple = (head_w1, head_b1, head_w2, head_b2, head_w3, head_b3)
                logits_h, (_, _, h2) = twin_sys.forward_with_weights(bx, *w_head_tuple, T=4)
                e_out = F.one_hot(by, 10).float() - F.softmax(logits_h, dim=-1)
                B = bx.shape[0]
                dw3 = torch.mm(h2.t(), e_out) / (B * math.sqrt(d_hidden))
                db3 = e_out.mean(dim=0)
                head_w3 = (1.0 - 1e-4) * head_w3 + lr3 * dw3
                head_b3 = head_b3 + lr3 * db3

        # C. Full-Weight Twin Update (Zero Backprop, Continuous Stream on ALL layers)
        with torch.no_grad():
            for bx, by in loader_p2:
                bx, by = bx.to(device), by.to(device)
                w_full_tuple = (full_w1, full_b1, full_w2, full_b2, full_w3, full_b3)
                deltas, _ = twin_sys.compute_step_deltas(bx, by, *w_full_tuple, T=4)
                dw1, db1, dw2, db2, dw3, db3 = deltas

                # Continuous update for ALL layers
                full_w1 = (1.0 - 1e-4) * full_w1 + lr1 * dw1
                full_b1 = full_b1 + lr1 * db1
                full_w2 = (1.0 - 1e-4) * full_w2 + lr2 * dw2
                full_b2 = full_b2 + lr2 * db2
                full_w3 = (1.0 - 1e-4) * full_w3 + lr3 * dw3
                full_b3 = full_b3 + lr3 * db3

        # Evaluate on 10,000 held-out test images
        acc_sgd = evaluate_std(sgd_model, loader_test)
        w_head_eval = (head_w1, head_b1, head_w2, head_b2, head_w3, head_b3)
        acc_head = evaluate_twin(twin_sys, w_head_eval, loader_test)
        w_full_eval = (full_w1, full_b1, full_w2, full_b2, full_w3, full_b3)
        acc_full = evaluate_twin(twin_sys, w_full_eval, loader_test)

        delta_sgd = acc_sgd - base_static_acc
        delta_head = acc_head - init_twin_acc
        delta_full = acc_full - init_twin_acc

        print(f"Epoch {ep:>2}/5 on 40% Stream | {base_static_acc:>10.2f}% | {acc_sgd:>15.2f}% ({delta_sgd:+.2f}%) | {acc_head:>18.2f}% ({delta_head:+.2f}%) | {acc_full:>21.2f}% ({delta_full:+.2f}%)")

    print("=" * 125)

@app.local_entrypoint()
def main():
    run_continuous_streaming_study.remote()
