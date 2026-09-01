import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy",
        "requests",
        "pillow"
    )
)

app = modal.App("subq-hebbian-meta-omniglot", image=image)

@app.function(gpu="T4", timeout=900)
def run_hebbian_meta_study():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.datasets import Omniglot
    from torchvision import transforms
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 95)
    print("  STUDY: BILINEAR HEBBIAN SUBQ META-OPTIMIZER (5-WAY OMNIGLOT RECOGNITION)")
    print("=" * 95)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Setup (Omniglot)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    print("Loading Omniglot dataset...")
    train_dataset = Omniglot(root="./data", background=True, download=True, transform=transform)
    test_dataset = Omniglot(root="./data", background=False, download=True, transform=transform)

    def build_class_dict(dataset):
        class_dict = {}
        for img, label in dataset:
            if label not in class_dict:
                class_dict[label] = []
            class_dict[label].append(img)
        return {k: torch.stack(v) for k, v in class_dict.items()}

    train_classes = build_class_dict(train_dataset)
    test_classes = build_class_dict(test_dataset)

    print(f"Dataset Ready: {len(train_classes)} Meta-Train Classes | {len(test_classes)} Held-Out Meta-Test Classes")

    # 2. Episode Sampler
    def sample_episode(classes_dict, n_way=5, k_shot=5, n_query=5):
        all_cls = list(classes_dict.keys())
        chosen_cls = np.random.choice(all_cls, n_way, replace=False)

        support_imgs, support_lbls = [], []
        query_imgs, query_lbls = [], []

        for new_lbl, c in enumerate(chosen_cls):
            imgs = classes_dict[c]
            perm = torch.randperm(len(imgs))
            supp_idx = perm[:k_shot]
            query_idx = perm[k_shot : k_shot + n_query]

            support_imgs.append(imgs[supp_idx])
            support_lbls.append(torch.full((k_shot,), new_lbl, dtype=torch.long))

            query_imgs.append(imgs[query_idx])
            query_lbls.append(torch.full((n_query,), new_lbl, dtype=torch.long))

        support_imgs = torch.cat(support_imgs, dim=0)
        support_lbls = torch.cat(support_lbls, dim=0)
        query_imgs = torch.cat(query_imgs, dim=0)
        query_lbls = torch.cat(query_lbls, dim=0)

        perm_s = torch.randperm(len(support_imgs))
        return (
            support_imgs[perm_s].to(device),
            support_lbls[perm_s].to(device),
            query_imgs.to(device),
            query_lbls.to(device)
        )

    # 3. Vision Backbone (Conv4)
    class Conv4Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
            )

        def forward(self, x):
            return self.encoder(x).squeeze(-1).squeeze(-1) # [N, 64]

    d_feat = 64
    n_classes = 5

    # -------------------------------------------------------------------------
    # 4. Bilinear Hebbian SubQ Meta-Optimizer
    # -------------------------------------------------------------------------
    class BilinearSubQMetaOptimizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Conv4Backbone()

            # Base initial classifier weights
            self.w_init = nn.Parameter(torch.zeros(d_feat, n_classes))
            self.b_init = nn.Parameter(torch.zeros(n_classes))
            nn.init.xavier_uniform_(self.w_init)

            # Hebbian modulation scales
            self.eta = nn.Parameter(torch.tensor(0.5))
            self.decay = nn.Parameter(torch.tensor(0.1))

            # SubQ GRU-style coordinate gates for weight matrix W
            self.w_gate_error = nn.Linear(n_classes, 2 * n_classes, bias=True)
            self.w_gate_feat = nn.Linear(d_feat, 2 * d_feat, bias=True)
            self.cand_weight = nn.Parameter(torch.tensor(1.0))

        def adapt_and_evaluate(self, supp_x, supp_y, query_x):
            supp_feats = self.backbone(supp_x) # [K, 64]
            K = len(supp_feats)
            
            W = self.w_init.clone()
            b = self.b_init.clone()

            # Sequential Hebbian-SubQ adaptation
            for t in range(K):
                ft = supp_feats[t : t + 1] # [1, 64]
                yt = supp_y[t : t + 1]     # [1]
                yt_onehot = F.one_hot(yt, num_classes=n_classes).float() # [1, 5]

                # 1. Forward prediction
                logits_t = torch.mm(ft, W) + b # [1, 5]
                prob_t = F.softmax(logits_t, dim=-1) # [1, 5]
                error_t = yt_onehot - prob_t         # [1, 5] (Gradient direction: target - prediction)

                # 2. Bilinear Outer Product: Feature (64) x Error (5)
                # This explicitly binds the visual feature channels to class directions!
                hebbian_delta = torch.mm(ft.t(), error_t) # [64, 5]

                # 3. SubQ Coordinate Gating
                # Gate signals conditioned on feature activation and error magnitude
                gates_feat = self.w_gate_feat(ft) # [1, 128]
                gates_err = self.w_gate_error(error_t) # [1, 10]
                
                rf, zf = gates_feat.chunk(2, dim=-1) # [1, 64], [1, 64]
                re, ze = gates_err.chunk(2, dim=-1)   # [1, 5], [1, 5]

                r_gate = torch.sigmoid(torch.mm(rf.t(), re)) # [64, 5]
                z_gate = torch.sigmoid(torch.mm(zf.t(), ze)) # [64, 5]

                # Update candidate: Hebbian correlation modulated by reset gate
                cand = torch.tanh(self.cand_weight * hebbian_delta + r_gate * W)

                # SubQ Contraction Step: Convex combination of persistent memory and candidate
                W = (1.0 - z_gate) * (W + self.eta * hebbian_delta) + z_gate * cand
                b = b + self.eta * error_t.squeeze(0)

            query_feats = self.backbone(query_x)
            query_logits = torch.mm(query_feats, W) + b
            return query_logits

    # -------------------------------------------------------------------------
    # 5. MAML Baseline (Inner-Loop SGD on Head Parameters)
    # -------------------------------------------------------------------------
    class MAMLClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Conv4Backbone()
            self.w_init = nn.Parameter(torch.zeros(d_feat, n_classes))
            self.b_init = nn.Parameter(torch.zeros(n_classes))
            nn.init.xavier_uniform_(self.w_init)
            self.inner_lr = nn.Parameter(torch.tensor(0.1))

        def adapt_and_evaluate(self, supp_x, supp_y, query_x):
            supp_feats = self.backbone(supp_x)
            K = len(supp_feats)
            
            W = self.w_init.clone()
            b = self.b_init.clone()

            for t in range(K):
                ft = supp_feats[t : t + 1]
                yt = supp_y[t : t + 1]
                logits_t = torch.mm(ft, W) + b
                loss_t = F.cross_entropy(logits_t, yt)
                grad_W, grad_b = torch.autograd.grad(loss_t, [W, b], create_graph=True)
                W = W - self.inner_lr * grad_W
                b = b - self.inner_lr * grad_b

            query_feats = self.backbone(query_x)
            query_logits = torch.mm(query_feats, W) + b
            return query_logits

    # -------------------------------------------------------------------------
    # 6. Meta-Training Loop (800 Episodes)
    # -------------------------------------------------------------------------
    total_episodes = 800
    n_way = 5
    k_train = 5
    n_query = 5

    print(f"\nMeta-Training: 5-Way 5-Shot Episodes on Omniglot ({total_episodes} Episodes)...")

    # 1. Train Bilinear SubQ Meta-Optimizer
    print("\n[1/2] Training Bilinear Hebbian SubQ Meta-Optimizer...")
    subq_net = BilinearSubQMetaOptimizer().to(device)
    opt_subq = torch.optim.AdamW(subq_net.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for ep in range(1, total_episodes + 1):
        subq_net.train()
        s_x, s_y, q_x, q_y = sample_episode(train_classes, n_way=n_way, k_shot=k_train, n_query=n_query)

        q_logits = subq_net.adapt_and_evaluate(s_x, s_y, q_x)
        loss = F.cross_entropy(q_logits, q_y)
        acc = (q_logits.argmax(dim=-1) == q_y).float().mean().item() * 100.0

        opt_subq.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(subq_net.parameters(), 1.0)
        opt_subq.step()

        if ep % 200 == 0 or ep == total_episodes:
            print(f"  Episode {ep:>4}/{total_episodes} | Query Loss: {loss.item():.4f} | Top-1 Accuracy: {acc:.1f}%")
    print(f"Bilinear SubQ training completed in {time.time() - t0:.1f}s")

    # 2. Train MAML Baseline
    print("\n[2/2] Training MAML (Inner SGD) Baseline...")
    maml_net = MAMLClassifier().to(device)
    opt_maml = torch.optim.AdamW(maml_net.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for ep in range(1, total_episodes + 1):
        maml_net.train()
        s_x, s_y, q_x, q_y = sample_episode(train_classes, n_way=n_way, k_shot=k_train, n_query=n_query)

        q_logits = maml_net.adapt_and_evaluate(s_x, s_y, q_x)
        loss = F.cross_entropy(q_logits, q_y)
        acc = (q_logits.argmax(dim=-1) == q_y).float().mean().item() * 100.0

        opt_maml.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(maml_net.parameters(), 1.0)
        opt_maml.step()

        if ep % 200 == 0 or ep == total_episodes:
            print(f"  Episode {ep:>4}/{total_episodes} | Query Loss: {loss.item():.4f} | Top-1 Accuracy: {acc:.1f}%")
    print(f"MAML Baseline training completed in {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------------
    # 7. Evaluation on Held-Out Test Classes (100 Unseen Episodes)
    # -------------------------------------------------------------------------
    subq_net.eval()
    maml_net.eval()

    test_shots = [1, 2, 3, 5, 10]
    n_eval_episodes = 100

    print("\n" + "=" * 95)
    print("  FINAL EVALUATION ON COMPLETELY UNSEEN HELD-OUT OMNIGLOT TEST ALPHABETS (100 EPISODES)")
    print("=" * 95)
    print(f"{'Shot per Class (K)':<20} | {'Total Examples (T)':<20} | {'MAML (SGD) Top-1 Acc':<22} | {'Bilinear SubQ Top-1 Acc':<28} | {'Winner'}")
    print("-" * 110)

    for k in test_shots:
        total_T = k * n_way
        maml_accs = []
        subq_accs = []

        for _ in range(n_eval_episodes):
            s_x, s_y, q_x, q_y = sample_episode(test_classes, n_way=n_way, k_shot=k, n_query=n_query)

            # 1. Bilinear SubQ (Forward Only, No Inner Backprop)
            with torch.no_grad():
                q_logits_subq = subq_net.adapt_and_evaluate(s_x, s_y, q_x)
                subq_accs.append((q_logits_subq.argmax(dim=-1) == q_y).float().mean().item() * 100.0)

            # 2. MAML (requires inner backprop)
            with torch.enable_grad():
                q_logits_maml = maml_net.adapt_and_evaluate(s_x, s_y, q_x)
                maml_accs.append((q_logits_maml.argmax(dim=-1) == q_y).float().mean().item() * 100.0)

        mean_maml = np.mean(maml_accs)
        mean_subq = np.mean(subq_accs)
        winner = "🏆 Bilinear SubQ" if mean_subq > mean_maml else "— MAML"
        delta = mean_subq - mean_maml
        print(f"K = {k:<16} | T = {total_T:<16} | {mean_maml:>18.2f}% | {mean_subq:>24.2f}% ({delta:>+5.2f}%) | {winner}")

    print("=" * 110)

@app.local_entrypoint()
def main():
    run_hebbian_meta_study.remote()
