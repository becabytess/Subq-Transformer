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

app = modal.App("subq-meta-optimizer-omniglot", image=image)

@app.function(gpu="T4", timeout=900)
def run_omniglot_meta_study():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.datasets import Omniglot
    from torchvision import transforms
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90)
    print("  STUDY: SUBQ AS A DYNAMICAL META-OPTIMIZER ON REAL VISION TASKS (OMNIGLOT 5-WAY)")
    print("=" * 90)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------------------
    # 1. Dataset Setup: Omniglot (Background + Evaluation sets)
    # -------------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    print("Downloading / Loading Omniglot dataset...")
    train_dataset = Omniglot(root="./data", background=True, download=True, transform=transform)
    test_dataset = Omniglot(root="./data", background=False, download=True, transform=transform)

    # Group images by character class
    def build_class_dict(dataset):
        class_dict = {}
        for img, label in dataset:
            if label not in class_dict:
                class_dict[label] = []
            class_dict[label].append(img)
        return {k: torch.stack(v) for k, v in class_dict.items()}

    train_classes = build_class_dict(train_dataset)
    test_classes = build_class_dict(test_dataset)

    n_train_cls = len(train_classes)
    n_test_cls = len(test_classes)
    print(f"Loaded Omniglot: {n_train_cls} Meta-Train Classes | {n_test_cls} Held-Out Meta-Test Classes")

    # -------------------------------------------------------------------------
    # 2. Episode Sampler: 5-Way K-Shot Tasks
    # -------------------------------------------------------------------------
    def sample_episode(classes_dict, n_way=5, k_shot=5, n_query=5):
        # Pick n_way random classes
        all_cls = list(classes_dict.keys())
        chosen_cls = np.random.choice(all_cls, n_way, replace=False)

        support_imgs, support_lbls = [], []
        query_imgs, query_lbls = [], []

        for new_lbl, c in enumerate(chosen_cls):
            imgs = classes_dict[c] # [20, 1, 28, 28]
            perm = torch.randperm(len(imgs))
            supp_idx = perm[:k_shot]
            query_idx = perm[k_shot : k_shot + n_query]

            support_imgs.append(imgs[supp_idx])
            support_lbls.append(torch.full((k_shot,), new_lbl, dtype=torch.long))

            query_imgs.append(imgs[query_idx])
            query_lbls.append(torch.full((n_query,), new_lbl, dtype=torch.long))

        support_imgs = torch.cat(support_imgs, dim=0) # [N_supp, 1, 28, 28]
        support_lbls = torch.cat(support_lbls, dim=0) # [N_supp]
        query_imgs = torch.cat(query_imgs, dim=0)     # [N_query, 1, 28, 28]
        query_lbls = torch.cat(query_lbls, dim=0)     # [N_query]

        # Randomize order of support stream
        perm_s = torch.randperm(len(support_imgs))
        support_imgs = support_imgs[perm_s].to(device)
        support_lbls = support_lbls[perm_s].to(device)

        query_imgs = query_imgs.to(device)
        query_lbls = query_lbls.to(device)
        return support_imgs, support_lbls, query_imgs, query_lbls

    # -------------------------------------------------------------------------
    # 3. Vision Backbone (Standard Conv4)
    # -------------------------------------------------------------------------
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
            # x: [B, 1, 28, 28] -> [B, 64]
            return self.encoder(x).squeeze(-1).squeeze(-1)

    # -------------------------------------------------------------------------
    # 4. Dynamic Linear Classifier Head (W in R^{64 x 5}, b in R^5 -> 325 params)
    # -------------------------------------------------------------------------
    d_feat = 64
    n_classes = 5
    d_param = d_feat * n_classes + n_classes # 320 + 5 = 325

    def forward_classifier(feats, params):
        # feats: [N, 64], params: [325]
        W = params[:320].view(d_feat, n_classes) # [64, 5]
        b = params[320:].view(1, n_classes)      # [1, 5]
        logits = torch.mm(feats, W) + b          # [N, 5]
        return logits

    # -------------------------------------------------------------------------
    # 5. SubQ Vision Meta-Optimizer
    # -------------------------------------------------------------------------
    class SubQVisionMetaOptimizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Conv4Backbone()
            self.init_params = nn.Parameter(torch.zeros(d_param))
            nn.init.normal_(self.init_params, std=0.05)

            # Error + Context encoder
            # Inputs at step t: feat (64) + one_hot label (5) + error (5) -> 74 dims
            self.step_encoder = nn.Sequential(
                nn.Linear(d_feat + n_classes + n_classes, 256),
                nn.GELU(),
                nn.Linear(256, d_param)
            )

            # SubQ GRU contraction gates on the 325-dim parameter manifold
            self.w_gate_h = nn.Linear(d_param, 2 * d_param, bias=False)
            self.w_cand_h = nn.Linear(d_param, d_param, bias=False)
            self.w_ih = nn.Linear(d_param, 3 * d_param, bias=False)

        def adapt_and_evaluate(self, supp_x, supp_y, query_x):
            # Extract support features
            supp_feats = self.backbone(supp_x) # [K, 64]
            K = len(supp_feats)
            params = self.init_params

            # Sequential in-context adaptation loop (T = K examples shown one by one)
            for t in range(K):
                ft = supp_feats[t : t + 1] # [1, 64]
                yt = supp_y[t : t + 1]     # [1]
                yt_onehot = F.one_hot(yt, num_classes=n_classes).float() # [1, 5]

                # 1. Predict with current classifier state
                logits_t = forward_classifier(ft, params) # [1, 5]
                prob_t = F.softmax(logits_t, dim=-1)      # [1, 5]
                error_t = prob_t - yt_onehot              # [1, 5]

                # 2. Encode step sensory experience
                step_in = torch.cat([ft, yt_onehot, error_t], dim=-1) # [1, 74]
                ctx = self.step_encoder(step_in).squeeze(0)          # [325]

                # 3. SubQ Contraction Hop to update parameter state
                gates_ctx = self.w_ih(ctx)
                r_ctx, z_ctx, n_ctx = gates_ctx.chunk(3, dim=-1)

                gates_h = self.w_gate_h(params)
                r_h, z_h = gates_h.chunk(2, dim=-1)

                r = torch.sigmoid(r_ctx + r_h)
                z = torch.sigmoid(z_ctx + z_h)
                n = torch.tanh(n_ctx + self.w_cand_h(r * params))
                params = (1.0 - z) * n + z * params

            # Evaluate on query set with adapted params
            query_feats = self.backbone(query_x)
            query_logits = forward_classifier(query_feats, params)
            return query_logits

    # -------------------------------------------------------------------------
    # 6. MAML Baseline (Inner-Loop SGD on Head Parameters)
    # -------------------------------------------------------------------------
    class MAMLClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Conv4Backbone()
            self.init_params = nn.Parameter(torch.zeros(d_param))
            nn.init.normal_(self.init_params, std=0.05)
            self.inner_lr = nn.Parameter(torch.tensor(0.1))

        def adapt_and_evaluate(self, supp_x, supp_y, query_x):
            supp_feats = self.backbone(supp_x)
            K = len(supp_feats)
            params = self.init_params.clone()

            for t in range(K):
                ft = supp_feats[t : t + 1]
                yt = supp_y[t : t + 1]
                logits_t = forward_classifier(ft, params)
                loss_t = F.cross_entropy(logits_t, yt)
                grads = torch.autograd.grad(loss_t, params, create_graph=True)[0]
                params = params - self.inner_lr * grads

            query_feats = self.backbone(query_x)
            query_logits = forward_classifier(query_feats, params)
            return query_logits

    # -------------------------------------------------------------------------
    # 7. Meta-Training Loop (800 Episodes)
    # -------------------------------------------------------------------------
    total_episodes = 800
    n_way = 5
    k_train = 5 # 5-shot episodes during training
    n_query = 5

    print(f"\nMeta-Training: 5-Way 5-Shot Episodes on Omniglot ({total_episodes} Episodes)...")

    # 1. Train SubQ Meta-Optimizer
    print("\n[1/2] Training SubQ Vision Meta-Optimizer...")
    subq_net = SubQVisionMetaOptimizer().to(device)
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
    print(f"SubQ Meta-Optimizer training completed in {time.time() - t0:.1f}s")

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
    # 8. Evaluation on Held-Out Test Classes (Never Seen in Training)
    # -------------------------------------------------------------------------
    subq_net.eval()
    maml_net.eval()

    test_shots = [1, 2, 3, 5, 10]
    n_eval_episodes = 100

    print("\n" + "=" * 95)
    print("  FINAL EVALUATION ON COMPLETELY UNSEEN HELD-OUT OMNIGLOT TEST ALPHABETS (100 EPISODES)")
    print("=" * 95)
    print(f"{'Shot per Class (K)':<20} | {'Total Examples (T)':<20} | {'MAML (SGD) Top-1 Acc':<22} | {'SubQ Meta-Optimizer Top-1 Acc':<30} | {'Winner'}")
    print("-" * 110)

    for k in test_shots:
        total_T = k * n_way
        maml_accs = []
        subq_accs = []

        for _ in range(n_eval_episodes):
            s_x, s_y, q_x, q_y = sample_episode(test_classes, n_way=n_way, k_shot=k, n_query=n_query)

            # 1. SubQ
            with torch.no_grad():
                q_logits_subq = subq_net.adapt_and_evaluate(s_x, s_y, q_x)
                subq_accs.append((q_logits_subq.argmax(dim=-1) == q_y).float().mean().item() * 100.0)

            # 2. MAML (requires inner grad)
            with torch.enable_grad():
                q_logits_maml = maml_net.adapt_and_evaluate(s_x, s_y, q_x)
                maml_accs.append((q_logits_maml.argmax(dim=-1) == q_y).float().mean().item() * 100.0)

        mean_maml = np.mean(maml_accs)
        mean_subq = np.mean(subq_accs)
        winner = "🏆 SubQ Meta-Optimizer" if mean_subq > mean_maml else "— MAML"
        print(f"K = {k:<16} | T = {total_T:<16} | {mean_maml:>18.2f}% | {mean_subq:>26.2f}% | {winner}")

    print("=" * 110)

@app.local_entrypoint()
def main():
    run_omniglot_meta_study.remote()
