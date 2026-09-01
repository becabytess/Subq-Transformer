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

app = modal.App("subq-twin-vision-omniglot", image=image)

@app.function(gpu="T4", timeout=900)
def run_vision_twin_study():
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
    print("  STUDY 2/3: SUBQ FORWARD-BACKWARD TWIN ON REAL VISION (5-WAY OMNIGLOT)")
    print("=" * 95)
    print(f"Device: {torch.cuda.get_device_name(0)}")

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
            return self.encoder(x).squeeze(-1).squeeze(-1)

    class SubQBlock(nn.Module):
        def __init__(self, d_model=64):
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

    class VisionTwinSystem(nn.Module):
        def __init__(self, d_feat=64, n_classes=5):
            super().__init__()
            self.backbone = Conv4Backbone()
            self.d_feat = d_feat
            self.n_classes = n_classes

            # Forward SubQ
            self.fwd_subq = SubQBlock(d_model=d_feat)
            self.base_head = nn.Parameter(torch.zeros(d_feat, n_classes))
            nn.init.xavier_uniform_(self.base_head)

            # Backward Error Twin
            self.err_in_proj = nn.Linear(n_classes, d_feat)
            self.err_subq = SubQBlock(d_model=d_feat)
            self.meta_lr = nn.Parameter(torch.tensor(0.5))

        def self_teach_and_evaluate(self, s_x, s_y, q_x, T_fwd=4, T_err=4):
            # 1. Forward pass on support images
            s_feats = self.backbone(s_x).unsqueeze(0) # [1, K, 64]
            h_fwd = self.fwd_subq(s_feats, T=T_fwd).squeeze(0) # [K, 64]
            
            s_logits = torch.mm(h_fwd, self.base_head) # [K, 5]
            s_prob = F.softmax(s_logits, dim=-1)
            y_onehot = F.one_hot(s_y, num_classes=self.n_classes).float() # [K, 5]

            # 2. Error Signal
            error = y_onehot - s_prob # [K, 5]

            # 3. Error Twin Multi-Hop Settling
            h_err_in = self.err_in_proj(error).unsqueeze(0) # [1, K, 64]
            h_err = self.err_subq(h_err_in, T=T_err).squeeze(0) # [K, 64]

            # 4. Bilinear Co-Settling Outer Product Delta
            delta_head = torch.mm(h_fwd.t(), error) / len(s_x) # [64, 5]
            delta_twin = torch.mm(h_err.t(), error) / len(s_x) # [64, 5]

            adapted_head = self.base_head + self.meta_lr * (delta_head + delta_twin)

            # 5. Evaluate on Query Images
            q_feats = self.backbone(q_x).unsqueeze(0) # [1, N_q, 64]
            q_h_fwd = self.fwd_subq(q_feats, T=T_fwd).squeeze(0) # [N_q, 64]
            q_logits = torch.mm(q_h_fwd, adapted_head)
            return q_logits

    total_episodes = 600
    print(f"Meta-Training Vision Twin ({total_episodes} Episodes)...")
    twin_net = VisionTwinSystem(d_feat=64, n_classes=5).to(device)
    optimizer = torch.optim.AdamW(twin_net.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for ep in range(1, total_episodes + 1):
        twin_net.train()
        s_x, s_y, q_x, q_y = sample_episode(train_classes, n_way=5, k_shot=5, n_query=5)

        q_logits = twin_net.self_teach_and_evaluate(s_x, s_y, q_x, T_fwd=4, T_err=4)
        loss = F.cross_entropy(q_logits, q_y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(twin_net.parameters(), 1.0)
        optimizer.step()

        if ep % 200 == 0 or ep == total_episodes:
            acc = (q_logits.argmax(dim=-1) == q_y).float().mean().item() * 100.0
            print(f"  Episode {ep:>4}/{total_episodes} | Query Loss: {loss.item():.4f} | Top-1 Accuracy: {acc:.2f}%")
    print(f"Vision Twin training completed in {time.time() - t0:.1f}s\n")

    # Evaluation on Held-Out Test Alphabets (100 Episodes)
    twin_net.eval()
    test_shots = [1, 2, 3, 5, 10]
    n_eval_episodes = 100

    print("=" * 95)
    print("  FINAL EVALUATION ON COMPLETELY UNSEEN HELD-OUT OMNIGLOT TEST ALPHABETS")
    print("=" * 95)
    print(f"{'Shot per Class (K)':<22} | {'SubQ Twin Top-1 Accuracy':<30} | {'Status'}")
    print("-" * 95)

    with torch.no_grad():
        for k in test_shots:
            accs = []
            for _ in range(n_eval_episodes):
                s_x, s_y, q_x, q_y = sample_episode(test_classes, n_way=5, k_shot=k, n_query=5)
                q_logits = twin_net.self_teach_and_evaluate(s_x, s_y, q_x, T_fwd=4, T_err=4)
                accs.append((q_logits.argmax(dim=-1) == q_y).float().mean().item() * 100.0)
            mean_acc = np.mean(accs)
            print(f"K = {k:<18} | {mean_acc:>26.2f}% | Evaluated (100 Episodes)")

    print("=" * 95)

@app.local_entrypoint()
def main():
    run_vision_twin_study.remote()
