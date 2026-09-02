import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "datasets>=2.18.0",
        "numpy",
        "pillow"
    )
)

app = modal.App("exp-subq-vit-peak-clustering-ablation", image=image)
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)

@app.function(gpu="A10G", timeout=7200, volumes={"/models": volume})
def run_peak_clustering_ablation_experiment():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 135)
    print("  STUDY 70: DISSECTING PEAK CLUSTERING VS MACRO-REGION WINDOW SAMPLING IN HARMONIC SUBQ ViT")
    print("  Comparing 4 Token Routing Modes (Fixed Budget K = 8 Tokens, L = 257, T_train = 4 Hops, 20 Epochs):")
    print("    1. Standard Raw Top-8 Points (Baseline: Clustered Gradients)")
    print("    2. 4 Macro Peaks x 2-Neighbor Window (P=4, W=2 -> 8 Tokens)")
    print("    3. 2 Macro Peaks x 4-Neighbor Window (P=2, W=4 -> 8 Tokens)")
    print("    4. 8 Strictly Separated Peaks with NMS (P=8, W=1 -> 8 Tokens)")
    print("=" * 135)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Dataset & Transforms (L = 257 Patches)
    print("\n[1/4] Loading CIFAR-100 Dataset via Fast Cloud CDN...")
    t_d0 = time.time()
    hf_dataset = load_dataset("uoft-cs/cifar100")
    print(f"--> Loaded CIFAR-100 in {time.time()-t_d0:.2f}s! Train: {len(hf_dataset['train'])}, Test: {len(hf_dataset['test'])}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    class Cifar100Wrapper(Dataset):
        def __init__(self, split_data, transform=None):
            self.data = split_data
            self.transform = transform
            self.has_img_key = 'img' in self.data.column_names

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            img = item['img'] if self.has_img_key else item['image']
            label = item['fine_label'] if 'fine_label' in item else item['label']
            if self.transform:
                img = self.transform(img)
            return img, label

    train_dataset = Cifar100Wrapper(hf_dataset['train'], transform=transform_train)
    test_dataset = Cifar100Wrapper(hf_dataset['test'], transform=transform_test)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    d_model = 192
    n_heads = 6
    head_dim = d_model // n_heads # 32
    patch_size = 2
    num_patches = (32 // patch_size) ** 2 # 256 patches
    seq_len = num_patches + 1 # 257 tokens (with [CLS])
    num_classes = 100
    num_waves = 12
    epochs = 20
    T_train = 4
    K_total = 8

    # -------------------------------------------------------------------------
    # 2. Patch Embedding Layer
    # -------------------------------------------------------------------------
    class HighResPatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        def forward(self, x):
            B = x.shape[0]
            x = self.proj(x).flatten(2).transpose(1, 2)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            return x

    # -------------------------------------------------------------------------
    # 3. Harmonic SubQ ViT with Flexible Peak Extraction Mode
    # -------------------------------------------------------------------------
    class HarmonicSubQViTAblation(nn.Module):
        def __init__(self, routing_mode="raw_topk"):
            """
            routing_mode:
              - 'raw_topk': Standard topk(k=8) points (allows clustering)
              - '4peaks_2neigh': 4 separated peaks x 2-neighbor window = 8 tokens
              - '2peaks_4neigh': 2 separated peaks x 4-neighbor window = 8 tokens
              - '8peaks_nms': 8 strictly separated peaks with Non-Maximum Suppression = 8 tokens
            """
            super().__init__()
            self.routing_mode = routing_mode
            self.patch_embed = HighResPatchEmbed()

            self.ln_1 = nn.LayerNorm(d_model)
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)

            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )

            # Continuous Spatial Wave Parameters
            init_latents = torch.zeros(n_heads, num_waves, 4)
            init_latents[..., 0] = 0.5
            init_latents[..., 3] = 0.1
            self.init_wave_latent = nn.Parameter(init_latents.view(n_heads, num_waves * 4))
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4)
            )

            log_freqs = torch.linspace(math.log10(math.pi / 1.0), math.log10(math.pi / 64.0), num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.max_d = seq_len
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, 1, self.max_d - 1, 1))

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def extract_peak_offsets_and_biases(self, wave_1d, B):
            """
            wave_1d: [B, n_heads, seq_len - 1] (distance d in [1..256])
            Returns:
              peak_offsets: [B, n_heads, 8]
              peak_vals: [B, n_heads, 8]
            """
            device = wave_1d.device
            D_len = wave_1d.shape[-1] # 256

            if self.routing_mode == "raw_topk":
                # Mode 1: Standard top-k points
                topk_vals, past_peak_offsets = torch.topk(wave_1d, k=K_total - 1, dim=-1)
                past_peak_offsets = past_peak_offsets + 1
                zero_offset = torch.zeros((B, n_heads, 1), dtype=torch.long, device=device)
                zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=device)
                peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
                peak_vals = torch.cat([zero_val, topk_vals], dim=-1)
                return peak_offsets, peak_vals

            elif self.routing_mode == "4peaks_2neigh":
                # Mode 2: 4 macro peaks (kernel_size=5 NMS) x 2 neighbors [d*, d*+1]
                # 1. 1D Max Pool for local maxima suppression
                pad = 2
                padded = F.pad(wave_1d, (pad, pad), mode='replicate')
                pooled = F.max_pool1d(padded, kernel_size=5, stride=1)
                is_max = (wave_1d == pooled)
                suppressed = torch.where(is_max, wave_1d, torch.tensor(-1e4, device=device))

                # Top 4 distinct macro peaks
                top4_vals, top4_indices = torch.topk(suppressed, k=4, dim=-1)
                top4_centers = top4_indices + 1 # [B, n_heads, 4]

                # 2. Expand 2-neighbor window [d*, d*+1]
                offsets_list = []
                vals_list = []
                for w_off in [0, 1]:
                    cur_d = (top4_centers + w_off - 1) % D_len + 1 # [B, n_heads, 4]
                    # Gather corresponding wave value
                    idx_gather = (cur_d - 1).clamp(0, D_len - 1)
                    cur_v = torch.gather(wave_1d, dim=-1, index=idx_gather)
                    offsets_list.append(cur_d)
                    vals_list.append(cur_v)

                all_offsets = torch.cat(offsets_list, dim=-1) # [B, n_heads, 8]
                all_vals = torch.cat(vals_list, dim=-1)       # [B, n_heads, 8]
                # Ensure self-attention (0) is present
                all_offsets[:, :, 0] = 0
                all_vals[:, :, 0] = 0.0
                return all_offsets, all_vals

            elif self.routing_mode == "2peaks_4neigh":
                # Mode 3: 2 macro peaks (kernel_size=9 NMS) x 4 neighbors [d*-1, d*, d*+1, d*+2]
                pad = 4
                padded = F.pad(wave_1d, (pad, pad), mode='replicate')
                pooled = F.max_pool1d(padded, kernel_size=9, stride=1)
                is_max = (wave_1d == pooled)
                suppressed = torch.where(is_max, wave_1d, torch.tensor(-1e4, device=device))

                # Top 2 macro peaks
                top2_vals, top2_indices = torch.topk(suppressed, k=2, dim=-1)
                top2_centers = top2_indices + 1 # [B, n_heads, 2]

                # 4-neighbor window [-1, 0, 1, 2]
                offsets_list = []
                vals_list = []
                for w_off in [-1, 0, 1, 2]:
                    cur_d = (top2_centers + w_off - 1) % D_len + 1
                    idx_gather = (cur_d - 1).clamp(0, D_len - 1)
                    cur_v = torch.gather(wave_1d, dim=-1, index=idx_gather)
                    offsets_list.append(cur_d)
                    vals_list.append(cur_v)

                all_offsets = torch.cat(offsets_list, dim=-1) # [B, n_heads, 8]
                all_vals = torch.cat(vals_list, dim=-1)       # [B, n_heads, 8]
                all_offsets[:, :, 0] = 0
                all_vals[:, :, 0] = 0.0
                return all_offsets, all_vals

            elif self.routing_mode == "8peaks_nms":
                # Mode 4: 8 strictly separated peaks with NMS (radius 4)
                pad = 2
                padded = F.pad(wave_1d, (pad, pad), mode='replicate')
                pooled = F.max_pool1d(padded, kernel_size=5, stride=1)
                is_max = (wave_1d == pooled)
                suppressed = torch.where(is_max, wave_1d, torch.tensor(-1e4, device=device))

                # Top 7 distinct peaks (plus self 0 = 8 total)
                top7_vals, top7_indices = torch.topk(suppressed, k=K_total - 1, dim=-1)
                top7_offsets = top7_indices + 1
                zero_offset = torch.zeros((B, n_heads, 1), dtype=torch.long, device=device)
                zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=device)
                peak_offsets = torch.cat([zero_offset, top7_offsets], dim=-1)
                peak_vals = torch.cat([zero_val, top7_vals], dim=-1)
                return peak_offsets, peak_vals

            else:
                raise ValueError(f"Unknown routing mode: {self.routing_mode}")

        def forward(self, x):
            B = x.shape[0]
            s = self.patch_embed(x)
            L = s.shape[1]

            curr_wave = self.init_wave_latent
            step_mult = 1.0 / math.sqrt(T_train)

            for t in range(T_train):
                s_norm = self.ln_1(s)

                # 1. Full Evolving Q, K, V
                qkv = self.c_attn(s_norm).chunk(3, dim=-1)
                q, k, v = [t_tensor.view(B, L, n_heads, head_dim).transpose(1, 2) for t_tensor in qkv]

                # 2. Continuous Spatial Wave Carrier
                params = curr_wave.view(n_heads, num_waves, 4)
                amp = torch.tanh(params[..., 0]).view(1, n_heads, 1, num_waves)
                omega = (F.softplus(params[..., 1]).view(1, n_heads, 1, num_waves) * self.base_freqs)
                phi = (params[..., 2] * math.pi).view(1, n_heads, 1, num_waves)
                decay = (F.softplus(params[..., 3]) * 0.05).view(1, n_heads, 1, num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

                # 3. Peak Extraction according to Routing Mode
                peak_offsets, peak_vals = self.extract_peak_offsets_and_biases(wave_1d, B)

                # 4. Vectorized Gather Keys & Values
                q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
                target_indices = (q_pos - peak_offsets.unsqueeze(2)) % L
                idx_exp = target_indices.unsqueeze(-1).expand(B, n_heads, L, K_total, head_dim)

                K_g = torch.gather(k.unsqueeze(3).expand(B, n_heads, L, K_total, head_dim), dim=2, index=idx_exp)
                V_g = torch.gather(v.unsqueeze(3).expand(B, n_heads, L, K_total, head_dim), dim=2, index=idx_exp)

                # 5. Attention with Harmonic Logit Bias
                scores = (q.unsqueeze(3) * K_g).sum(dim=-1) / math.sqrt(head_dim) + peak_vals.unsqueeze(2)
                attn = F.softmax(scores, dim=-1)
                attn_out = (attn.unsqueeze(-1) * V_g).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
                attn_out = self.c_proj(attn_out)

                # 6. Recurrent Residual Update
                s = s + step_mult * attn_out
                s = s + step_mult * self.mlp(self.ln_2(s))

                # 7. Wave Transition
                if t < T_train - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            s_final = self.ln_f(s)
            cls_out = s_final[:, 0]
            logits = self.head(cls_out)
            return logits

    # -------------------------------------------------------------------------
    # 4. Train & Evaluate Helper Function
    # -------------------------------------------------------------------------
    def train_and_eval_mode(mode_name, display_title):
        print("\n" + "=" * 115)
        print(f"  RUNNING VARIANT: {display_title}")
        print("=" * 115)
        model = HarmonicSubQViTAblation(routing_mode=mode_name).to(device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {param_count:,} | Thought Hops: {T_train} | Epochs: {epochs} | Token Budget: {K_total}")

        scaler = torch.amp.GradScaler('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        t_start = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            ep_start = time.time()

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                train_correct += preds.eq(labels).sum().item()
                train_total += images.size(0)

            train_acc = (train_correct / train_total) * 100.0
            ep_time = time.time() - ep_start
            if epoch % 5 == 0 or epoch == epochs or epoch == 1:
                print(f"  Epoch {epoch:>2d}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss/train_total:.4f}, Train Acc: {train_acc:>5.2f}%")

        total_train_time = time.time() - t_start

        # Evaluate on Test Set
        model.eval()
        test_loss, test_correct_top1, test_correct_top5, test_total = 0.0, 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, top1 = outputs.max(1)
                test_correct_top1 += top1.eq(labels).sum().item()
                _, top5 = outputs.topk(5, dim=1)
                test_correct_top5 += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
                test_total += images.size(0)

        top1_acc = (test_correct_top1 / test_total) * 100.0
        top5_acc = (test_correct_top5 / test_total) * 100.0
        avg_loss = test_loss / test_total

        print(f"  --> {display_title} | Top-1 Acc: {top1_acc:.2f}% | Top-5 Acc: {top5_acc:.2f}% | Test Loss: {avg_loss:.4f} | Total Time: {total_train_time:.1f}s")
        return {
            "mode": mode_name,
            "title": display_title,
            "params": param_count,
            "top1": top1_acc,
            "top5": top5_acc,
            "loss": avg_loss,
            "time": total_train_time
        }

    # -------------------------------------------------------------------------
    # 5. Run the 4-Way Shootout
    # -------------------------------------------------------------------------
    results = []
    variants = [
        ("raw_topk", "Mode 1: Standard Raw Top-8 Points (Baseline Clustered)"),
        ("4peaks_2neigh", "Mode 2: 4 Macro Peaks x 2 Neighbors (P=4, W=2)"),
        ("2peaks_4neigh", "Mode 3: 2 Macro Peaks x 4 Neighbors (P=2, W=4)"),
        ("8peaks_nms", "Mode 4: 8 Strictly Separated Peaks with NMS (P=8, W=1)")
    ]

    for mode_name, display_title in variants:
        res = train_and_eval_mode(mode_name, display_title)
        results.append(res)

    # -------------------------------------------------------------------------
    # 6. Final Comparative Results Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 145)
    print("  STUDY 70 FINAL COMPARATIVE RESULTS: PEAK CLUSTERING VS MACRO-REGION WINDOW SAMPLING")
    print(f"  Dataset: CIFAR-100 (High-Res L=257 Patches) | Total Token Budget K = {K_total} | Thought Hops T = {T_train} | Epochs = {epochs}")
    print("=" * 145)
    print(f"{'Routing Strategy':<52} | {'Tokens / Peak':<16} | {'Top-1 Acc':<12} | {'Top-5 Acc':<12} | {'Test Loss':<12} | {'Train Time':<10}")
    print("-" * 145)
    tokens_desc = {
        "raw_topk": "8 unconstrained",
        "4peaks_2neigh": "4 peaks x 2 nbr",
        "2peaks_4neigh": "2 peaks x 4 nbr",
        "8peaks_nms": "8 distinct peaks"
    }
    for r in results:
        print(f"{r['title']:<52} | {tokens_desc[r['mode']]:<16} | {r['top1']:<10.2f}% | {r['top5']:<10.2f}% | {r['loss']:<12.4f} | {r['time']:>8.1f}s")
    print("=" * 145)

    return results

@app.local_entrypoint()
def main():
    run_peak_clustering_ablation_experiment.remote()
