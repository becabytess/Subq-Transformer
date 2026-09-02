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

app = modal.App("exp-subq-vit-state-filterbank-shootout", image=image)

@app.function(gpu="A10G", timeout=3600)
def run_state_filterbank_shootout():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 140)
    print("  STUDY 72: FAIR COMPARISON OF STATE-AVERAGED HARMONIC SUPER-TOKENS VS DISCRETE TOKEN ATTENTION")
    print("  High-Res CIFAR-100 (L = 257 Patches, 10 Epochs, T_train = 4 Hops, Pre-LayerNorm State Pooling)")
    print("  Evaluating:")
    print("    1. Discrete K = 8 Points (8 raw tokens covered, 8 attention dot-products)")
    print("    2. Discrete K = 40 Points (40 raw tokens covered, 40 attention dot-products)")
    print("    3. State Filterbank 8 Super-Tokens x 5-Patch Pool (40 raw tokens covered, 8 attention dot-products - 5x cheaper)")
    print("    4. State Filterbank 4 Super-Tokens x 2-Patch Pool (8 raw tokens covered, 4 attention dot-products - 2x cheaper)")
    print("=" * 140)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Dataset & Transforms (L = 257 Patches)
    print("\n[1/3] Loading CIFAR-100 Dataset via Fast Cloud CDN...")
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
    epochs = 10 # Fast 10 epochs
    T_train = 4

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
    # 3. Model A: Discrete Point Attention (K = 8 or K = 40)
    # -------------------------------------------------------------------------
    class DiscreteSubQViT(nn.Module):
        def __init__(self, K_points=8):
            super().__init__()
            self.K_points = K_points
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

        def forward(self, x):
            B = x.shape[0]
            s = self.patch_embed(x)
            L = s.shape[1]
            device = x.device

            curr_wave = self.init_wave_latent
            step_mult = 1.0 / math.sqrt(T_train)

            for t in range(T_train):
                s_norm = self.ln_1(s)

                # Q, K, V
                qkv = self.c_attn(s_norm).chunk(3, dim=-1)
                q, k, v = [t_tensor.view(B, L, n_heads, head_dim).transpose(1, 2) for t_tensor in qkv]

                # Wave Carrier
                params = curr_wave.view(n_heads, num_waves, 4)
                amp = torch.tanh(params[..., 0]).view(1, n_heads, 1, num_waves)
                omega = (F.softplus(params[..., 1]).view(1, n_heads, 1, num_waves) * self.base_freqs)
                phi = (params[..., 2] * math.pi).view(1, n_heads, 1, num_waves)
                decay = (F.softplus(params[..., 3]) * 0.05).view(1, n_heads, 1, num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

                topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.K_points - 1, dim=-1)
                past_peak_offsets = past_peak_offsets + 1
                zero_offset = torch.zeros((B, n_heads, 1), dtype=torch.long, device=device)
                zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=device)
                peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
                peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

                q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
                target_indices = (q_pos - peak_offsets.unsqueeze(2)) % L
                idx_exp = target_indices.unsqueeze(-1).expand(B, n_heads, L, self.K_points, head_dim)

                K_g = torch.gather(k.unsqueeze(3).expand(B, n_heads, L, self.K_points, head_dim), dim=2, index=idx_exp)
                V_g = torch.gather(v.unsqueeze(3).expand(B, n_heads, L, self.K_points, head_dim), dim=2, index=idx_exp)

                scores = (q.unsqueeze(3) * K_g).sum(dim=-1) / math.sqrt(head_dim) + peak_vals.unsqueeze(2)
                attn = F.softmax(scores, dim=-1)
                attn_out = (attn.unsqueeze(-1) * V_g).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
                attn_out = self.c_proj(attn_out)

                s = s + step_mult * attn_out
                s = s + step_mult * self.mlp(self.ln_2(s))

                if t < T_train - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            s_final = self.ln_f(s)
            cls_out = s_final[:, 0]
            logits = self.head(cls_out)
            return logits

    # -------------------------------------------------------------------------
    # 4. Model B: State-Averaged Harmonic Super-Token ViT (Fast & Clean)
    # -------------------------------------------------------------------------
    class StateAveragedFilterbankViT(nn.Module):
        def __init__(self, P_super_tokens=8, filter_radius=2):
            super().__init__()
            self.P_super_tokens = P_super_tokens
            self.filter_radius = filter_radius
            self.window_size = 2 * filter_radius + 1
            self.patch_embed = HighResPatchEmbed()

            self.ln_q = nn.LayerNorm(d_model)
            self.ln_kv = nn.LayerNorm(d_model) # Normalizes the pooled state!
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.c_proj = nn.Linear(d_model, d_model)

            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )

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

            win_offs = torch.arange(-self.filter_radius, self.filter_radius + 1).float()
            tri_weights = 1.0 - torch.abs(win_offs) / (self.filter_radius + 1.0)
            self.register_buffer("win_offs", win_offs.long().view(1, 1, 1, self.window_size))
            self.register_buffer("tri_prior", tri_weights.view(1, 1, 1, self.window_size))

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            B = x.shape[0]
            s = self.patch_embed(x)
            L = s.shape[1]
            device = x.device

            curr_wave = self.init_wave_latent
            step_mult = 1.0 / math.sqrt(T_train)
            D_len = self.max_d - 1

            for t in range(T_train):
                # 1. Project Query from Current State
                q = self.q_proj(self.ln_q(s)).view(B, L, n_heads, head_dim).transpose(1, 2)

                # 2. Continuous Wave & Peak Centers
                params = curr_wave.view(n_heads, num_waves, 4)
                amp = torch.tanh(params[..., 0]).view(1, n_heads, 1, num_waves)
                omega = (F.softplus(params[..., 1]).view(1, n_heads, 1, num_waves) * self.base_freqs)
                phi = (params[..., 2] * math.pi).view(1, n_heads, 1, num_waves)
                decay = (F.softplus(params[..., 3]) * 0.05).view(1, n_heads, 1, num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1) # [B, n_heads, 256]

                # 3. Peak Finding via 1D NMS
                pad = self.filter_radius
                padded = F.pad(wave_1d, (pad, pad), mode='replicate')
                pooled = F.max_pool1d(padded, kernel_size=self.window_size, stride=1)
                is_max = (wave_1d == pooled)
                suppressed = torch.where(is_max, wave_1d, torch.tensor(-1e4, device=device))

                top_vals, top_indices = torch.topk(suppressed, k=self.P_super_tokens - 1, dim=-1)
                peak_centers = top_indices + 1

                zero_center = torch.zeros((B, n_heads, 1), dtype=torch.long, device=device)
                zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=device)
                all_centers = torch.cat([zero_center, peak_centers], dim=-1) # [B, n_heads, P]
                all_peak_vals = torch.cat([zero_val, top_vals], dim=-1)      # [B, n_heads, P]

                # 4. Filterbank Weights
                neighbor_offsets = (all_centers.unsqueeze(-1) + self.win_offs) % L # [B, n_heads, P, W]
                wave_lookup_idx = (neighbor_offsets - 1).clamp(0, D_len - 1)
                local_wave_energy = torch.gather(wave_1d.unsqueeze(2).expand(B, n_heads, self.P_super_tokens, D_len), dim=-1, index=wave_lookup_idx)
                filter_logits = local_wave_energy * 0.5 + torch.log(self.tri_prior.clamp(min=1e-4))
                filter_weights = F.softmax(filter_logits, dim=-1) # [B, n_heads, P, W]

                # 5. Fast State-Averaging: Gather States directly in D-dim space
                q_pos = torch.arange(L, device=device).view(1, 1, L, 1, 1) # [1, 1, L, 1, 1]
                # target_tokens: [B, n_heads, L, P, W]
                target_tokens = (q_pos - neighbor_offsets.unsqueeze(2)) % L

                # Reshape s to [B, 1, L, 1, 1, d_model] and gather along L
                s_expanded = s.view(B, 1, L, 1, 1, d_model).expand(B, n_heads, L, self.P_super_tokens, self.window_size, d_model)
                idx_exp = target_tokens.unsqueeze(-1).expand(B, n_heads, L, self.P_super_tokens, self.window_size, d_model)
                s_neighbors = torch.gather(s_expanded, dim=2, index=idx_exp) # [B, n_heads, L, P, W, d_model]

                # Weighted Pool State -> Super-States: s_super in [B, n_heads, L, P, d_model]
                w_exp = filter_weights.unsqueeze(2).unsqueeze(-1) # [B, n_heads, 1, P, W, 1]
                s_super = (w_exp * s_neighbors).sum(dim=4) # [B, n_heads, L, P, d_model]

                # 6. Normalize Pooled State and Project to K, V Super-Tokens!
                s_super_norm = self.ln_kv(s_super)
                k_super = self.k_proj(s_super_norm).view(B, n_heads, L, self.P_super_tokens, n_heads, head_dim)
                v_super = self.v_proj(s_super_norm).view(B, n_heads, L, self.P_super_tokens, n_heads, head_dim)
                # Select the corresponding head slice: [B, n_heads, L, P, head_dim]
                head_idx = torch.arange(n_heads, device=device).view(1, n_heads, 1, 1, 1, 1)
                k_super = torch.gather(k_super, dim=4, index=head_idx.expand(B, n_heads, L, self.P_super_tokens, 1, head_dim)).squeeze(4)
                v_super = torch.gather(v_super, dim=4, index=head_idx.expand(B, n_heads, L, self.P_super_tokens, 1, head_dim)).squeeze(4)

                # 7. Attention against P Super-Tokens
                scores = (q.unsqueeze(3) * k_super).sum(dim=-1) / math.sqrt(head_dim) + all_peak_vals.unsqueeze(2)
                attn = F.softmax(scores, dim=-1) # [B, n_heads, L, P]
                attn_out = (attn.unsqueeze(-1) * v_super).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
                attn_out = self.c_proj(attn_out)

                s = s + step_mult * attn_out
                s = s + step_mult * self.mlp(self.ln_2(s))

                if t < T_train - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            s_final = self.ln_f(s)
            cls_out = s_final[:, 0]
            logits = self.head(cls_out)
            return logits

    # -------------------------------------------------------------------------
    # 5. Helper to Train & Evaluate Each Candidate (10 Epochs)
    # -------------------------------------------------------------------------
    def train_and_eval(model, name, raw_tokens_covered, attn_ops):
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n" + "=" * 115)
        print(f"  RUNNING: {name}")
        print(f"  Params: {param_count:,} | Raw Tokens Covered: {raw_tokens_covered} | Attention Dot-Products: {attn_ops} | Epochs: {epochs}")
        print("=" * 115)

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
            if epoch % 2 == 0 or epoch == epochs or epoch == 1:
                print(f"  Epoch {epoch:>2d}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss/train_total:.4f}, Train Acc: {train_acc:>5.2f}%")

        total_train_time = time.time() - t_start

        # Test Set Evaluation
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

        print(f"  --> {name} | Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}% | Loss: {avg_loss:.4f} | Time: {total_train_time:.1f}s")
        return {
            "name": name,
            "raw_tokens": raw_tokens_covered,
            "attn_ops": attn_ops,
            "top1": top1_acc,
            "top5": top5_acc,
            "loss": avg_loss,
            "time": total_train_time,
            "params": param_count
        }

    # -------------------------------------------------------------------------
    # 6. Run the 4 Candidates
    # -------------------------------------------------------------------------
    candidates = [
        (DiscreteSubQViT(K_points=8), "1. Discrete Points (K = 8)", 8, 8),
        (DiscreteSubQViT(K_points=40), "2. Discrete Points (K = 40)", 40, 40),
        (StateAveragedFilterbankViT(P_super_tokens=8, filter_radius=2), "3. State Super-Tokens (P=8, Win=5)", 40, 8),
        (StateAveragedFilterbankViT(P_super_tokens=4, filter_radius=1), "4. State Super-Tokens (P=4, Win=3)", 12, 4)
    ]

    results = []
    for m, name, raw_toks, attn_ops in candidates:
        res = train_and_eval(m, name, raw_toks, attn_ops)
        results.append(res)

    # -------------------------------------------------------------------------
    # 7. Final Comparative Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 150)
    print("  STUDY 72 FINAL COMPARATIVE BENCHMARK: STATE-AVERAGED HARMONIC SUPER-TOKENS VS DISCRETE ATTENTION")
    print(f"  Dataset: High-Res CIFAR-100 (L = 257 Patches) | Thought Hops T = {T_train} | Epochs = {epochs}")
    print("=" * 150)
    print(f"{'Architecture':<42} | {'Raw Tokens Covered':<20} | {'Attention Dot-Products':<24} | {'Top-1 Acc':<12} | {'Top-5 Acc':<12} | {'Test Loss':<12} | {'Train Time':<10}")
    print("-" * 150)
    for r in results:
        print(f"{r['name']:<42} | {r['raw_tokens']:<20} | {r['attn_ops']:<24} | {r['top1']:<10.2f}% | {r['top5']:<10.2f}% | {r['loss']:<12.4f} | {r['time']:>8.1f}s")
    print("=" * 150)

    return results

@app.local_entrypoint()
def main():
    run_state_filterbank_shootout.remote()
