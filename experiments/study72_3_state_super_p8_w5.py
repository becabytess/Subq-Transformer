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

app = modal.App("study72-3-state-super-p8-w5", image=image)
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)

@app.function(gpu="A10G", timeout=1800, volumes={"/models": volume})
def train_state_super_p8_w5():
    import math, time, json, torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 110)
    print("  STUDY 72 - CANDIDATE 3: STATE SUPER-TOKENS (P = 8 Super-Tokens, Win = 5 Patches)")
    print("  High-Res CIFAR-100 (L = 257 Patches, 10 Epochs, T_train = 4 Hops, Pre-LN State Pooling)")
    print("  Raw Tokens Covered: 40 | Attention Dot-Products: 8 (5x Cheaper than K=40!)")
    print("=" * 110)

    hf_dataset = load_dataset("uoft-cs/cifar100")
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
            if self.transform: img = self.transform(img)
            return img, label

    train_loader = DataLoader(Cifar100Wrapper(hf_dataset['train'], transform=transform_train), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(Cifar100Wrapper(hf_dataset['test'], transform=transform_test), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    d_model, n_heads = 192, 6
    head_dim = d_model // n_heads # 32
    patch_size = 2
    num_patches = (32 // patch_size) ** 2 # 256
    seq_len = num_patches + 1 # 257
    num_classes = 100
    num_waves = 12
    epochs = 10
    T_train = 4
    P_super = 8
    filter_radius = 2
    window_size = 2 * filter_radius + 1 # 5

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
            return torch.cat((cls_tokens, x), dim=1) + self.pos_embed

    class StateSuperTokenViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = HighResPatchEmbed()
            self.ln_q = nn.LayerNorm(d_model)
            self.ln_kv = nn.LayerNorm(head_dim)
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(head_dim, head_dim)
            self.v_proj = nn.Linear(head_dim, head_dim)
            self.c_proj = nn.Linear(d_model, d_model)
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

            init_latents = torch.zeros(n_heads, num_waves, 4)
            init_latents[..., 0] = 0.5; init_latents[..., 3] = 0.1
            self.init_wave_latent = nn.Parameter(init_latents.view(n_heads, num_waves * 4))
            self.wave_transition = nn.Sequential(nn.Linear(num_waves * 4, 32), nn.GELU(), nn.Linear(32, num_waves * 4))
            log_freqs = torch.linspace(math.log10(math.pi / 1.0), math.log10(math.pi / 64.0), num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.max_d = seq_len
            self.register_buffer("d_grid", torch.arange(1, seq_len).float().view(1, 1, seq_len - 1, 1))

            win_offs = torch.arange(-filter_radius, filter_radius + 1).float()
            tri_weights = 1.0 - torch.abs(win_offs) / (filter_radius + 1.0)
            self.register_buffer("win_offs", win_offs.long().view(1, 1, 1, window_size))
            self.register_buffer("tri_prior", tri_weights.view(1, 1, 1, window_size))

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            B = x.shape[0]
            s = self.patch_embed(x)
            L = s.shape[1]
            device = x.device
            curr_wave = self.init_wave_latent
            step_mult = 1.0 / math.sqrt(T_train)
            D_len = seq_len - 1

            for t in range(T_train):
                q = self.q_proj(self.ln_q(s)).view(B, L, n_heads, head_dim).transpose(1, 2)

                params = curr_wave.view(n_heads, num_waves, 4)
                amp = torch.tanh(params[..., 0]).view(1, n_heads, 1, num_waves)
                omega = (F.softplus(params[..., 1]).view(1, n_heads, 1, num_waves) * self.base_freqs)
                phi = (params[..., 2] * math.pi).view(1, n_heads, 1, num_waves)
                decay = (F.softplus(params[..., 3]) * 0.05).view(1, n_heads, 1, num_waves)
                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

                pad = filter_radius
                padded = F.pad(wave_1d, (pad, pad), mode='replicate')
                pooled = F.max_pool1d(padded, kernel_size=window_size, stride=1)
                is_max = (wave_1d == pooled)
                suppressed = torch.where(is_max, wave_1d, torch.tensor(-1e4, device=device))

                top_vals, top_indices = torch.topk(suppressed, k=P_super - 1, dim=-1)
                peak_centers = top_indices + 1
                zero_center = torch.zeros((B, n_heads, 1), dtype=torch.long, device=device)
                zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=device)
                all_centers = torch.cat([zero_center, peak_centers], dim=-1)
                all_peak_vals = torch.cat([zero_val, top_vals], dim=-1)

                neighbor_offsets = (all_centers.unsqueeze(-1) + self.win_offs) % L
                wave_lookup_idx = (neighbor_offsets - 1).clamp(0, D_len - 1)
                local_wave_energy = torch.gather(wave_1d.unsqueeze(2).expand(B, n_heads, P_super, D_len), dim=-1, index=wave_lookup_idx)
                filter_logits = local_wave_energy * 0.5 + torch.log(self.tri_prior.clamp(min=1e-4))
                filter_weights = F.softmax(filter_logits, dim=-1)

                # 5. Memory-Efficient Head-Sliced State-Averaging:
                # s_heads: [B, n_heads, L, head_dim] (slashes memory 6x!)
                s_heads = s.view(B, L, n_heads, head_dim).transpose(1, 2)
                
                q_pos = torch.arange(L, device=device).view(1, 1, L, 1, 1)
                target_tokens = (q_pos - neighbor_offsets.unsqueeze(2)) % L # [B, n_heads, L, P, W]

                s_expanded = s_heads.unsqueeze(3).unsqueeze(4).expand(B, n_heads, L, P_super, window_size, head_dim)
                idx_exp = target_tokens.unsqueeze(-1).expand(B, n_heads, L, P_super, window_size, head_dim)
                s_neighbors = torch.gather(s_expanded, dim=2, index=idx_exp) # [B, H, L, P, W, head_dim]

                w_exp = filter_weights.unsqueeze(2).unsqueeze(-1)
                s_super = (w_exp * s_neighbors).sum(dim=4) # [B, H, L, P, head_dim]

                # 6. Normalize and Project to K, V Super-Tokens
                s_super_norm = self.ln_kv(s_super)
                k_super = self.k_proj(s_super_norm)
                v_super = self.v_proj(s_super_norm)

                scores = (q.unsqueeze(3) * k_super).sum(dim=-1) / math.sqrt(head_dim) + all_peak_vals.unsqueeze(2)
                attn = F.softmax(scores, dim=-1)
                attn_out = (attn.unsqueeze(-1) * v_super).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
                attn_out = self.c_proj(attn_out)

                s = s + step_mult * attn_out
                s = s + step_mult * self.mlp(self.ln_2(s))
                if t < T_train - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            return self.head(self.ln_f(s)[:, 0])

    model = StateSuperTokenViT().to(device)
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
        print(f"  Epoch {epoch:>2d}/{epochs} ({time.time()-ep_start:.1f}s) | Train Loss: {train_loss/train_total:.4f}, Train Acc: {train_acc:>5.2f}%")

    total_time = time.time() - t_start

    # Evaluate
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

    top1 = (test_correct_top1 / test_total) * 100.0
    top5 = (test_correct_top5 / test_total) * 100.0
    avg_loss = test_loss / test_total

    print(f"\n✓ FINISHED Candidate 3 (State Super-Tokens P=8, W=5): Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | Loss: {avg_loss:.4f} | Time: {total_time:.1f}s")
    
    # Save checkpoint & results to volume
    os.makedirs("/models", exist_ok=True)
    torch.save(model.state_dict(), "/models/study72_3_state_super_p8_w5.pt")
    res = {"name": "State Super-Tokens (P=8, Win=5)", "raw_tokens": 40, "attn_ops": 8, "top1": top1, "top5": top5, "loss": avg_loss, "time": total_time}
    with open("/models/study72_3_results.json", "w") as f:
        json.dump(res, f)
    volume.commit()
    return res

@app.local_entrypoint()
def main():
    train_state_super_p8_w5.remote()
