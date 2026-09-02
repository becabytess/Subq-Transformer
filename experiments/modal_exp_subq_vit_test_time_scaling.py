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

app = modal.App("exp-subq-vit-test-time-scaling", image=image)

# Persistent volume to save the trained model checkpoint
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)

@app.function(gpu="A10G", timeout=7200, volumes={"/models": volume})
def run_test_time_scaling_experiment():
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
    print("  STUDY 69: TEST-TIME THINKING COMPUTE SCALING IN HARMONIC SUBQ ViT")
    print("  Training at T_train = 4 Hops -> Evaluating Zero-Shot Extrapolation at T_eval in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]")
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
    K_peaks = 8
    num_waves = 12
    epochs = 20
    T_train = 4

    print(f"High-Res ViT Config: Patch Size {patch_size}x{patch_size} -> {num_patches} patches (Seq Len L = {seq_len}), Dim={d_model}, Heads={n_heads}")

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
    # 3. Harmonic SubQ ViT with Arbitrary Test-Time T Support
    # -------------------------------------------------------------------------
    class HarmonicSubQViT(nn.Module):
        def __init__(self, T_train=4):
            super().__init__()
            self.T_train = T_train
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

        def forward(self, x, T=None, step_scaling="fixed_train"):
            """
            T: Number of unrolled hops (defaults to T_train).
            step_scaling:
              - 'fixed_train': residual multiplier = 1 / sqrt(T_train) (preserves learned step magnitude)
              - 'adaptive': residual multiplier = 1 / sqrt(T) (adjusts scale dynamically)
            """
            num_hops = self.T_train if T is None else T
            B = x.shape[0]
            s = self.patch_embed(x)
            L = s.shape[1]

            curr_wave = self.init_wave_latent
            if step_scaling == "fixed_train":
                step_mult = 1.0 / math.sqrt(self.T_train)
            else:
                step_mult = 1.0 / math.sqrt(num_hops)

            state_deltas = []

            for t in range(num_hops):
                s_prev = s
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

                # 3. Extract Top-K Peak Offsets
                topk_vals, past_peak_offsets = torch.topk(wave_1d, k=K_peaks - 1, dim=-1)
                past_peak_offsets = past_peak_offsets + 1
                zero_offset = torch.zeros((B, n_heads, 1), dtype=torch.long, device=x.device)
                zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=x.device)
                peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
                peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

                # 4. Vectorized Gather Keys & Values
                q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
                target_indices = (q_pos - peak_offsets.unsqueeze(2)) % L
                idx_exp = target_indices.unsqueeze(-1).expand(B, n_heads, L, K_peaks, head_dim)

                K_g = torch.gather(k.unsqueeze(3).expand(B, n_heads, L, K_peaks, head_dim), dim=2, index=idx_exp)
                V_g = torch.gather(v.unsqueeze(3).expand(B, n_heads, L, K_peaks, head_dim), dim=2, index=idx_exp)

                # 5. Attention with Harmonic Logit Bias
                scores = (q.unsqueeze(3) * K_g).sum(dim=-1) / math.sqrt(head_dim) + peak_vals.unsqueeze(2)
                attn = F.softmax(scores, dim=-1)
                attn_out = (attn.unsqueeze(-1) * V_g).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
                attn_out = self.c_proj(attn_out)

                # 6. Recurrent Residual Update
                s = s + step_mult * attn_out
                s = s + step_mult * self.mlp(self.ln_2(s))

                delta = torch.norm(s - s_prev, dim=-1).mean().item()
                state_deltas.append(delta)

                # 7. Wave Transition
                if t < num_hops - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            s_final = self.ln_f(s)
            cls_out = s_final[:, 0]
            logits = self.head(cls_out)
            return logits, state_deltas

    # -------------------------------------------------------------------------
    # 4. Train Model with T_train = 4 Hops
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    model = HarmonicSubQViT(T_train=T_train).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [STAGE 1] TRAINING HARMONIC SUBQ ViT (T_train = {T_train} Hops, Params: {param_count:,}, Epochs: {epochs}, L = {seq_len})")
    print("=" * 110)

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    t_train_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        ep_start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs, _ = model(images, T=T_train)
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

    total_train_time = time.time() - t_train_start
    print(f"  --> Finished Training in {total_train_time:.1f}s! Saving checkpoint to /models/harmonic_subq_vit_T4.pt...")
    os.makedirs("/models", exist_ok=True)
    torch.save(model.state_dict(), "/models/harmonic_subq_vit_T4.pt")
    volume.commit()
    print("  --> Model Checkpoint Successfully Saved to Persistent Volume!")

    # -------------------------------------------------------------------------
    # 5. [STAGE 2] TEST-TIME THINKING COMPUTE SWEEP (T_eval in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32])
    # -------------------------------------------------------------------------
    print("\n" + "=" * 135)
    print("  [STAGE 2] ZERO-SHOT TEST-TIME THINKING COMPUTE SWEEP ACROSS T_eval in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]")
    print("  Evaluating identical trained weights across both Fixed Step Scaling (1/sqrt(4)) & Adaptive Step Scaling (1/sqrt(T_eval))")
    print("=" * 135)

    T_eval_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]

    def evaluate_model_at_T(eval_T, scaling_mode):
        model.eval()
        test_loss, test_correct_top1, test_correct_top5, test_total = 0.0, 0, 0, 0
        deltas_sum = None
        batches = 0
        t_eval_start = time.time()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs, deltas = model(images, T=eval_T, step_scaling=scaling_mode)
                    loss = F.cross_entropy(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, top1 = outputs.max(1)
                test_correct_top1 += top1.eq(labels).sum().item()
                _, top5 = outputs.topk(5, dim=1)
                test_correct_top5 += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
                test_total += images.size(0)

                if deltas_sum is None:
                    deltas_sum = [0.0] * len(deltas)
                for i_d, d_val in enumerate(deltas):
                    deltas_sum[i_d] += d_val
                batches += 1

        eval_time = time.time() - t_eval_start
        acc1 = (test_correct_top1 / test_total) * 100.0
        acc5 = (test_correct_top5 / test_total) * 100.0
        avg_loss = test_loss / test_total
        avg_deltas = [d / batches for d in deltas_sum]
        final_delta = avg_deltas[-1] if avg_deltas else 0.0
        return {
            "T_eval": eval_T,
            "scaling": scaling_mode,
            "top1": acc1,
            "top5": acc5,
            "loss": avg_loss,
            "eval_time": eval_time,
            "final_delta": final_delta,
            "is_trained_T": (eval_T == T_train)
        }

    results_fixed = []
    print("\n--- Running Evaluation Mode 1: Fixed Step Scaling (step_mult = 1/sqrt(4) = 0.5) ---")
    for t_eval in T_eval_list:
        res = evaluate_model_at_T(t_eval, "fixed_train")
        tag = " <-- TRAINED TARGET" if res["is_trained_T"] else (" (Under-thinking)" if t_eval < T_train else " (Deep Extrapolation)")
        print(f"  T_eval = {t_eval:>2d} | Top-1: {res['top1']:>5.2f}% | Top-5: {res['top5']:>5.2f}% | Loss: {res['loss']:.4f} | Final Delta: {res['final_delta']:.4f} | Eval Time: {res['eval_time']:>4.1f}s{tag}")
        results_fixed.append(res)

    results_adaptive = []
    print("\n--- Running Evaluation Mode 2: Adaptive Step Scaling (step_mult = 1/sqrt(T_eval)) ---")
    for t_eval in T_eval_list:
        res = evaluate_model_at_T(t_eval, "adaptive")
        tag = " <-- TRAINED TARGET" if res["is_trained_T"] else (" (Under-thinking)" if t_eval < T_train else " (Deep Extrapolation)")
        print(f"  T_eval = {t_eval:>2d} | Top-1: {res['top1']:>5.2f}% | Top-5: {res['top5']:>5.2f}% | Loss: {res['loss']:.4f} | Final Delta: {res['final_delta']:.4f} | Eval Time: {res['eval_time']:>4.1f}s{tag}")
        results_adaptive.append(res)

    # -------------------------------------------------------------------------
    # 6. Final Comprehensive Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 145)
    print(f"  STUDY 69 FINAL RESULTS: ZERO-SHOT TEST-TIME THINKING COMPUTE SCALING (T_train = {T_train} Hops, L = {seq_len})")
    print("=" * 145)
    print(f"{'T_eval Hops':<12} | {'Regime':<22} | {'Fixed Top-1':<12} | {'Fixed Top-5':<12} | {'Fixed Loss':<12} | {'Adaptive Top-1':<15} | {'Adaptive Top-5':<15} | {'Eval Time':<10}")
    print("-" * 145)
    for rf, ra in zip(results_fixed, results_adaptive):
        t_val = rf["T_eval"]
        if t_val < T_train:
            regime = "Under-thinking"
        elif t_val == T_train:
            regime = "**TRAINED TARGET (T=4)**"
        elif t_val <= 8:
            regime = "Near Extrapolation"
        else:
            regime = "Deep Extrapolation"

        print(f"{t_val:<12} | {regime:<22} | {rf['top1']:<10.2f}% | {rf['top5']:<10.2f}% | {rf['loss']:<12.4f} | {ra['top1']:<13.2f}% | {ra['top5']:<13.2f}% | {rf['eval_time']:>6.1f}s")
    print("=" * 145)

    return {"fixed": results_fixed, "adaptive": results_adaptive}

@app.local_entrypoint()
def main():
    run_test_time_scaling_experiment.remote()
