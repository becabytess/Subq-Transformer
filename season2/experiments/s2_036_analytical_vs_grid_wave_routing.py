import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "matplotlib"
    )
)

app = modal.App("exp-s2-036-analytical-vs-grid-wave", image=image)

@app.function(gpu="A10G", timeout=2400)
def run_analytical_vs_grid_experiment():
    import math
    import time
    import urllib.request
    import json
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 115)
    print("  STUDY S2-036: RIGOROUS CONTROLLED BENCHMARK (3,000 STEPS, T=8, NO WEIGHT TYING)")
    print("  100% Matched to S2-035 Canonical Baseline: Grid-Search vs. Analytical Deformable vs. Fixed Basis")
    print(f"  Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")
    print("=" * 115)

    # 1. Download TinyShakespeare Dataset (Identical to S2-035)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.urlopen(url)
    text = req.read().decode('utf-8')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    print(f"TinyShakespeare: {len(data):,} Characters | Vocab: {vocab_size}")

    seq_len = 256
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_k = d_model // n_heads
    d_mlp = 512
    num_waves = 12
    K_peaks = 8
    max_d = 128
    T_hops = 8      # Matched to S2-035 (T=8)
    n_steps = 3000  # Trained longer as requested (3,000 steps)

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # =========================================================================
    # Model 1: Exact S2-035 Grid-Search Waves (12 waves, max_d=128, torch.topk)
    # =========================================================================
    class GridSearchWaveAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4, num_waves=12, K_peaks=8, max_d=128):
            super().__init__()
            self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
            self.num_waves, self.K_peaks, self.max_d = num_waves, K_peaks, max_d

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.init_wave_latent = nn.Parameter(torch.randn(n_heads, num_waves * 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 64),
                nn.GELU(),
                nn.Linear(64, num_waves * 4)
            )

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

        def get_offsets_and_bias(self, wave_latent, B, device):
            curr_params = wave_latent.view(self.n_heads, self.num_waves, 4)
            amp = torch.tanh(curr_params[..., 0]).view(1, self.n_heads, 1, self.num_waves)
            omega = (F.softplus(curr_params[..., 1]).view(1, self.n_heads, 1, self.num_waves) * self.base_freqs)
            phi = (curr_params[..., 2] * math.pi).view(1, self.n_heads, 1, self.num_waves)
            decay = (F.softplus(curr_params[..., 3]) * 0.05).view(1, self.n_heads, 1, self.num_waves)

            wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
            wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

            topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.K_peaks - 1, dim=-1)
            past_peak_offsets = past_peak_offsets + 1

            zero_offset = torch.zeros((B, self.n_heads, 1), dtype=torch.long, device=device)
            zero_val = torch.zeros((B, self.n_heads, 1), dtype=torch.float, device=device)
            peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
            peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

            next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
            return peak_offsets, peak_vals, next_wave_latent

        def forward(self, x, s, wave_latent=None):
            B, L, D = x.shape
            device = x.device
            if wave_latent is None:
                wave_latent = self.init_wave_latent
            peak_offsets, peak_vals, next_wave_latent = self.get_offsets_and_bias(wave_latent, B, device)

            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
            target_indices = q_pos - peak_offsets.unsqueeze(2)
            valid_mask = target_indices >= 0
            target_indices_clamped = torch.clamp(target_indices, min=0)

            idx_expanded = target_indices_clamped.unsqueeze(-1).expand(B, self.n_heads, L, self.K_peaks, self.d_k)
            K_gathered = torch.gather(K.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_expanded)
            V_gathered = torch.gather(V.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_expanded)

            Q_exp = Q.unsqueeze(3)
            scores = (Q_exp * K_gathered).sum(dim=-1) / math.sqrt(self.d_k) + peak_vals.unsqueeze(2)
            scores = scores.masked_fill(~valid_mask, -1e4)
            attn_weights = F.softmax(scores, dim=-1) * valid_mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            out = (attn_weights.unsqueeze(-1) * V_gathered).sum(dim=3)
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(out), next_wave_latent

    # =========================================================================
    # Model 2: Analytical Deformable Waves (Continuous math + STE, K=8)
    # =========================================================================
    class AnalyticalDeformableWaveAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4, K_peaks=8):
            super().__init__()
            self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
            self.K_peaks = K_peaks
            self.num_carriers = K_peaks - 1  # 7 carrier slots for past offsets

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            # 7 octave base periods spanning from local (1, 2, 4) to long-range (16, 64, 128)
            base_periods = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 128.0])
            base_omegas = 2.0 * math.pi / base_periods
            self.register_buffer("base_periods", base_periods.view(1, self.num_carriers))
            self.register_buffer("base_omegas", base_omegas.view(1, self.num_carriers))

            # 4 parameters per carrier: [amp, delta_omega, phi, decay]
            self.init_wave_latent = nn.Parameter(torch.randn(n_heads, self.num_carriers * 4) * 0.05)
            self.wave_transition = nn.Sequential(
                nn.Linear(self.num_carriers * 4, 32),
                nn.GELU(),
                nn.Linear(32, self.num_carriers * 4)
            )

        def get_offsets_and_bias(self, wave_latent, B, L, device):
            curr_params = wave_latent.view(self.n_heads, self.num_carriers, 4)
            amp = torch.tanh(curr_params[..., 0])  # (H, M)
            delta_omega = torch.tanh(curr_params[..., 1]) * 0.5  # (H, M)
            phi = curr_params[..., 2] * math.pi  # (H, M)
            decay = F.softplus(curr_params[..., 3]) * 0.05  # (H, M)

            omega = self.base_omegas * torch.exp(delta_omega)
            x_continuous = (2.0 * math.pi - phi) / omega
            x_clamped = torch.clamp(x_continuous, min=1.0, max=float(L - 1))

            # Straight-Through Estimator (STE) for discrete integer snapping
            d_int = torch.round(x_clamped)
            d_ste = x_clamped + (d_int - x_clamped).detach()
            past_offsets = d_ste.long()  # (H, M)

            past_bias = amp * torch.exp(-decay * x_clamped)  # (H, M)

            zero_off = torch.zeros((self.n_heads, 1), dtype=torch.long, device=device)
            zero_bias = torch.zeros((self.n_heads, 1), dtype=torch.float, device=device)
            offsets = torch.cat([zero_off, past_offsets], dim=-1)  # (H, K)
            biases = torch.cat([zero_bias, past_bias], dim=-1)    # (H, K)

            next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
            return offsets, biases, x_clamped, next_wave_latent

        def forward(self, x, s, wave_latent=None):
            B, L, D = x.shape
            device = x.device
            if wave_latent is None:
                wave_latent = self.init_wave_latent
            offsets, biases, _, next_wave_latent = self.get_offsets_and_bias(wave_latent, B, L, device)

            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
            target_indices = q_pos - offsets.view(1, self.n_heads, 1, self.K_peaks)
            valid_mask = target_indices >= 0
            target_clamped = torch.clamp(target_indices, min=0)

            idx_exp = target_clamped.unsqueeze(-1).expand(B, self.n_heads, L, self.K_peaks, self.d_k)
            K_gathered = torch.gather(K.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_exp)
            V_gathered = torch.gather(V.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_exp)

            Q_exp = Q.unsqueeze(3)
            scores = (Q_exp * K_gathered).sum(dim=-1) / math.sqrt(self.d_k) + biases.view(1, self.n_heads, 1, self.K_peaks)
            scores = scores.masked_fill(~valid_mask, -1e4)
            attn_weights = F.softmax(scores, dim=-1) * valid_mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            out = (attn_weights.unsqueeze(-1) * V_gathered).sum(dim=3)
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(out), next_wave_latent

    # =========================================================================
    # Model 3: Fixed Complete Basis [0, 1, 2, 4, 8, 16, 63, 128]
    # =========================================================================
    class FixedBasisAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4):
            super().__init__()
            self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
            self.offsets = [0, 1, 2, 4, 8, 16, 63, 128]
            self.K_peaks = len(self.offsets)

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x, s, wave_latent=None):
            B, L, D = x.shape
            device = x.device
            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            offsets = torch.tensor(self.offsets, dtype=torch.long, device=device)
            q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
            target_indices = q_pos - offsets.view(1, 1, 1, self.K_peaks)
            valid_mask = target_indices >= 0
            target_clamped = torch.clamp(target_indices, min=0)

            idx_exp = target_clamped.unsqueeze(-1).expand(B, self.n_heads, L, self.K_peaks, self.d_k)
            K_gathered = torch.gather(K.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_exp)
            V_gathered = torch.gather(V.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_exp)

            Q_exp = Q.unsqueeze(3)
            scores = (Q_exp * K_gathered).sum(dim=-1) / math.sqrt(self.d_k)
            scores = scores.masked_fill(~valid_mask, -1e4)
            attn_weights = F.softmax(scores, dim=-1) * valid_mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            out = (attn_weights.unsqueeze(-1) * V_gathered).sum(dim=3)
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(out), None

    # =========================================================================
    # Unified Model Wrapper: 100% Matched to S2-035 SubQ_NoInLoopNonLinearity
    # =========================================================================
    class SubQLM(nn.Module):
        def __init__(self, attn_module, vocab_size=65, d_model=128, d_mlp=512, T=8):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.attn = attn_module
            self.ln_attn = nn.LayerNorm(d_model)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)  # NO WEIGHT TYING (Matched to S2-035)
            self.T = T

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            s = x

            w = getattr(self.attn, "init_wave_latent", None)
            inv_sqrt_T = 1.0 / math.sqrt(self.T)
            for step in range(self.T):
                z = self.ln_attn(s)
                attn_out, w = self.attn(x, z, wave_latent=w)
                s = s + inv_sqrt_T * attn_out

            s = s + self.mlp(self.ln_mlp(s))
            s = self.ln_f(s)
            logits = self.head(s)
            return logits

    # =========================================================================
    # Training & Evaluation Loop (3,000 Steps, Matched to S2-035)
    # =========================================================================
    def train_and_eval(name, model_fn):
        print(f"\n{'='*40} Training: {name} {'='*40}")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()

        model = model_fn().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {name} | Parameters: {n_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-4)

        t0 = time.time()
        eval_history = []

        for step in range(1, n_steps + 1):
            model.train()
            bx, by = get_batch('train', step_seed=10000 + step)
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            if step % 500 == 0 or step == n_steps:
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for v_step in range(20):
                        vx, vy = get_batch('val', step_seed=90000 + v_step)
                        v_logits = model(vx)
                        v_loss = F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1))
                        val_losses.append(v_loss.item())
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    val_ppl = math.exp(avg_val_loss)
                    elapsed = time.time() - t0
                    print(f"  Step {step:>4}/{n_steps} | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f} | Val PPL: {val_ppl:>6.2f} | Time: {elapsed:.1f}s")
                    eval_history.append({
                        "step": step,
                        "train_loss": round(loss.item(), 4),
                        "val_loss": round(avg_val_loss, 4),
                        "val_ppl": round(val_ppl, 2),
                        "elapsed_s": round(elapsed, 1)
                    })

        total_time = time.time() - t0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        throughput = (n_steps * batch_size * seq_len) / total_time
        print(f"  Final Val PPL: {val_ppl:.2f} | Total Time: {total_time:.1f}s | Throughput: {throughput:,.0f} tok/s | Peak VRAM: {peak_vram_mb:.1f} MB")

        return {
            "name": name,
            "params": n_params,
            "final_val_loss": round(avg_val_loss, 4),
            "final_val_ppl": round(val_ppl, 2),
            "total_time_s": round(total_time, 1),
            "throughput_tok_s": round(throughput, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "eval_history": eval_history,
            "model": model
        }

    # Execute all 3 runs (Exact matched parameters & seeds)
    res_grid = train_and_eval("grid_search_waves", lambda: SubQLM(GridSearchWaveAttention(d_model=d_model, n_heads=n_heads, num_waves=num_waves, K_peaks=K_peaks, max_d=max_d), vocab_size=vocab_size, d_model=d_model, d_mlp=d_mlp, T=T_hops))
    res_analytical = train_and_eval("analytical_deformable_waves", lambda: SubQLM(AnalyticalDeformableWaveAttention(d_model=d_model, n_heads=n_heads, K_peaks=K_peaks), vocab_size=vocab_size, d_model=d_model, d_mlp=d_mlp, T=T_hops))
    res_fixed = train_and_eval("fixed_complete_basis", lambda: SubQLM(FixedBasisAttention(d_model=d_model, n_heads=n_heads), vocab_size=vocab_size, d_model=d_model, d_mlp=d_mlp, T=T_hops))

    # =========================================================================
    # Learned Pattern & Offset Diagnostics Across Hops
    # =========================================================================
    print("\n" + "=" * 115)
    print("  DETAILED LEARNED OFFSET PATTERNS ACROSS HEADS & HOPS")
    print("=" * 115)

    # 1. Profile Grid Search Router
    print("\n>>> [Grid-Search Router: Discovered Wave Peaks Across Hops T=1..8]")
    grid_attn = res_grid["model"].attn
    grid_latent = grid_attn.init_wave_latent
    grid_patterns = {}
    for hop in range(1, T_hops + 1):
        peak_offsets, peak_vals, next_latent = grid_attn.get_offsets_and_bias(grid_latent, 1, device)
        grid_latent = next_latent
        grid_patterns[f"hop_{hop}"] = {}
        if hop in [1, 2, 4, 8]:
            print(f"\n--- Hop {hop} (Grid-Search Router) ---")
            for h in range(n_heads):
                int_offs = peak_offsets[0, h].detach().cpu().tolist()
                grid_biases = [round(v, 3) for v in peak_vals[0, h].detach().cpu().tolist()]
                grid_patterns[f"hop_{hop}"][f"head_{h}"] = {"integer_offsets": int_offs, "prior_biases": grid_biases}
                print(f"  Head {h}: Top-K Offsets = {int_offs} | Biases = {grid_biases}")

    # 2. Profile Analytical Deformable Router
    print("\n>>> [Analytical Deformable Router: Learned Offsets Across Hops T=1..8]")
    ana_attn = res_analytical["model"].attn
    ana_latent = ana_attn.init_wave_latent
    analytical_patterns = {}
    for hop in range(1, T_hops + 1):
        offsets, biases, x_cont, next_latent = ana_attn.get_offsets_and_bias(ana_latent, 1, seq_len, device)
        ana_latent = next_latent
        analytical_patterns[f"hop_{hop}"] = {}
        if hop in [1, 2, 4, 8]:
            print(f"\n--- Hop {hop} (Analytical Deformable Router) ---")
            for h in range(n_heads):
                int_offs = offsets[h].detach().cpu().tolist()
                cont_offs = [round(v, 2) for v in x_cont[h].detach().cpu().tolist()]
                bias_vals = [round(v, 3) for v in biases[h].detach().cpu().tolist()]
                analytical_patterns[f"hop_{hop}"][f"head_{h}"] = {
                    "integer_offsets": int_offs,
                    "continuous_desired": cont_offs,
                    "prior_biases": bias_vals
                }
                print(f"  Head {h}: Integer Offsets = {int_offs}")
                print(f"          Continuous Desired = [0.0] + {cont_offs}")
                print(f"          Spatial Biases     = {bias_vals}")

    # Summary Scorecard
    print("\n" + "=" * 115)
    print("  FINAL MASTER SCORECARD: STUDY S2-036 (3,000 STEPS, T=8)")
    print("=" * 115)
    print(f"  {'Model Architecture':<32} | {'Val Loss':<10} | {'Val PPL':<10} | {'Time (s)':<10} | {'Throughput':<16} | {'Peak VRAM':<10}")
    print(f"  {'-'*32}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*16}-+-{'-'*10}")
    for res in [res_grid, res_analytical, res_fixed]:
        print(f"  {res['name']:<32} | {res['final_val_loss']:<10.4f} | {res['final_val_ppl']:<10.2f} | {res['total_time_s']:<10.1f} | {res['throughput_tok_s']:<14,.0f} tok/s | {res['peak_vram_mb']:<7.1f} MB")
    print("=" * 115)

    return {
        "study": "S2-036",
        "benchmark": "TinyShakespeare Character LM (Controlled 3,000 Steps, T=8)",
        "seq_len": seq_len,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "T_hops": T_hops,
        "n_steps": n_steps,
        "results": {
            "grid_search_waves": {
                "val_loss": res_grid["final_val_loss"],
                "val_ppl": res_grid["final_val_ppl"],
                "total_time_s": res_grid["total_time_s"],
                "throughput_tok_s": res_grid["throughput_tok_s"],
                "peak_vram_mb": res_grid["peak_vram_mb"],
                "eval_history": res_grid["eval_history"],
                "patterns": grid_patterns
            },
            "analytical_deformable_waves": {
                "val_loss": res_analytical["final_val_loss"],
                "val_ppl": res_analytical["final_val_ppl"],
                "total_time_s": res_analytical["total_time_s"],
                "throughput_tok_s": res_analytical["throughput_tok_s"],
                "peak_vram_mb": res_analytical["peak_vram_mb"],
                "eval_history": res_analytical["eval_history"],
                "patterns": analytical_patterns
            },
            "fixed_complete_basis": {
                "val_loss": res_fixed["final_val_loss"],
                "val_ppl": res_fixed["final_val_ppl"],
                "total_time_s": res_fixed["total_time_s"],
                "throughput_tok_s": res_fixed["throughput_tok_s"],
                "peak_vram_mb": res_fixed["peak_vram_mb"],
                "eval_history": res_fixed["eval_history"]
            }
        }
    }

@app.local_entrypoint()
def main():
    import json
    res = run_analytical_vs_grid_experiment.remote()
    out_dir = os.path.join("season2", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "s2_036_analytical_vs_grid_wave_routing.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[Saved full structured results to {out_file}]")
