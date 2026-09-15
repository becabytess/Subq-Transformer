"""
==========================================================================================
  STUDY S4-006 (LEAN): MULTI-STREAM CONSENSUS SUBQ (M=3, K=3, T=8)
==========================================================================================
Architectural Paradigm:
- M = 3 Completely Autonomous, Parallel SubQ Streams.
- Strictly NO parameter sharing across streams (independent Q, K, V, wave routers, MLPs).
- Conserved Attention Budget: K = 3 peaks per stream (1 anchor + 2 backward peaks).
  Total attention lookups across all streams: M * K * T = 3 * 3 * 8 = 72 (only 28% of Dense!).
- Lean Compute: d_mlp = 256 (2x expansion) for fast training and low memory footprint.
- Transport Protocol: Pure Optical Transport (state = context, NO W_o, NO additive residual
  during the 4-hop runway).
- Delayed MLP: Each stream fires its dedicated 2x MLP at T=4.
- Consensus Checkpoint (T=4): Zero-parameter average across all streams:
    s_common = (1/M) * sum(s^(m))
- Synchronized Phase 2 (T=5..8): All streams resume from s_common, run 4 more optical hops,
  and fire their respective MLPs at T=8.
- Final Merge (T=8): Final consensus average s_final = (1/M) * sum(s^(m)) -> LM Head.
- Evaluation tracks both Consensus PPL and Individual Stream PPLs in real time!
==========================================================================================
"""

import math
import time
import json
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# 1. Continuous Harmonic Wave Router (Strict K=3 Peaks, Learnable Carrier Latent)
# ------------------------------------------------------------------------------
class ContinuousHarmonicWaveRouter(nn.Module):
    def __init__(self, num_waves=12, K_peaks=3, max_d=128, n_heads=4):
        super().__init__()
        self.num_waves = num_waves
        self.K_peaks = K_peaks
        self.max_d = max_d
        self.n_heads = n_heads

        self.init_wave_latent = nn.Parameter(torch.randn(1, num_waves * 4) * 0.1)
        self.wave_transition = nn.Sequential(
            nn.Linear(num_waves * 4, 64),
            nn.GELU(),
            nn.Linear(64, num_waves * 4),
        )

        log_freqs = torch.linspace(0.0, -2.0, num_waves)
        self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
        self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

    def forward(self, wave_latent, B, dev):
        curr_params = wave_latent.view(1, self.num_waves, 4)
        amp = torch.tanh(curr_params[..., 0]).view(1, 1, 1, self.num_waves)
        omega = F.softplus(curr_params[..., 1]).view(1, 1, 1, self.num_waves) * self.base_freqs
        phi = curr_params[..., 2] * math.pi
        phi = phi.view(1, 1, 1, self.num_waves)
        decay = F.softplus(curr_params[..., 3]) * 0.05
        decay = decay.view(1, 1, 1, self.num_waves)

        # Synthesize continuous interference wave pattern
        wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
        wave_1d = wave_comps.sum(dim=-1).expand(B, 1, -1)

        # Select Top-(K-1) causal backward peak offsets (K=3 -> top 2 backward peaks)
        topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.K_peaks - 1, dim=-1)
        past_peak_offsets = past_peak_offsets + 1

        # Offset 0 is strictly anchor self-attention (always causal, always valid)
        zero_offset = torch.zeros((B, 1, 1), dtype=torch.long, device=dev)
        zero_val = torch.zeros((B, 1, 1), dtype=torch.float, device=dev)
        peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
        peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

        # Broadcast offsets and vals to all attention heads
        peak_offsets = peak_offsets.expand(B, self.n_heads, self.K_peaks)
        peak_vals = peak_vals.expand(B, self.n_heads, self.K_peaks)

        # Autonomous wave transition across hops
        next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
        return peak_offsets, peak_vals, next_wave_latent

# ------------------------------------------------------------------------------
# 2. Autonomous SubQ Stream (Clean Optical Transport, NO W_o, Delayed MLP)
# ------------------------------------------------------------------------------
class AutonomousSubQStream(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_mlp=256, num_waves=12, K_peaks=3, max_d=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.K_peaks = K_peaks
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Stream dedicated attention projections (COMPLETELY UNTIED ACROSS STREAMS)
        self.ln_attn = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        # Stream dedicated wave router
        self.router = ContinuousHarmonicWaveRouter(
            num_waves=num_waves, K_peaks=K_peaks, max_d=max_d, n_heads=n_heads
        )

        # Stream dedicated physical MLP (Lean 2x expansion)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

    def run_optical_hops(self, state, wave_latent, n_hops=4):
        B, L, D = state.shape
        dev = state.device
        q_pos = torch.arange(L, device=dev).view(1, 1, L, 1)

        for _ in range(n_hops):
            z = self.ln_attn(state)
            Q = self.q(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(z).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            peak_offsets, peak_vals, wave_latent = self.router(wave_latent, B, dev)

            # Strictly causal index gathering: target_pos = q_pos - peak_offsets
            target_indices = q_pos - peak_offsets.unsqueeze(2)  # [B, n_heads, L, K]
            valid_mask = target_indices >= 0
            target_clamped = torch.clamp(target_indices, min=0)

            idx_exp = target_clamped.unsqueeze(-1).expand(B, self.n_heads, L, self.K_peaks, self.d_k)
            K_gathered = torch.gather(K.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_exp)
            V_gathered = torch.gather(V.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_exp)

            scores = (Q.unsqueeze(3) * K_gathered).sum(dim=-1) * self.scale + peak_vals.unsqueeze(2)
            scores = scores.masked_fill(~valid_mask, -1e4)
            weights = F.softmax(scores, dim=-1) * valid_mask.float()
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

            context = (weights.unsqueeze(-1) * V_gathered).sum(dim=3)

            # Pure Optical Transport (Law of S4-005):
            # NO W_o projection, NO additive residual inside the 4-hop runway!
            state = context.transpose(1, 2).contiguous().view(B, L, D)

        # Delayed MLP: Fired only at the end of the 4-hop runway
        state = state + self.mlp(self.ln_mlp(state))
        return state, wave_latent

# ------------------------------------------------------------------------------
# 3. Multi-Stream Consensus SubQ Model (M=3, K=3, T=8)
# ------------------------------------------------------------------------------
class MultiStreamConsensusSubQLM(nn.Module):
    def __init__(self, vocab_size=65, seq_len=256, d_model=128, n_heads=4, d_mlp=256, num_streams=3, hops_per_phase=4, k_peaks=3):
        super().__init__()
        self.num_streams = num_streams
        self.hops_per_phase = hops_per_phase
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # M completely autonomous SubQ streams
        self.streams = nn.ModuleList([
            AutonomousSubQStream(d_model=d_model, n_heads=n_heads, d_mlp=d_mlp, K_peaks=k_peaks)
            for _ in range(num_streams)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, return_individual=False):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # PHASE 1: Independent Parallel Exploration (Hops 1 to 4 -> Dedicated MLPs)
        states_p1 = []
        latents_p1 = []
        for stream in self.streams:
            s_out, w_out = stream.run_optical_hops(x, stream.router.init_wave_latent, n_hops=self.hops_per_phase)
            states_p1.append(s_out)
            latents_p1.append(w_out)

        # CONSENSUS CHECKPOINT (T=4): Zero-Parameter Average Merge
        s_common = torch.stack(states_p1, dim=0).mean(dim=0)

        # PHASE 2: Synchronized Parallel Reasoning (Hops 5 to 8 -> Dedicated MLPs)
        states_p2 = []
        for i, stream in enumerate(self.streams):
            s_out, _ = stream.run_optical_hops(s_common, latents_p1[i], n_hops=self.hops_per_phase)
            states_p2.append(s_out)

        # FINAL CONSENSUS MERGE (T=8)
        s_final = torch.stack(states_p2, dim=0).mean(dim=0)
        logits = self.head(self.ln_f(s_final))

        if return_individual:
            individual_logits = [self.head(self.ln_f(s)) for s in states_p2]
            return logits, individual_logits

        return logits

# ------------------------------------------------------------------------------
# 4. Evaluation Routine (Consensus PPL vs Individual Stream PPLs)
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model, val_data, seq_len=256, batch_size=32, vocab_size=65, num_streams=3, num_batches=30, dev="cpu"):
    model.eval()
    total_loss_consensus = 0.0
    total_loss_streams = [0.0 for _ in range(num_streams)]
    total_tokens = 0

    for b in range(num_batches):
        torch.manual_seed(20260 + b)
        ix = torch.randint(len(val_data) - seq_len, (batch_size,))
        bx = torch.stack([val_data[i : i + seq_len] for i in ix]).to(dev)
        by = torch.stack([val_data[i + 1 : i + seq_len + 1] for i in ix]).to(dev)

        logits_consensus, indiv_logits = model(bx, return_individual=True)

        loss_c = F.cross_entropy(logits_consensus.view(-1, vocab_size), by.view(-1), reduction="sum")
        total_loss_consensus += loss_c.item()

        for s_idx, s_logits in enumerate(indiv_logits):
            loss_s = F.cross_entropy(s_logits.view(-1, vocab_size), by.view(-1), reduction="sum")
            total_loss_streams[s_idx] += loss_s.item()

        total_tokens += by.numel()

    avg_loss_c = total_loss_consensus / total_tokens
    ppl_c = math.exp(avg_loss_c)
    ppl_streams = [math.exp(l / total_tokens) for l in total_loss_streams]

    return avg_loss_c, ppl_c, ppl_streams

# ------------------------------------------------------------------------------
# 5. Main Training Routine
# ------------------------------------------------------------------------------
def main():
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("  EXPERIMENT S4-006 (LEAN): MULTI-STREAM CONSENSUS SUBQ (M=3, K=3, T=8)")
    print(f"  Compute Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)

    # Download Dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("Downloading TinyShakespeare dataset...")
    req = urllib.request.urlopen(url)
    text = req.read().decode("utf-8")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)

    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    print(f"Dataset Loaded: {len(data):,} chars | Vocab: {vocab_size} | Train: {len(train_data):,} | Val: {len(val_data):,}")

    seq_len = 256
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_mlp = 256        # Lean 2x MLP ratio
    k_peaks = 3        # Conserved attention budget (1 anchor + 2 backward peaks)
    num_streams = 3    # 3 Parallel streams
    hops_per_phase = 4 # 4 hops -> consensus -> 4 hops (Total T=8)
    total_steps = 2000
    eval_interval = 250
    val_batches = 30

    def get_batch(split: str, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i : i + seq_len] for i in ix])
        y = torch.stack([d[i + 1 : i + seq_len + 1] for i in ix])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    model = MultiStreamConsensusSubQLM(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_mlp=d_mlp,
        num_streams=num_streams,
        hops_per_phase=hops_per_phase,
        k_peaks=k_peaks,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Multi-Stream SubQ M={num_streams}, K={k_peaks}] Total Trainable Parameters: {n_params:,}")
    print("Architectural Breakdown:")
    print(f"  • {num_streams} Independent Streams x (QKV + WaveRouter(K={k_peaks}) + {d_mlp}-wide MLP)")
    print(f"  • Total lookups per token: M*K*T = {num_streams} * {k_peaks} * {hops_per_phase*2} = {num_streams*k_peaks*hops_per_phase*2} (only {round(100*num_streams*k_peaks*hops_per_phase*2/seq_len)}% of Dense!)")
    print(f"  • Consensus checkpoints: Simple average at T=4 and T=8")
    print("-" * 95)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)

    # Initial Pre-Training Evaluation
    val_loss, val_ppl, stream_ppls = evaluate_model(
        model, val_data, seq_len=seq_len, batch_size=batch_size,
        vocab_size=vocab_size, num_streams=num_streams, num_batches=10, dev=device
    )
    print(f"Step      0/{total_steps} | Init Val Loss: {val_loss:.4f} | Consensus PPL: {val_ppl:.2f}")
    print("=" * 95)

    history = []
    t0 = time.time()

    for step in range(1, total_steps + 1):
        model.train()
        bx, by = get_batch("train", step_seed=10000 + step)

        optimizer.zero_grad(set_to_none=True)
        logits = model(bx)
        loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0 or step == total_steps:
            elapsed = time.time() - t0
            val_loss, val_ppl, stream_ppls = evaluate_model(
                model, val_data, seq_len=seq_len, batch_size=batch_size,
                vocab_size=vocab_size, num_streams=num_streams, num_batches=val_batches, dev=device
            )

            streams_str = " | ".join([f"S{i+1}: {ppl:.2f}" for i, ppl in enumerate(stream_ppls)])

            print(
                f"  Step {step:4d}/{total_steps} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"CONSENSUS PPL: {val_ppl:5.2f} | "
                f"[{streams_str}] | "
                f"Time: {elapsed:5.1f}s"
            )

            history.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "consensus_ppl": round(val_ppl, 2),
                "stream_ppls": [round(p, 2) for p in stream_ppls],
                "elapsed_seconds": round(elapsed, 1),
            })

    print("=" * 95)
    print("TRAINING COMPLETE!")
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final CONSENSUS Perplexity: {val_ppl:.2f}")
    for i, ppl in enumerate(stream_ppls):
        gain = ppl - val_ppl
        print(f"  Stream {i+1} Perplexity: {ppl:.2f} (Consensus Boost: -{gain:+.2f} PPL)")
    print("=" * 95)

    result_data = {
        "experiment": "S4-006_lean",
        "architecture": "MultiStreamConsensusSubQLM_Lean",
        "num_streams": num_streams,
        "k_peaks": k_peaks,
        "d_mlp": d_mlp,
        "parameters": n_params,
        "final_val_loss": round(val_loss, 4),
        "final_consensus_ppl": round(val_ppl, 2),
        "final_stream_ppls": [round(p, 2) for p in stream_ppls],
        "history": history,
    }

    output_path = "s4_006_lean_multi_stream_results.json"
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Saved full experiment history to '{output_path}'")

if __name__ == "__main__":
    main()
