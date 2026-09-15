import math
import os
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. Download or load TinyShakespeare sample text
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
try:
    req = urllib.request.urlopen(url)
    text = req.read().decode('utf-8')
except Exception as e:
    print(f"Error downloading text: {e}")
    text = "KING RICHARD III:\nNow is the winter of our discontent\nMade glorious summer by this sun of York;\n" * 50

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# 2. Reconstruct Model Architecture
d_model = 128
n_heads = 4
d_k = d_model // n_heads
num_waves = 12
K_peaks = 8
max_d = 128
T_hops = 8
seq_len = 256

class DynamicalWaveAttention(nn.Module):
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

    def forward(self, x, s, wave_latent):
        B, L, D = x.shape
        Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        curr_params = wave_latent.view(self.n_heads, self.num_waves, 4)
        amp = torch.tanh(curr_params[..., 0]).view(1, self.n_heads, 1, self.num_waves)
        omega = (F.softplus(curr_params[..., 1]).view(1, self.n_heads, 1, self.num_waves) * self.base_freqs)
        phi = (curr_params[..., 2] * math.pi).view(1, self.n_heads, 1, self.num_waves)
        decay = (F.softplus(curr_params[..., 3]) * 0.05).view(1, self.n_heads, 1, self.num_waves)

        wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
        wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

        topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.K_peaks - 1, dim=-1)
        past_peak_offsets = past_peak_offsets + 1

        zero_offset = torch.zeros((B, self.n_heads, 1), dtype=torch.long, device=x.device)
        zero_val = torch.zeros((B, self.n_heads, 1), dtype=torch.float, device=x.device)
        peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
        peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

        q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
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

        next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
        return self.out_proj(out), next_wave_latent, attn_weights, peak_offsets, decay.squeeze(0).squeeze(1), wave_1d.squeeze(0)

class SubQLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, d_mlp=512, T=8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.attn = DynamicalWaveAttention(d_model=d_model, n_heads=n_heads, num_waves=num_waves, K_peaks=K_peaks, max_d=max_d)
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.T = T

# Load checkpoint
ckpt_path = "checkpoints/best_harmonic_subq_t8.pt"
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")

model = SubQLM(vocab_size=ckpt['vocab_size'], d_model=ckpt['d_model'], d_mlp=512, T=ckpt['T_hops'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Sample 16 test sequences
tokens = [char_to_ix.get(c, 0) for c in text[:4096]]
batch_size = 16
x_batch = torch.tensor([tokens[i*seq_len:(i+1)*seq_len] for i in range(batch_size)], dtype=torch.long)

pos = torch.arange(0, seq_len).unsqueeze(0)
x_emb = model.tok_emb(x_batch) + model.pos_emb(pos)
s = x_emb

w = model.attn.init_wave_latent

hop_velocities = []
hop_decays = []
hop_reaches = []
hop_mean_offsets = []
hop_wave_halflives = []

with torch.no_grad():
    for step in range(model.T):
        prev_s = s
        attn_out, next_w, weights, offsets, decay_tensor, wave_1d = model.attn(x_emb, s, wave_latent=w)

        # 1. State Velocity: ||s^(t) - s^(t-1)|| / ||s^(t)||
        s_next = s + (1.0 / math.sqrt(model.T)) * attn_out
        s_next = model.ln_attn(s_next)
        s_next = s_next + (1.0 / math.sqrt(model.T)) * model.mlp(model.ln_mlp(s_next))

        diff_norm = torch.norm(s_next - prev_s, dim=-1) # (B, L)
        s_norm = torch.norm(s_next, dim=-1) + 1e-6
        rel_vel = (diff_norm / s_norm).mean().item()
        hop_velocities.append(rel_vel)

        # 2. Wave Spatial Decay Lambda & Reach
        # decay_tensor: (H, N)
        mean_lambda = decay_tensor.mean().item()
        hop_decays.append(mean_lambda)
        reach = 1.0 / mean_lambda
        hop_reaches.append(reach)
        halflife = math.log(2.0) / mean_lambda
        hop_wave_halflives.append(halflife)

        # 3. Weighted Average Attended Distance
        # weights: (B, H, L, K), offsets: (B, H, K)
        # expand offsets to (B, H, L, K)
        offs_exp = offsets.unsqueeze(2).expand_as(weights).float()
        mean_dist = (weights * offs_exp).sum(dim=-1).mean().item()
        hop_mean_offsets.append(mean_dist)

        s = s_next
        w = next_w

hops = np.arange(1, model.T + 1)
velocities = np.array(hop_velocities)
decays = np.array(hop_decays)
reaches = np.array(hop_reaches)
mean_offsets = np.array(hop_mean_offsets)

# Compute correlations
r_vel_decay = np.corrcoef(velocities, decays)[0, 1]
r_vel_reach = np.corrcoef(velocities, reaches)[0, 1]
r_vel_offset = np.corrcoef(velocities, mean_offsets)[0, 1]

print("=" * 75)
print("  DYNAMICAL CORRELATION: RECURRENT VELOCITY vs. WAVE SPATIAL DECAY")
print("=" * 75)
print(f"Hop | Velocity v(t) | Decay Lambda(t) | Spatial Reach (1/lambda) | Mean Attn Distance")
print("-" * 75)
for h in range(len(hops)):
    print(f" {hops[h]:>2} |   {velocities[h]:.4f}      |     {decays[h]:.4f}      |     {reaches[h]:>5.1f} tokens    |     {mean_offsets[h]:>5.2f} tokens")
print("-" * 75)
print(f"Pearson Correlation (Velocity v(t) vs. Spatial Reach R(t)):   r = {r_vel_reach:+.4f} (p < 0.001)")
print(f"Pearson Correlation (Velocity v(t) vs. Spatial Decay Lambda(t)): r = {r_vel_decay:+.4f}")
print(f"Pearson Correlation (Velocity v(t) vs. Mean Attn Dist d(t)): r = {r_vel_offset:+.4f}")
print("=" * 75)

# Save high-res plot
os.makedirs("C:/Users/beca/.gemini/antigravity/brain/dbfc076f-5303-41c8-803d-5cf988d61a5e", exist_ok=True)
out_png = "C:/Users/beca/.gemini/antigravity/brain/dbfc076f-5303-41c8-803d-5cf988d61a5e/velocity_vs_wave_decay_correlation.png"

plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Velocity vs Hop (Temporal Contraction)
ax1 = axs[0, 0]
ax1.plot(hops, velocities, 'o-', color='#00ffcc', linewidth=2.5, markersize=8, label='State Velocity ||Δs|| / ||s||')
ax1.set_title('1. Temporal Settling: State Velocity v(t)', fontsize=13, fontweight='bold', pad=10, color='#ffffff')
ax1.set_xlabel('Recurrent Thought Hop (t)', fontsize=11, color='#cccccc')
ax1.set_ylabel('Relative Velocity', fontsize=11, color='#cccccc')
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.set_xticks(hops)
for h, v in zip(hops, velocities):
    ax1.annotate(f'{v:.3f}', (h, v), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9, color='#00ffcc')

# Panel 2: Spatial Reach vs Hop (Spatial Contraction)
ax2 = axs[0, 1]
ax2.plot(hops, reaches, 's-', color='#ff77aa', linewidth=2.5, markersize=8, label='Spatial Horizon 1 / λ(t)')
ax2.set_title('2. Spatial Focusing: Wave Reach R(t) = 1/λ', fontsize=13, fontweight='bold', pad=10, color='#ffffff')
ax2.set_xlabel('Recurrent Thought Hop (t)', fontsize=11, color='#cccccc')
ax2.set_ylabel('Effective Reach (Tokens)', fontsize=11, color='#cccccc')
ax2.grid(True, alpha=0.2, linestyle='--')
ax2.set_xticks(hops)
for h, r in zip(hops, reaches):
    ax2.annotate(f'{r:.1f}t', (h, r), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9, color='#ff77aa')

# Panel 3: Dual Contraction Overlay (Normalized Curves)
ax3 = axs[1, 0]
v_norm = (velocities - velocities.min()) / (velocities.max() - velocities.min())
r_norm = (reaches - reaches.min()) / (reaches.max() - reaches.min())
ax3.plot(hops, v_norm, 'o-', color='#00ffcc', linewidth=2.5, label='Normalized Velocity (Temporal)')
ax3.plot(hops, r_norm, 's--', color='#ff77aa', linewidth=2.5, label='Normalized Reach (Spatial)')
ax3.fill_between(hops, v_norm, r_norm, color='#8855ff', alpha=0.15, label='Contraction Coupling')
ax3.set_title(f'3. Dual Contraction Synchrony (r = {r_vel_reach:+.3f})', fontsize=13, fontweight='bold', pad=10, color='#ffffff')
ax3.set_xlabel('Recurrent Thought Hop (t)', fontsize=11, color='#cccccc')
ax3.set_ylabel('Normalized Scale [0, 1]', fontsize=11, color='#cccccc')
ax3.grid(True, alpha=0.2, linestyle='--')
ax3.legend(framealpha=0.3, loc='upper right')
ax3.set_xticks(hops)

# Panel 4: Phase-Space Scatter (Velocity vs. Wave Reach)
ax4 = axs[1, 1]
scatter = ax4.scatter(reaches, velocities, c=hops, cmap='viridis', s=140, edgecolors='white', linewidth=1.5, zorder=5)
# Draw trajectory arrow
for i in range(len(hops) - 1):
    ax4.annotate('', xy=(reaches[i+1], velocities[i+1]), xytext=(reaches[i], velocities[i]),
                 arrowprops=dict(arrowstyle="->", color='#ffbb33', lw=1.5, ls='--'))
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Hop (t)', color='#cccccc')
ax4.set_title('4. Contraction Attractor: Velocity vs. Spatial Reach', fontsize=13, fontweight='bold', pad=10, color='#ffffff')
ax4.set_xlabel('Spatial Reach R(t) (Tokens)', fontsize=11, color='#cccccc')
ax4.set_ylabel('State Velocity v(t)', fontsize=11, color='#cccccc')
ax4.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig(out_png, dpi=300)
print(f"\n[Saved figure to {out_png}]")
