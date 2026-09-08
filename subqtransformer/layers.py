"""
Core Neural Layers for SubQTransformer:
- SubQSurfer: Multi-scale sparse graph constructor with evolving QKV, contractive linear residuals,
              and support for both Complete Basis and Continuous Harmonic Wave routing.
- SubQBlock: High-throughput Transformer block combining SubQSurfer with configurable MLP scheduling
             (executed at block end by default, or interleaved every X hops).
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from subqtransformer.config import SubQConfig


class SubQSurfer(nn.Module):
    """
    Sub-Quadratic Multi-Scale Attention Engine.
    
    Features:
    1. Sparse Candidate Routing O(L * K): Evaluates strictly K candidate positions per query.
    2. Routing Modes:
       - 'complete': Fixed dyadic multi-scale complete basis [0, 1, 2, 4, 8, 16, 63, 127, 128].
       - 'harmonic': Learned continuous Fourier carrier waves with dynamical hop transitions and Top-K peaks.
       - 'custom': User-defined discrete relative offset jump list.
    3. Causal and Bidirectional Support: Causal (i - d) or Bidirectional (i ± d).
    4. Evolving Q, K, V: Projects keys, queries, and values from accumulated state at each hop.
    5. Contractive Linear Attention Residual: state <- state + (1 / sqrt(T)) * W_o(context).
    6. Adaptive Dynamical Halting: Per-token early exiting via relative velocity convergence.
    """
    def __init__(self, config: SubQConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        assert self.head_dim * config.n_heads == config.d_model, "d_model must be divisible by n_heads"

        self.routing_mode = config.routing_mode
        self.is_causal = config.is_causal
        self.default_T = config.default_T
        self.max_seq_len = config.max_seq_len
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Mode-specific setup
        if self.routing_mode == "harmonic":
            self.num_waves = config.num_waves
            self.K_peaks = config.K_peaks
            self.max_d = min(config.max_seq_len, 256)

            self.init_wave_latent = nn.Parameter(torch.randn(self.n_heads, self.num_waves * 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(self.num_waves * 4, 64),
                nn.GELU(),
                nn.Linear(64, self.num_waves * 4)
            )

            log_freqs = torch.linspace(0.0, -2.0, self.num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, self.num_waves))
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, 1, self.max_d - 1, 1))
            self.K = self.K_peaks
        else:
            self.jump_offsets = config.jump_offsets or [0, 1, 2, 4, 8, 16, 32, 64]
            # If bidirectional, expand offsets to include negative offsets
            if not self.is_causal:
                pos_offsets = [o for o in self.jump_offsets if o > 0]
                neg_offsets = [-o for o in pos_offsets]
                self.effective_offsets = sorted(list(set([0] + pos_offsets + neg_offsets)))
            else:
                self.effective_offsets = sorted(list(set(self.jump_offsets)))
            self.K = len(self.effective_offsets)
            self._build_index_buffers(config.max_seq_len)

    def _build_index_buffers(self, seq_len: int):
        target_indices = torch.zeros((seq_len, self.K), dtype=torch.long)
        valid_mask = torch.zeros((seq_len, self.K), dtype=torch.bool)
        for i in range(seq_len):
            for k_idx, offset in enumerate(self.effective_offsets):
                target_pos = i - offset
                if 0 <= target_pos < seq_len:
                    target_indices[i, k_idx] = target_pos
                    valid_mask[i, k_idx] = True
                else:
                    target_indices[i, k_idx] = 0
                    valid_mask[i, k_idx] = False

        self.register_buffer("target_indices", target_indices, persistent=False)
        self.register_buffer("valid_mask", valid_mask, persistent=False)

    def compute_wave_offsets(self, wave_latent: torch.Tensor, B: int, device: torch.device):
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

    def single_hop_attention(
        self,
        s: torch.Tensor,
        wave_latent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes one single attention hop from evolving state s."""
        B, L, D = s.shape
        device = s.device

        # 1. Project evolving Q, K, V
        Q = self.q_proj(s).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d_k)
        K = self.k_proj(s).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(s).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if self.routing_mode == "harmonic":
            if wave_latent is None:
                wave_latent = self.init_wave_latent
            peak_offsets, peak_vals, next_wave_latent = self.compute_wave_offsets(wave_latent, B, device)

            q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
            if not self.is_causal:
                # In bidirectional mode, candidates extend both backward and forward
                past_offsets = peak_offsets  # [0, d1, d2, ...]
                future_offsets = -peak_offsets[..., 1:]  # [-d1, -d2, ...]
                all_offsets = torch.cat([past_offsets, future_offsets], dim=-1)
                all_vals = torch.cat([peak_vals, peak_vals[..., 1:]], dim=-1)
                target_indices = q_pos - all_offsets.unsqueeze(2)
                curr_K = all_offsets.size(-1)
            else:
                target_indices = q_pos - peak_offsets.unsqueeze(2)
                all_vals = peak_vals
                curr_K = self.K_peaks

            valid_mask = (target_indices >= 0) & (target_indices < L)
            target_clamped = torch.clamp(target_indices, min=0, max=L - 1)

            idx_exp = target_clamped.unsqueeze(-1).expand(B, self.n_heads, L, curr_K, self.head_dim)
            K_gathered = torch.gather(K.unsqueeze(3).expand(B, self.n_heads, L, curr_K, self.head_dim), dim=2, index=idx_exp)
            V_gathered = torch.gather(V.unsqueeze(3).expand(B, self.n_heads, L, curr_K, self.head_dim), dim=2, index=idx_exp)

            Q_exp = Q.unsqueeze(3)
            scores = (Q_exp * K_gathered).sum(dim=-1) * self.scale + all_vals.unsqueeze(2)
            scores = scores.masked_fill(~valid_mask, float("-inf"))
            weights = F.softmax(scores, dim=-1) * valid_mask.float()
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            weights = self.attn_dropout(weights)

            context = (weights.unsqueeze(-1) * V_gathered).sum(dim=3)
            context = context.transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(context), next_wave_latent
        else:
            if L > self.target_indices.size(0):
                self._build_index_buffers(L)
                self.target_indices = self.target_indices.to(device)
                self.valid_mask = self.valid_mask.to(device)

            indices = self.target_indices[:L]
            valid = self.valid_mask[:L] & (indices < L) & (indices >= 0)
            clamped_indices = torch.clamp(indices, 0, L - 1)
            mask = valid.unsqueeze(0).unsqueeze(1)  # (1, 1, L, K)

            K_cand = K[:, :, clamped_indices, :]
            V_cand = V[:, :, clamped_indices, :]

            Q_exp = Q.unsqueeze(3)
            scores = (Q_exp * K_cand).sum(dim=-1) * self.scale
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=-1) * mask.float()
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            weights = self.attn_dropout(weights)

            context = (weights.unsqueeze(-1) * V_cand).sum(dim=3)
            context = context.transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(context), None

    def forward(
        self,
        x: torch.Tensor,
        T: Optional[int] = None,
        adaptive_halting: Optional[bool] = None,
        halt_threshold: Optional[float] = None,
        return_trajectory: bool = False,
        return_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Standalone forward pass unrolling T pure attention hops."""
        B, L, D = x.shape
        device = x.device
        steps = T if T is not None else self.default_T
        use_adaptive = adaptive_halting if adaptive_halting is not None else self.config.adaptive_halting
        eps = halt_threshold if halt_threshold is not None else self.config.halt_threshold

        s = x
        inv_sqrt_T = 1.0 / math.sqrt(steps)
        wave_latent = self.init_wave_latent if self.routing_mode == "harmonic" else None

        trajectory = [s] if return_trajectory else None
        exited = torch.zeros(B, L, dtype=torch.bool, device=device) if use_adaptive else None
        token_hops = torch.full((B, L), steps, dtype=torch.long, device=device) if (use_adaptive or return_stats) else None
        final_s = torch.zeros_like(s) if use_adaptive else None

        for t_step in range(1, steps + 1):
            prev_s = s
            attn_out, wave_latent = self.single_hop_attention(s, wave_latent)
            s = s + inv_sqrt_T * attn_out  # Contractive linear residual

            if return_trajectory:
                trajectory.append(s)

            if use_adaptive:
                diff_norm = torch.norm(s - prev_s, dim=-1)
                s_norm = torch.norm(s, dim=-1) + 1e-6
                rel_diff = diff_norm / s_norm
                halt_cond = (rel_diff <= eps)

                newly_halted = (halt_cond & ~exited) | ((t_step == steps) & ~exited)
                final_s[newly_halted] = s[newly_halted]
                token_hops[newly_halted] = t_step
                exited = exited | halt_cond

        out = final_s if use_adaptive else s
        out = self.resid_dropout(out)

        if return_stats:
            stats = {
                "avg_hops": token_hops.float().mean().item(),
                "compute_savings": (1.0 - (token_hops.float().mean().item() / steps)) * 100.0,
                "token_hops": token_hops,
                "trajectory": trajectory
            }
            return out, stats

        if return_trajectory:
            return out, {"trajectory": trajectory}

        return out


class SubQBlock(nn.Module):
    """
    Modular SubQTransformer Block.
    
    Supports:
    - Pre-LayerNorm architecture.
    - Configurable MLP scheduling via mlp_interval:
        * mlp_interval = 0: MLP evaluated strictly once at block end (Canonical Season 2, optimal for memory & speed).
        * mlp_interval = 1: MLP evaluated at every recurrent hop.
        * mlp_interval = X: MLP evaluated every X hops.
    """
    def __init__(self, config: SubQConfig):
        super().__init__()
        self.config = config
        self.default_T = config.default_T
        self.mlp_interval = config.mlp_interval

        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.surfer = SubQSurfer(config)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_mlp, bias=config.bias),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_mlp, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        T: Optional[int] = None,
        adaptive_halting: Optional[bool] = None,
        halt_threshold: Optional[float] = None,
        return_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        steps = T if T is not None else self.default_T
        interval = self.mlp_interval
        use_adaptive = adaptive_halting if adaptive_halting is not None else self.config.adaptive_halting
        eps = halt_threshold if halt_threshold is not None else self.config.halt_threshold

        B, L, D = x.shape
        device = x.device

        # Calculate number of MLP executions for contraction scaling
        if interval > 0:
            num_mlp_passes = max(1, math.ceil(steps / interval))
            mlp_scale = 1.0 / math.sqrt(num_mlp_passes)
        else:
            mlp_scale = 1.0

        inv_sqrt_T = 1.0 / math.sqrt(steps)
        s = x
        wave_latent = self.surfer.init_wave_latent if self.surfer.routing_mode == "harmonic" else None

        exited = torch.zeros(B, L, dtype=torch.bool, device=device) if use_adaptive else None
        token_hops = torch.full((B, L), steps, dtype=torch.long, device=device) if (use_adaptive or return_stats) else None
        final_s = torch.zeros_like(s) if use_adaptive else None

        # Recurrent thought loop
        for step in range(1, steps + 1):
            prev_s = s
            z = self.ln1(s)
            attn_out, wave_latent = self.surfer.single_hop_attention(z, wave_latent)
            s = s + inv_sqrt_T * attn_out  # Per-hop attention residual

            # Check interleaved MLP execution
            if interval > 0 and step % interval == 0:
                s = s + mlp_scale * self.mlp(self.ln2(s))

            if use_adaptive:
                diff_norm = torch.norm(s - prev_s, dim=-1)
                s_norm = torch.norm(s, dim=-1) + 1e-6
                rel_diff = diff_norm / s_norm
                halt_cond = (rel_diff <= eps)

                newly_halted = (halt_cond & ~exited) | ((step == steps) & ~exited)
                final_s[newly_halted] = s[newly_halted]
                token_hops[newly_halted] = step
                exited = exited | halt_cond

        s = final_s if use_adaptive else s

        # If mlp_interval is 0 (or final step did not trigger interleaved MLP), execute block MLP
        if interval == 0 or (steps % interval != 0):
            s = s + mlp_scale * self.mlp(self.ln2(s))

        if return_stats:
            avg_hops = token_hops.float().mean().item() if token_hops is not None else float(steps)
            stats = {
                "avg_hops": avg_hops,
                "compute_savings": (1.0 - (avg_hops / steps)) * 100.0,
                "token_hops": token_hops,
                "hops_executed": steps
            }
            return s, stats

        return s
