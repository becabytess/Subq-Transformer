"""
Core Neural Layers for SubQTransformer:
- SubQSurfer: Multi-scale sparse graph constructor & iterative contractive message-passing engine.
- SubQBlock: High-throughput Transformer block combining SubQSurfer with Feedforward MLP.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from subqtransformer.config import SubQConfig


class SubQSurfer(nn.Module):
    """
    Sub-Quadratic Multi-Scale Sparse Graph Message-Passing Layer.
    
    Architecture:
    1. Sparse Candidate Indexing: Maps every token index i to K logarithmic relative offsets.
    2. Relational Graph Routing: Computes multi-head attention routing policy pi^(1) over K candidates.
    3. Dynamical Message Passing: Unrolls T recurrent thought hops over the learned graph via contractive GRU dynamics.
    4. Optional Adaptive Early-Exit: Per-token velocity halting (||Δs|| / ||s|| <= eps) to halt settled tokens early.
    """
    def __init__(self, config: SubQConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        assert self.head_dim * config.n_heads == config.d_model, "d_model must be divisible by n_heads"
        
        self.jump_offsets = config.jump_offsets
        self.K = len(self.jump_offsets)
        self.default_T = config.default_T
        self.max_seq_len = config.max_seq_len
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # Fused GRU Gate Projections (Single unified GEMMs)
        self.w_ih = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.w_hh = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precompute indexing buffer
        self._build_index_buffers(config.max_seq_len)

    def _build_index_buffers(self, seq_len: int):
        target_indices = torch.zeros((seq_len, self.K), dtype=torch.long)
        valid_mask = torch.zeros((seq_len, self.K), dtype=torch.bool)
        for i in range(seq_len):
            for k_idx, offset in enumerate(self.jump_offsets):
                target_pos = i - offset
                if target_pos >= 0:
                    target_indices[i, k_idx] = target_pos
                    valid_mask[i, k_idx] = True
                else:
                    target_indices[i, k_idx] = 0
                    valid_mask[i, k_idx] = False

        self.register_buffer("target_indices", target_indices, persistent=False)
        self.register_buffer("valid_mask", valid_mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        T: Optional[int] = None,
        adaptive_halting: Optional[bool] = None,
        halt_threshold: Optional[float] = None,
        return_trajectory: bool = False,
        return_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass with sub-quadratic O(L * K) message passing.
        
        Args:
            x: Input tensor (B, L, D)
            T: Number of thought hops to unroll (overrides config.default_T)
            adaptive_halting: Enable per-token dynamical early exit
            halt_threshold: Epsilon threshold for early stopping
            return_trajectory: Return list of states across all hops
            return_stats: Return compute metrics (average hops, distribution)
        """
        B, L, D = x.shape
        device = x.device
        steps = T if T is not None else self.default_T
        use_adaptive = adaptive_halting if adaptive_halting is not None else self.config.adaptive_halting
        eps = halt_threshold if halt_threshold is not None else self.config.halt_threshold

        # Ensure index buffers match current sequence length
        if L > self.target_indices.size(0):
            self._build_index_buffers(L)
            self.target_indices = self.target_indices.to(device)
            self.valid_mask = self.valid_mask.to(device)

        indices = self.target_indices[:L]
        mask = self.valid_mask[:L].unsqueeze(0).unsqueeze(1)  # (1, 1, L, K)

        # 1. Project Q, K, V
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d_k)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d_k)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d_k)

        # 2. Gather K, V at candidate offsets: (B, H, L, K, d_k)
        k_cand = k[:, :, indices, :]
        v_cand = v[:, :, indices, :]

        # 3. Compute Relational Graph Topology π^(1) in Hop 1
        q_exp = q.unsqueeze(3)  # (B, H, L, 1, d_k)
        scores = (q_exp * k_cand).sum(dim=-1) * self.scale  # (B, H, L, K)
        scores = scores.masked_fill(~mask, float("-inf"))
        pi = F.softmax(scores, dim=-1)
        pi = self.attn_dropout(pi)

        # 4. Context Aggregation: (B, L, D)
        ctx_heads = (pi.unsqueeze(-1) * v_cand).sum(dim=3)  # (B, H, L, d_k)
        ctx = ctx_heads.transpose(1, 2).contiguous().view(B, L, D)

        # 5. Precompute Context Gates (Invariant across T hops)
        gates_ctx = self.w_ih(ctx)  # (B, L, 3D)

        # 6. Recurrent Message Passing & Contractive Dynamical Settling
        s = x  # Initial state at hop t=0
        trajectory = [s] if return_trajectory else None
        
        # Adaptive halting tracking
        exited = torch.zeros(B, L, dtype=torch.bool, device=device) if use_adaptive else None
        token_hops = torch.full((B, L), steps, dtype=torch.long, device=device) if (use_adaptive or return_stats) else None
        final_s = torch.zeros_like(s) if use_adaptive else None

        for t_step in range(1, steps + 1):
            prev_s = s
            gates_h = self.w_hh(s)
            gates = gates_ctx + gates_h

            r_gate, z_gate, n_gate = gates.chunk(3, dim=-1)
            r = torch.sigmoid(r_gate)
            z = torch.sigmoid(z_gate)
            
            # Gated candidate state
            c_in = gates_ctx.chunk(3, dim=-1)[2] + (self.w_hh(r * s)).chunk(3, dim=-1)[2]
            n = torch.tanh(c_in)
            
            # Gated state update
            s = (1.0 - z) * n + z * s

            if return_trajectory:
                trajectory.append(s)

            if use_adaptive:
                # Dynamical state velocity: ||s^(t) - s^(t-1)|| / (||s^(t)|| + 1e-6)
                diff_norm = torch.norm(s - prev_s, dim=-1)
                s_norm = torch.norm(s, dim=-1) + 1e-6
                rel_diff = diff_norm / s_norm
                halt_cond = (rel_diff <= eps)

                newly_halted = (halt_cond & ~exited) | ((t_step == steps) & ~exited)
                final_s[newly_halted] = s[newly_halted]
                token_hops[newly_halted] = t_step
                exited = exited | halt_cond

        out_s = final_s if use_adaptive else s
        out = self.resid_dropout(self.out_proj(out_s))

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
    Unified SubQTransformer Block.
    Pre-LayerNorm architecture with SubQSurfer and Feedforward MLP.
    """
    def __init__(self, config: SubQConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.surfer = SubQSurfer(config)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_mlp, bias=config.bias) ,
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
        if return_stats:
            surfer_out, stats = self.surfer(
                self.ln1(x),
                T=T,
                adaptive_halting=adaptive_halting,
                halt_threshold=halt_threshold,
                return_stats=True
            )
            x = x + surfer_out
            x = x + self.mlp(self.ln2(x))
            return x, stats

        x = x + self.surfer(
            self.ln1(x),
            T=T,
            adaptive_halting=adaptive_halting,
            halt_threshold=halt_threshold
        )
        x = x + self.mlp(self.ln2(x))
        return x
