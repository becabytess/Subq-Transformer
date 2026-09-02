import torch
import torch.nn as nn
import torch.nn.functional as F
import math

B, L, d_model, n_heads = 2, 257, 192, 6
head_dim = d_model // n_heads # 32
P = 8
radius = 2
window_size = 2 * radius + 1 # 5

s = torch.randn(B, L, d_model)
q_proj = nn.Linear(d_model, d_model)
k_proj = nn.Linear(d_model, d_model)
v_proj = nn.Linear(d_model, d_model)
ln_q = nn.LayerNorm(d_model)
ln_kv = nn.LayerNorm(d_model)
c_proj = nn.Linear(d_model, d_model)

q = q_proj(ln_q(s)).view(B, L, n_heads, head_dim).transpose(1, 2) # [B, H, L, 32]

# Simulated peak centers: [B, H, P]
all_centers = torch.randint(0, L, (B, n_heads, P))
win_offs = torch.arange(-radius, radius + 1).view(1, 1, 1, window_size) # [1, 1, 1, 5]
tri_prior = (1.0 - torch.abs(torch.arange(-radius, radius + 1).float()) / (radius + 1.0)).view(1, 1, 1, window_size)

neighbor_offsets = (all_centers.unsqueeze(-1) + win_offs) % L # [B, H, P, 5]
filter_weights = F.softmax(torch.log(tri_prior.clamp(min=1e-4)), dim=-1).expand(B, n_heads, P, window_size)

q_pos = torch.arange(L).view(1, 1, L, 1, 1)
target_tokens = (q_pos - neighbor_offsets.unsqueeze(2)) % L # [B, H, L, P, 5]

s_expanded = s.view(B, 1, L, 1, 1, d_model).expand(B, n_heads, L, P, window_size, d_model)
idx_exp = target_tokens.unsqueeze(-1).expand(B, n_heads, L, P, window_size, d_model)
s_neighbors = torch.gather(s_expanded, dim=2, index=idx_exp) # [B, H, L, P, 5, 192]

w_exp = filter_weights.unsqueeze(2).unsqueeze(-1) # [B, H, 1, P, 5, 1]
s_super = (w_exp * s_neighbors).sum(dim=4) # [B, H, L, P, 192]

s_super_norm = ln_kv(s_super) # [B, H, L, P, 192]
k_all = k_proj(s_super_norm).view(B, n_heads, L, P, n_heads, head_dim)
v_all = v_proj(s_super_norm).view(B, n_heads, L, P, n_heads, head_dim)

head_idx = torch.arange(n_heads).view(1, n_heads, 1, 1, 1, 1).expand(B, n_heads, L, P, 1, head_dim)
k_super = torch.gather(k_all, dim=4, index=head_idx).squeeze(4) # [B, H, L, P, 32]
v_super = torch.gather(v_all, dim=4, index=head_idx).squeeze(4) # [B, H, L, P, 32]

peak_vals = torch.zeros(B, n_heads, P)
scores = (q.unsqueeze(3) * k_super).sum(dim=-1) / math.sqrt(head_dim) + peak_vals.unsqueeze(2) # [B, H, L, P]
attn = F.softmax(scores, dim=-1)
attn_out = (attn.unsqueeze(-1) * v_super).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
out = c_proj(attn_out)

print("Forward pass successful! Output shape:", out.shape)
