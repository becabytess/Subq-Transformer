import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision",
        "datasets",
        "numpy"
    )
)

app = modal.App("subq-compile-benchmark", image=image)

@app.function(gpu="T4", timeout=600)
def run_compile_benchmark():
    import time
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  SUBQ COMPILATION & OPTIMIZATION SPEED BENCHMARK (Tesla T4)")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    B = 128
    L = 256
    d_model = 128
    d_mlp = 512
    n_heads = 4
    head_dim = d_model // n_heads
    grid_size = 16
    n_patches = 256
    patch_size = 2
    T_hops = 3

    # Build 2D Candidate Map
    fib_strides = [2, 3, 5, 8, 13]
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    cand_map = []
    max_k = 0
    for r in range(grid_size):
        for c in range(grid_size):
            p_list = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        p_list.append(nr * grid_size + nc)
            for dr, dc in directions:
                for stride in fib_strides:
                    nr, nc = r + dr * stride, c + dc * stride
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        p_list.append(nr * grid_size + nc)
            p_list = sorted(list(set(p_list)))
            cand_map.append(p_list)
            if len(p_list) > max_k:
                max_k = len(p_list)

    K_2d = max_k
    cand_indices_2d = torch.zeros((n_patches, K_2d), dtype=torch.long, device=device)
    cand_mask_2d = torch.zeros((n_patches, K_2d), dtype=torch.bool, device=device)
    for i in range(n_patches):
        p_list = cand_map[i]
        for k_idx, p in enumerate(p_list):
            cand_indices_2d[i, k_idx] = p
            cand_mask_2d[i, k_idx] = True

    # -------------------------------------------------------------------------
    # Baseline SubQSurfer2D
    # -------------------------------------------------------------------------
    class SubQSurfer2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)
            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=False)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x_norm):
            B, L, D = x_norm.shape
            q = self.q_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)
            k = self.k_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)
            v = self.v_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)

            k_cand = k[:, :, cand_indices_2d, :]
            v_cand = v[:, :, cand_indices_2d, :]

            q_exp = q.unsqueeze(3)
            scores = (q_exp * k_cand).sum(dim=-1) / math.sqrt(head_dim)
            scores = scores.masked_fill(~cand_mask_2d.unsqueeze(0).unsqueeze(0), -1e9)
            pi = F.softmax(scores, dim=-1)

            ctx = (pi.unsqueeze(-1) * v_cand).sum(dim=3).transpose(1, 2).contiguous().view(B, L, D)
            gates_ctx = self.w_ih(ctx)
            r_ctx, z_ctx, n_ctx = gates_ctx.chunk(3, dim=-1)

            s = x_norm
            for _ in range(T_hops):
                gates_h = self.w_gate_h(s)
                r_h, z_h = gates_h.chunk(2, dim=-1)
                r = torch.sigmoid(r_ctx + r_h)
                z = torch.sigmoid(z_ctx + z_h)
                n = torch.tanh(n_ctx + self.w_cand_h(r * s))
                s = (1.0 - z) * n + z * s

            return self.out_proj(s)

    class SubQViT_1Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
            self.ln1 = nn.LayerNorm(d_model)
            self.surfer = SubQSurfer2D()
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp, bias=False),
                nn.GELU(),
                nn.Linear(d_mlp, d_model, bias=False)
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 10)

        def forward(self, x):
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x = x + self.pos_embed
            x = x + self.surfer(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            x = self.norm(x.mean(dim=1))
            return self.head(x)

    # Standard ViT for direct comparison
    class ViT1LBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_mlp,
                dropout=0.0, activation='gelu', batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 10)

        def forward(self, x):
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x = x + self.pos_embed
            x = self.transformer(x)
            x = self.norm(x.mean(dim=1))
            return self.head(x)

    dummy_input = torch.randn(B, 3, 32, 32, device=device)

    def benchmark_model(name, model):
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(15):
                _ = model(dummy_input)
        torch.cuda.synchronize()

        # Benchmark
        n_iters = 100
        start = time.time()
        with torch.no_grad():
            for _ in range(n_iters):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        fps = (n_iters * B) / elapsed
        latency_ms = (elapsed / n_iters) * 1000
        print(f"| {name:<45} | {fps:>8.1f} img/s | {latency_ms:>7.2f} ms/batch |")
        return fps, latency_ms

    print("-" * 75)
    print(f"| {'Model / Optimization Variant':<45} | {'Throughput':>8}       | {'Latency':>7}        |")
    print("-" * 75)

    # 1. Standard ViT-1L (PyTorch native SDPA)
    vit_model = ViT1LBaseline().to(device)
    benchmark_model("0. Standard ViT-1L (Native cuDNN)", vit_model)

    # 2. Uncompiled PyTorch SubQ Baseline
    subq_model = SubQViT_1Layer().to(device)
    benchmark_model("1. Pure PyTorch SubQ-ViT (Uncompiled)", subq_model)

    # 3. torch.compile (default Inductor fusion)
    compiled_subq = torch.compile(SubQViT_1Layer().to(device))
    benchmark_model("2. SubQ-ViT + torch.compile (Inductor)", compiled_subq)

    # 4. torch.compile (reduce-overhead / CUDA Graphs)
    compiled_cuda_subq = torch.compile(SubQViT_1Layer().to(device), mode="reduce-overhead")
    benchmark_model("3. SubQ-ViT + torch.compile (CUDA Graphs)", compiled_cuda_subq)

    print("-" * 75)

@app.local_entrypoint()
def main():
    run_compile_benchmark.remote()
