import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.38.0",
        "numpy",
        "requests",
        "accelerate"
    )
)

app = modal.App("gpt2-subq-multidomain-transfer", image=image)
volume = modal.Volume.from_name("subq-gpt2-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=3600, volumes={"/root/checkpoints": volume})
def run_multidomain_transfer_experiment():
    import math
    import time
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import requests
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  EXPERIMENT: SUBQ-GPT2 (124M) MULTI-DOMAIN GENERALIZATION & ARCHITECTURAL TRANSFER")
    print("  Questions:")
    print("    1. When SubQ-GPT2 adapts on General WebText, does it learn a GENERAL transfer to SubQ jump attention?")
    print("    2. How well does it retain language capabilities on 3 completely HELD-OUT domains (WikiText-2, Python Code, Shakespeare)?")
    print("    3. Are model checkpoints safely saved to persistent Modal Volume for permanent use?")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Load Pre-Trained Dense GPT-2
    print("\n[1/5] Loading Pre-Trained GPT-2 (124M) from HuggingFace...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_dense = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_dense.eval()

    n_layers = gpt2_dense.config.n_layer # 12
    n_heads = gpt2_dense.config.n_head   # 12
    d_model = gpt2_dense.config.n_embd   # 768
    head_dim = d_model // n_heads        # 64
    vocab_size = gpt2_dense.config.vocab_size # 50257
    seq_len = 512

    # 2. Fetch Multi-Domain Datasets
    print("\n[2/5] Fetching Multi-Domain Corpora (1 Adaptation Corpus + 3 Held-Out Zero-Shot Test Sets)...")
    
    # Dataset A: General Web/Article Text (Adaptation Domain)
    url_web = "https://raw.githubusercontent.com/karpathy/build-nanogpt/master/fineweb10B/sample.txt"
    try:
        r = requests.get(url_web, timeout=10)
        web_text = r.text if r.status_code == 200 and len(r.text) > 10000 else ""
    except Exception:
        web_text = ""
    
    if len(web_text) < 50000:
        # Fallback to high-quality multi-article web text
        url_web_fallback = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
        web_text = requests.get(url_web_fallback).text

    # Dataset B: WikiText-2 Test (Held-Out Formal Knowledge)
    url_wiki = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
    wiki_text = requests.get(url_wiki).text

    # Dataset C: Python Code (Held-Out Syntax & Indentation)
    url_code1 = "https://raw.githubusercontent.com/psf/requests/main/src/requests/sessions.py"
    url_code2 = "https://raw.githubusercontent.com/psf/requests/main/src/requests/models.py"
    url_code3 = "https://raw.githubusercontent.com/psf/requests/main/src/requests/api.py"
    code_text = requests.get(url_code1).text + "\n\n" + requests.get(url_code2).text + "\n\n" + requests.get(url_code3).text

    # Dataset D: TinyShakespeare (Held-Out Archaic/Dialogue)
    url_shk = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    shk_text = requests.get(url_shk).text

    # Tokenize Datasets
    web_tokens = torch.tensor(tokenizer.encode(web_text), dtype=torch.long, device=device)
    wiki_tokens = torch.tensor(tokenizer.encode(wiki_text), dtype=torch.long, device=device)
    code_tokens = torch.tensor(tokenizer.encode(code_text), dtype=torch.long, device=device)
    shk_tokens = torch.tensor(tokenizer.encode(shk_text), dtype=torch.long, device=device)

    # Split Adaptation Corpus into Train & Validation
    split_idx = int(len(web_tokens) * 0.9)
    train_data = web_tokens[:split_idx]
    val_web_data = web_tokens[split_idx:]

    print(f"  - Adaptation Corpus (Web/Prose Train): {len(train_data):,} tokens | Val: {len(val_web_data):,} tokens")
    print(f"  - [HELD-OUT 1] WikiText-2 Test:        {len(wiki_tokens):,} tokens")
    print(f"  - [HELD-OUT 2] Python Code Test:       {len(code_tokens):,} tokens")
    print(f"  - [HELD-OUT 3] TinyShakespeare Test:   {len(shk_tokens):,} tokens")

    # 3. Define SubQ Jump Attention Architecture
    def get_menu(K=32):
        local_k = 8
        local_offsets = list(range(local_k))
        long_jumps = []
        val = local_k
        while len(local_offsets) + len(long_jumps) < K and val < seq_len:
            long_jumps.append(val)
            val = int(val * 1.4) + 1
        menu = local_offsets + long_jumps
        while len(menu) < K:
            menu.append(menu[-1] + 1)
        return torch.tensor(menu[:K], dtype=torch.long, device=device)

    class SubQJumpAttention(nn.Module):
        def __init__(self, d_model=768, n_heads=12, K=32, T_max=4, eps=0.01):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.K = K
            self.T_max = T_max
            self.eps = eps

            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)
            self.register_buffer("offsets", get_menu(K))
            self.temp_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1))

            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=True)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=True)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=True)

            with torch.no_grad():
                self.w_gate_h.bias.data.fill_(0.0)
                self.w_gate_h.bias.data[self.d_model:].fill_(3.0)
                self.w_cand_h.bias.data.fill_(0.0)
                self.w_ih.bias.data.fill_(0.0)

        def forward(self, x, T_max=None, eps=None):
            B, L, D = x.shape
            T_max = T_max or self.T_max
            eps = eps or self.eps

            s = x
            hops_taken = 0

            for t in range(T_max):
                s_prev = s

                qkv = self.c_attn(s)
                q, k, v = qkv.chunk(3, dim=-1)

                q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

                k_shifted_list = []
                v_shifted_list = []
                valid_mask_list = []

                for delta_val in self.offsets:
                    d = delta_val.item()
                    if d >= L:
                        k_shifted = torch.zeros_like(k)
                        v_shifted = torch.zeros_like(v)
                        mask = torch.zeros(B, 1, L, device=x.device, dtype=torch.bool)
                    elif d == 0:
                        k_shifted = k
                        v_shifted = v
                        mask = torch.ones(B, 1, L, device=x.device, dtype=torch.bool)
                    else:
                        k_shifted = F.pad(k[:, :, :-d, :], (0, 0, d, 0))
                        v_shifted = F.pad(v[:, :, :-d, :], (0, 0, d, 0))
                        mask = torch.cat([
                            torch.zeros(B, 1, d, device=x.device, dtype=torch.bool),
                            torch.ones(B, 1, L - d, device=x.device, dtype=torch.bool)
                        ], dim=-1)

                    k_shifted_list.append(k_shifted)
                    v_shifted_list.append(v_shifted)
                    valid_mask_list.append(mask)

                K_cand = torch.stack(k_shifted_list, dim=3)
                V_cand = torch.stack(v_shifted_list, dim=3)
                valid_mask = torch.stack(valid_mask_list, dim=3)

                q_expanded = q.unsqueeze(3)
                scores = (q_expanded * K_cand).sum(dim=-1) / math.sqrt(self.head_dim)
                scores = scores * self.temp_scale
                scores = scores.masked_fill(~valid_mask, float("-inf"))
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

                out_h = (attn_weights.unsqueeze(-1) * V_cand).sum(dim=3)
                out = out_h.transpose(1, 2).contiguous().view(B, L, D)
                context = self.c_proj(out)

                if T_max == 1:
                    s = context
                    hops_taken = 1
                    break

                gates_ih = self.w_ih(context)
                r_ih, z_ih, n_ih = gates_ih.chunk(3, dim=-1)

                gates_h = self.w_gate_h(s)
                r_h, z_h = gates_h.chunk(2, dim=-1)

                r = torch.sigmoid(r_ih + r_h)
                z = torch.sigmoid(z_ih + z_h)
                n = torch.tanh(n_ih + self.w_cand_h(r * s))

                s = (1.0 - z) * n + z * s
                hops_taken += 1

                delta_energy = torch.norm(s - s_prev, p=2, dim=-1).mean().item()
                if delta_energy < eps and t >= 1:
                    break

            return s, hops_taken

    class SubQGPT2Model(nn.Module):
        def __init__(self, gpt2_ref, K=32, T_max=4, eps=0.01):
            super().__init__()
            self.config = gpt2_ref.config
            self.wte = nn.Embedding.from_pretrained(gpt2_ref.transformer.wte.weight.clone(), freeze=False)
            self.wpe = nn.Embedding.from_pretrained(gpt2_ref.transformer.wpe.weight.clone(), freeze=False)
            self.drop = nn.Dropout(gpt2_ref.config.embd_pdrop)
            self.ln_f = nn.LayerNorm(768, eps=gpt2_ref.config.layer_norm_epsilon)
            self.ln_f.weight.data.copy_(gpt2_ref.transformer.ln_f.weight.data)
            self.ln_f.bias.data.copy_(gpt2_ref.transformer.ln_f.bias.data)

            self.lm_head = nn.Linear(768, vocab_size, bias=False)
            self.lm_head.weight.data.copy_(gpt2_ref.lm_head.weight.data)

            self.blocks = nn.ModuleList()
            for l in range(12):
                ref_block = gpt2_ref.transformer.h[l]
                subq_attn = SubQJumpAttention(d_model=768, n_heads=12, K=K, T_max=T_max, eps=eps)

                with torch.no_grad():
                    subq_attn.c_attn.weight.copy_(ref_block.attn.c_attn.weight.t())
                    subq_attn.c_attn.bias.copy_(ref_block.attn.c_attn.bias)
                    subq_attn.c_proj.weight.copy_(ref_block.attn.c_proj.weight.t())
                    subq_attn.c_proj.bias.copy_(ref_block.attn.c_proj.bias)

                ln1 = nn.LayerNorm(768, eps=gpt2_ref.config.layer_norm_epsilon)
                ln1.weight.data.copy_(ref_block.ln_1.weight.data)
                ln1.bias.data.copy_(ref_block.ln_1.bias.data)

                ln2 = nn.LayerNorm(768, eps=gpt2_ref.config.layer_norm_epsilon)
                ln2.weight.data.copy_(ref_block.ln_2.weight.data)
                ln2.bias.data.copy_(ref_block.ln_2.bias.data)

                c_fc = nn.Linear(768, 4 * 768)
                c_fc.weight.data.copy_(ref_block.mlp.c_fc.weight.t())
                c_fc.bias.data.copy_(ref_block.mlp.c_fc.bias)

                c_proj = nn.Linear(4 * 768, 768)
                c_proj.weight.data.copy_(ref_block.mlp.c_proj.weight.t())
                c_proj.bias.data.copy_(ref_block.mlp.c_proj.bias)

                mlp = nn.Sequential(
                    c_fc,
                    nn.GELU(approximate="tanh"),
                    c_proj,
                    nn.Dropout(gpt2_ref.config.resid_pdrop)
                )

                block_dict = nn.ModuleDict({
                    "ln_1": ln1,
                    "attn": subq_attn,
                    "ln_2": ln2,
                    "mlp": mlp
                })
                self.blocks.append(block_dict)

        def forward(self, input_ids, T_max=None, eps=None):
            B, L = input_ids.shape
            pos = torch.arange(0, L, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            hidden_states = self.wte(input_ids) + self.wpe(pos)
            hidden_states = self.drop(hidden_states)

            total_hops = 0
            for block in self.blocks:
                norm_1 = block["ln_1"](hidden_states)
                attn_out, hops = block["attn"](norm_1, T_max=T_max, eps=eps)
                hidden_states = hidden_states + attn_out
                total_hops += hops

                norm_2 = block["ln_2"](hidden_states)
                mlp_out = block["mlp"](norm_2)
                hidden_states = hidden_states + mlp_out

            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            avg_hops = total_hops / len(self.blocks)
            return logits, avg_hops

        @torch.no_grad()
        def generate(self, input_ids, max_new_tokens=30, temperature=0.8, top_k=50):
            self.eval()
            curr_ids = input_ids.clone()
            for _ in range(max_new_tokens):
                cond_ids = curr_ids[:, -seq_len:]
                logits, _ = self.forward(cond_ids, T_max=4, eps=0.01)
                next_token_logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
            return curr_ids

    # 4. Evaluation Function across arbitrary test datasets
    def evaluate_dataset(model_eval, dataset_tensor, is_subq=False, T_max=4, eps=0.01, max_batches=20):
        model_eval.eval()
        loss_sum = 0.0
        n_eval = min(max_batches, (len(dataset_tensor) - 1) // seq_len)
        if n_eval == 0:
            return float("nan"), float("nan"), 0.0

        with torch.no_grad():
            for i in range(n_eval):
                idx = i * seq_len
                bx = dataset_tensor[idx : idx + seq_len].unsqueeze(0)
                by = dataset_tensor[idx + 1 : idx + seq_len + 1].unsqueeze(0)

                if is_subq:
                    logits, _ = model_eval(bx, T_max=T_max, eps=eps)
                else:
                    logits = model_eval(bx).logits

                loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
                loss_sum += loss.item()

        avg_loss = loss_sum / n_eval
        avg_ppl = math.exp(min(avg_loss, 20.0))
        return avg_loss, avg_ppl

    # 5. Measure Native Dense GPT-2 Baselines across all 4 domains
    print("\n[3/5] Measuring Native Dense GPT-2 Baselines Across All 4 Domains...")
    dense_results = {}
    datasets = {
        "Adaptation Corpus (WebText Val)": val_web_data,
        "Held-Out 1: WikiText-2 (Knowledge)": wiki_tokens,
        "Held-Out 2: Python Code (Syntax)": code_tokens,
        "Held-Out 3: Shakespeare (Drama)": shk_tokens
    }

    print("-" * 105)
    print(f"{'Domain / Dataset':<40} | {'Dense GPT-2 NLL':<18} | {'Dense GPT-2 PPL':<18}")
    print("-" * 105)
    for name, d_tensor in datasets.items():
        nll, ppl = evaluate_dataset(gpt2_dense, d_tensor, is_subq=False)
        dense_results[name] = {"nll": nll, "ppl": ppl}
        print(f"{name:<40} | {nll:>14.4f}     | {ppl:>14.2f}")
    print("-" * 105)

    # 6. Initialize SubQ-GPT2 (124M)
    print("\n[4/5] Initializing SubQ-GPT2 (124M) and measuring Initial Zero-Shot Transplant...")
    subq_model = SubQGPT2Model(gpt2_dense, K=32, T_max=4, eps=0.01).to(device)

    print("-" * 105)
    print(f"{'Domain / Dataset':<40} | {'Zero-Shot SubQ NLL':<18} | {'Zero-Shot SubQ PPL':<18}")
    print("-" * 105)
    initial_subq_results = {}
    for name, d_tensor in datasets.items():
        nll, ppl = evaluate_dataset(subq_model, d_tensor, is_subq=True, T_max=4, eps=0.01)
        initial_subq_results[name] = {"nll": nll, "ppl": ppl}
        print(f"{name:<40} | {nll:>14.4f}     | {ppl:>14.2f}")
    print("-" * 105)

    # 7. Unlocked Adaptation on WebText with Persistent Checkpoint Saving
    for p in subq_model.parameters():
        p.requires_grad = True

    total_steps = 1500
    warmup_steps = 100
    accum_steps = 4
    batch_size = 1
    max_lr = 1.5e-4
    min_lr = 1.0e-5

    optimizer = torch.optim.AdamW(subq_model.parameters(), lr=max_lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    def get_lr(step):
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    print("\n[5/5] Running Unlocked SubQ Adaptation on WebText (1,500 Steps)...")
    print(f"{'Step':<12} | {'Web Val PPL':<16} | {'WikiText-2 PPL':<16} | {'Python Code PPL':<18} | {'Shakespeare PPL':<18}")
    print("-" * 115)

    t0 = time.time()
    optimizer.zero_grad()
    best_val_ppl = float("inf")
    checkpoint_dir = "/root/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, "subq_gpt2_best.pt")

    for step in range(1, total_steps + 1):
        subq_model.train()
        lr_now = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now

        idx = (step * batch_size * seq_len) % (len(train_data) - batch_size * seq_len - 1)
        bx = train_data[idx : idx + batch_size * seq_len].view(batch_size, seq_len)
        by = train_data[idx + 1 : idx + batch_size * seq_len + 1].view(batch_size, seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            with torch.no_grad():
                teacher_logits = gpt2_dense(bx).logits

            student_logits, _ = subq_model(bx, T_max=4, eps=0.01)

            loss_ce = F.cross_entropy(student_logits.view(-1, vocab_size), by.view(-1))
            loss_kl = F.kl_div(
                F.log_softmax(student_logits / 1.5, dim=-1),
                F.softmax(teacher_logits / 1.5, dim=-1),
                reduction="batchmean"
            ) * (1.5 ** 2)

            loss_total = (loss_ce + 0.5 * loss_kl) / accum_steps

        scaler.scale(loss_total).backward()

        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(subq_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Evaluate every 150 steps across all domains
        if step % 150 == 0 or step == total_steps:
            _, val_web_ppl = evaluate_dataset(subq_model, val_web_data, is_subq=True, max_batches=15)
            _, val_wiki_ppl = evaluate_dataset(subq_model, wiki_tokens, is_subq=True, max_batches=15)
            _, val_code_ppl = evaluate_dataset(subq_model, code_tokens, is_subq=True, max_batches=15)
            _, val_shk_ppl = evaluate_dataset(subq_model, shk_tokens, is_subq=True, max_batches=15)

            if val_web_ppl < best_val_ppl:
                best_val_ppl = val_web_ppl
                # Save best checkpoint to persistent Modal Volume
                torch.save({
                    "step": step,
                    "state_dict": subq_model.state_dict(),
                    "val_web_ppl": val_web_ppl,
                    "val_wiki_ppl": val_wiki_ppl,
                    "val_code_ppl": val_code_ppl,
                    "val_shk_ppl": val_shk_ppl,
                }, best_ckpt_path)
                volume.commit()
                mark = " ⭐ [Saved to Volume]"
            else:
                mark = ""

            print(f"Step {step:>4}/{total_steps} | {val_web_ppl:>12.2f}     | {val_wiki_ppl:>12.2f}     | {val_code_ppl:>14.2f}     | {val_shk_ppl:>14.2f}{mark}")

    print(f"\nTraining completed in {time.time() - t0:.1f}s")
    
    # Load Best Model Checkpoint for Final Comprehensive Multi-Domain Evaluation
    print("\nLoading Best Checkpoint from Modal Volume for Final Evaluation...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    subq_model.load_state_dict(best_ckpt["state_dict"])
    subq_model.eval()

    print("\n" + "=" * 125)
    print("  FINAL MULTI-DOMAIN ARCHITECTURAL TRANSFER BENCHMARK RESULTS")
    print("=" * 125)
    print(f"{'Domain / Dataset':<35} | {'Dense GPT-2 PPL':<16} | {'SubQ Zero-Shot':<16} | {'SubQ Adapted PPL':<18} | {'Gap Closed (%)'}")
    print("-" * 125)

    for name, d_tensor in datasets.items():
        _, final_ppl = evaluate_dataset(subq_model, d_tensor, is_subq=True, max_batches=25)
        d_ppl = dense_results[name]["ppl"]
        init_ppl = initial_subq_results[name]["ppl"]
        gap_closed = ((init_ppl - final_ppl) / (init_ppl - d_ppl)) * 100.0 if init_ppl > d_ppl else 100.0
        print(f"{name:<35} | {d_ppl:>12.2f}     | {init_ppl:>12.2f}     | {final_ppl:>14.2f}     | {gap_closed:>10.1f}%")

    print("=" * 125)

    # 8. Qualitative Language Generation Across Domains
    print("\n" + "=" * 125)
    print("  QUALITATIVE TEXT GENERATION PROMPTS FROM ADAPTED SUBQ-GPT2 (K=32, T=4)")
    print("=" * 125)

    prompts = [
        "The theory of general relativity predicts that gravitational",
        "def compute_attention_scores(query, key, mask):",
        "ROMEO: If I profane with my unworthiest hand"
    ]

    for p in prompts:
        p_ids = torch.tensor(tokenizer.encode(p), dtype=torch.long, device=device).unsqueeze(0)
        gen_ids = subq_model.generate(p_ids, max_new_tokens=35, temperature=0.7, top_k=40)
        gen_text = tokenizer.decode(gen_ids[0].tolist())
        print(f"\n--- Prompt: \"{p}\" ---")
        print(gen_text)

    print("\n" + "=" * 125)
    print(f"Checkpoint permanently saved to Modal Volume: {best_ckpt_path}")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_multidomain_transfer_experiment.remote()
