import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "numpy"
    )
)

app = modal.App("exp-qwen-subq-recurrent-gsm8k", image=image)
vol = modal.Volume.from_name("subq-qwen-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=2400, volumes={"/root/checkpoints": vol})
def run_qwen_subq_recurrent_gsm8k():
    import json
    import math
    import os
    import re
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 135)
    print("  STUDY 45: QWEN2.5-0.5B SUBQ WITH RECURRENT THINKING LOOPS (T=6) & ADAPTIVE TOKEN HALTING ON GSM8K")
    print("  Multi-Hop Logarithmic Wave Dynamics (K=8, T=6), Effective Receptive Field = 384 Tokens per Layer")
    print("=" * 135)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # -------------------------------------------------------------------------
    # 1. Download GSM8K Dataset
    # -------------------------------------------------------------------------
    print("\n[1/5] Downloading GSM8K Dataset...")
    train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

    def fetch_jsonl(url):
        data = []
        req = urllib.request.urlopen(url)
        for line in req.read().decode('utf-8').strip().split('\n'):
            if line.strip():
                data.append(json.loads(line))
        return data

    train_data = fetch_jsonl(train_url)
    test_data = fetch_jsonl(test_url)
    print(f"Dataset: {len(train_data):,} Training Problems | {len(test_data):,} Test Problems")

    # -------------------------------------------------------------------------
    # 2. Load Pretrained Qwen2.5-0.5B Model & Tokenizer
    # -------------------------------------------------------------------------
    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"\n[2/5] Loading Pretrained Model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    base_model.config.use_cache = False
    config = base_model.config

    # -------------------------------------------------------------------------
    # 3. Define SubQ Recurrent Thinking Attention with Token Early Stopping
    # -------------------------------------------------------------------------
    OFFSETS = [0, 1, 2, 4, 8, 16, 32, 64]
    K = len(OFFSETS)
    num_heads = config.num_attention_heads       # 14
    num_kv_heads = config.num_key_value_heads   # 2
    head_dim = config.hidden_size // num_heads   # 64
    num_key_value_groups = num_heads // num_kv_heads # 7
    T_RECUR = 6 # T=6 thinking iterations per layer
    EPS_HALT = 0.025 # Relative change threshold for early token convergence

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, n_kv, slen, h_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv, n_rep, slen, h_dim)
        return hidden_states.reshape(batch, n_kv * n_rep, slen, h_dim)

    class RecurrentSubQQwenAttention(nn.Module):
        def __init__(self, orig_attn, rotary_emb=None, T_steps=T_RECUR, eps_halt=EPS_HALT):
            super().__init__()
            self.q_proj = orig_attn.q_proj
            self.k_proj = orig_attn.k_proj
            self.v_proj = orig_attn.v_proj
            self.o_proj = orig_attn.o_proj
            self.rotary_emb = rotary_emb or getattr(orig_attn, "rotary_emb", None)
            self.head_dim = head_dim
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.num_key_value_groups = num_key_value_groups
            self.scale = 1.0 / math.sqrt(head_dim)
            self.offsets = OFFSETS
            self.T_steps = T_steps
            self.eps_halt = eps_halt
            # Learnable recurrence step blending parameter (initialized close to 1/T)
            self.step_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.bfloat16))

        def forward(self, hidden_states, position_ids=None, attention_mask=None, past_key_values=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
            B, L, D = hidden_states.shape

            # Compute Key and Value representations once from the base context
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            key_states = key_states.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

            if position_ids is None:
                position_ids = torch.arange(0, L, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

            if position_embeddings is not None:
                cos, sin = position_embeddings
            elif self.rotary_emb is not None:
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = None, None

            # Repeat KV for Grouped Query Attention (14 heads)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # Recurrent Thinking State initialization
            s_curr = hidden_states.clone()
            accumulated_output = torch.zeros_like(hidden_states)
            active_mask = torch.ones(B, L, 1, dtype=torch.bool, device=hidden_states.device)
            gate = torch.sigmoid(self.step_gate)

            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

            for step in range(self.T_steps):
                # 1. Update Query from the current evolving thinking state s_curr
                query_states = self.q_proj(s_curr).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                if cos is not None and sin is not None:
                    q_rot, k_rot = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                else:
                    q_rot, k_rot = query_states, key_states

                # 2. Causal Logarithmic Wave Routing across offsets {0, 1, 2, 4, 8, 16, 32, 64}
                scores_list, valid_masks = [], []
                for d in self.offsets:
                    if d == 0:
                        s = (q_rot * k_rot).sum(dim=-1) * self.scale
                        m = torch.ones(B, self.num_heads, L, device=hidden_states.device, dtype=torch.bool)
                    elif d < L:
                        q_slice = q_rot[:, :, d:, :]
                        k_slice = k_rot[:, :, :-d, :]
                        s_valid = (q_slice * k_slice).sum(dim=-1) * self.scale
                        s = F.pad(s_valid, (d, 0), value=-1e4)
                        m = torch.cat([
                            torch.zeros(B, self.num_heads, d, device=hidden_states.device, dtype=torch.bool),
                            torch.ones(B, self.num_heads, L - d, device=hidden_states.device, dtype=torch.bool)
                        ], dim=-1)
                    else:
                        s = torch.full((B, self.num_heads, L), -1e4, device=hidden_states.device)
                        m = torch.zeros(B, self.num_heads, L, device=hidden_states.device, dtype=torch.bool)
                    scores_list.append(s)
                    valid_masks.append(m)

                scores = torch.stack(scores_list, dim=-1)
                valid_mask = torch.stack(valid_masks, dim=-1)
                scores = scores.masked_fill(~valid_mask, float("-inf"))
                attn_weights = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)

                # 3. Gather refined values
                out_step = torch.zeros_like(q_rot)
                for k_idx, d in enumerate(self.offsets):
                    w_k = attn_weights[:, :, :, k_idx : k_idx + 1]
                    if d == 0:
                        out_step = out_step + w_k * value_states
                    elif d < L:
                        v_shifted = F.pad(value_states[:, :, :-d, :], (0, 0, d, 0))
                        out_step = out_step + w_k * v_shifted

                out_step = out_step.transpose(1, 2).contiguous().view(B, L, -1)
                proj_step = self.o_proj(out_step)

                # 4. Token-Level Early Halting & State Update
                s_next = s_curr + (gate * proj_step) * active_mask
                
                # Check convergence per token
                if self.training is False and step > 0:
                    delta = torch.norm(s_next - s_curr, p=2, dim=-1, keepdim=True) / (torch.norm(s_curr, p=2, dim=-1, keepdim=True) + 1e-5)
                    # Tokens with small state drift have converged -> freeze them
                    active_mask = active_mask & (delta >= self.eps_halt)
                    if not active_mask.any():
                        s_curr = s_next
                        break

                s_curr = s_next

            # Final attention output difference
            attn_output = s_curr - hidden_states
            return attn_output, None

    # -------------------------------------------------------------------------
    # 4. Convert All 24 Layers to Recurrent SubQ (T=6)
    # -------------------------------------------------------------------------
    print("\n[3/5] Transplanting 24 Layers to Recurrent SubQ Attention (T=6 Hops per Layer)...")
    rotary_emb = getattr(base_model.model, "rotary_emb", None)
    for i, layer in enumerate(base_model.model.layers):
        orig_attn = layer.self_attn
        layer.self_attn = RecurrentSubQQwenAttention(orig_attn, rotary_emb=rotary_emb, T_steps=T_RECUR)
    print(f"✅ All 24 layers converted to Recurrent SubQ (T={T_RECUR}, K={K}, Max Receptive Field = {T_RECUR * 64} tokens)!")

    # -------------------------------------------------------------------------
    # 5. Fine-Tune Recurrent SubQ-Qwen on GSM8K
    # -------------------------------------------------------------------------
    def format_prompt(item):
        return f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n{item['answer']}<|im_end|>"

    formatted_texts = [format_prompt(item) for item in train_data]
    tokenized_train = [tokenizer.encode(t, max_length=384, truncation=True) for t in formatted_texts]

    print("\n" + "=" * 135)
    print(f"  [4/5] TRAINING RECURRENT SUBQ-QWEN2.5-0.5B (T={T_RECUR}) ON GSM8K MATH REASONING")
    print("=" * 135)

    base_model.train()
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-4, weight_decay=0.01)
    
    total_steps = 600
    batch_size = 4
    grad_accum_steps = 4
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    step = 0
    t0 = time.time()
    accum_loss = 0.0

    while step < total_steps:
        indices = torch.randint(0, len(tokenized_train), (batch_size,))
        batch_seqs = [tokenized_train[idx] for idx in indices]
        max_b_len = max(len(s) for s in batch_seqs)
        
        input_ids = torch.full((batch_size, max_b_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_b_len), -100, dtype=torch.long, device=device)
        
        for b_idx, seq in enumerate(batch_seqs):
            input_ids[b_idx, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            labels[b_idx, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = base_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps

        loss.backward()
        accum_loss += loss.item() * grad_accum_steps

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 50 == 0:
            avg_l = accum_loss / 50.0
            ppl = math.exp(min(15.0, avg_l))
            elapsed = time.time() - t0
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"Step {step+1:>3}/{total_steps} | Loss: {avg_l:.4f} | Train PPL: {ppl:>6.2f} | VRAM: {peak_vram:.1f} MB | Elapsed: {elapsed:.1f}s")
            accum_loss = 0.0

        step += 1

    # Save Checkpoint
    save_path = "/root/checkpoints/subq_qwen2_5_05b_recurrent_t6_gsm8k.pt"
    print(f"\nSaving Recurrent SubQ-Qwen (T=6) checkpoint to {save_path}...")
    torch.save(base_model.state_dict(), save_path)
    vol.commit()

    # -------------------------------------------------------------------------
    # 6. Quantitative Evaluation on GSM8K Test Problems
    # -------------------------------------------------------------------------
    print("\n" + "=" * 135)
    print("  [5/5] EVALUATING RECURRENT SUBQ-QWEN (T=6) EXACT-MATCH ACCURACY ON GSM8K TEST SET")
    print("=" * 135)

    base_model.eval()

    def extract_answer_number(text: str) -> str:
        if "####" in text:
            ans_part = text.split("####")[-1].strip()
            match = re.search(r"(-?[\d,]+(?:\.\d+)?)", ans_part)
            if match:
                return match.group(1).replace(",", "")
        numbers = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
        if numbers:
            return numbers[-1].replace(",", "")
        return ""

    num_eval_samples = 150
    correct = 0
    total = 0
    has_delimiter = 0
    t0_eval = time.time()

    for idx, item in enumerate(test_data[:num_eval_samples]):
        question = item["question"]
        gold_answer_text = item["answer"]
        gold_num = extract_answer_number(gold_answer_text)

        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        curr_ids = input_ids.clone()
        generated_tokens = []

        with torch.no_grad():
            for _ in range(160):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = base_model(curr_ids)
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
                if next_token.item() == tokenizer.eos_token_id or "<|im_end|>" in tokenizer.decode([next_token.item()]):
                    break
                
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())

        gen_text = tokenizer.decode(generated_tokens)
        pred_num = extract_answer_number(gen_text)

        if pred_num != "" and pred_num == gold_num:
            correct += 1
        if "####" in gen_text:
            has_delimiter += 1
        total += 1

        if (idx + 1) % 25 == 0 or (idx + 1) == num_eval_samples:
            cur_acc = (correct / total) * 100.0
            print(f"  [Recurrent SubQ (T=6)] Progress {idx+1:>3}/{num_eval_samples} | Exact Accuracy: {cur_acc:>5.2f}% ({correct}/{total}) | Delimiter Adherence: {(has_delimiter/total)*100:.1f}%")

    eval_time = time.time() - t0_eval
    final_acc = (correct / total) * 100.0
    final_fmt = (has_delimiter / total) * 100.0

    print("\n" + "=" * 135)
    print("  STUDY 45 SCORECARD: RECURRENT SUBQ-QWEN (T=6) VS. PREVIOUS BASELINES")
    print("=" * 135)
    print(f"1. Dense Qwen2.5-0.5B (Standard Attention) :  11.33% (17/150) | 71.3% Delimiter")
    print(f"2. Feedforward SubQ-Qwen (T=1, K=8)         :   1.33% ( 2/150) | 72.7% Delimiter")
    print(f"3. Recurrent SubQ-Qwen   (T=6, K=8, Ponder) :  {final_acc:>5.2f}% ({correct}/{total}) | {final_fmt:.1f}% Delimiter")
    print("=" * 135)

@app.local_entrypoint()
def main():
    run_qwen_subq_recurrent_gsm8k.remote()
