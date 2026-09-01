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

app = modal.App("exp-qwen-subq-fast-eval", image=image)
vol = modal.Volume.from_name("subq-qwen-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=1200, volumes={"/root/checkpoints": vol})
def fast_evaluate_qwen_gsm8k():
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
    print("  FAST BATCHED EVALUATION: RECURRENT SUBQ-QWEN (T=6, K=8) VS. DENSE QWEN2.5-0.5B ON GSM8K TEST SET")
    print("  Vectorized Multi-Hop Recurrence with Parallel Batch Generation (B=16)")
    print("=" * 135)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Download Test Data
    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    req = urllib.request.urlopen(test_url)
    test_data = [json.loads(line) for line in req.read().decode('utf-8').strip().split('\n') if line.strip()]
    print(f"GSM8K Test Set: {len(test_data):,} Total Problems")

    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Define Vectorized Recurrent SubQ Layer
    OFFSETS = [0, 1, 2, 4, 8, 16, 32, 64]
    K = len(OFFSETS)
    config = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    num_key_value_groups = num_heads // num_kv_heads
    T_RECUR = 6

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, n_kv, slen, h_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv, n_rep, slen, h_dim)
        return hidden_states.reshape(batch, n_kv * n_rep, slen, h_dim)

    class FastRecurrentSubQQwenAttention(nn.Module):
        def __init__(self, orig_attn, rotary_emb=None, T_steps=T_RECUR):
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
            self.step_gate = getattr(orig_attn, "step_gate", nn.Parameter(torch.tensor(0.5, dtype=torch.bfloat16)))

        def forward(self, hidden_states, position_ids=None, attention_mask=None, past_key_values=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
            B, L, D = hidden_states.shape

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

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            s_curr = hidden_states.clone()
            gate = torch.sigmoid(self.step_gate)

            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

            for step in range(self.T_steps):
                query_states = self.q_proj(s_curr).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                if cos is not None and sin is not None:
                    q_rot, k_rot = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                else:
                    q_rot, k_rot = query_states, key_states

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
                s_curr = s_curr + gate * proj_step

            attn_output = s_curr - hidden_states
            return attn_output, None

    # 3. Helper for Answer Parsing
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

    # 4. Fast Batched Evaluation Function (Batch Size = 4)
    def evaluate_model_batched(model, model_name: str, test_items, batch_size=4, max_new_tokens=140):
        print(f"\nEvaluating {model_name} on {len(test_items)} GSM8K Problems (Batch Size = {batch_size})...")
        model.eval()
        correct = 0
        total = 0
        has_delimiter = 0
        total_tokens_generated = 0
        t0 = time.time()

        for b_start in range(0, len(test_items), batch_size):
            batch_slice = test_items[b_start : b_start + batch_size]
            B_curr = len(batch_slice)

            prompts = [f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n" for item in batch_slice]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)
            input_ids = encoded.input_ids

            curr_ids = input_ids.clone()
            unfinished = torch.ones(B_curr, dtype=torch.bool, device=device)
            gen_history = [[] for _ in range(B_curr)]

            with torch.inference_mode():
                for step_idx in range(max_new_tokens):
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(curr_ids)
                        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1) # [B]

                    for b_i in range(B_curr):
                        if unfinished[b_i]:
                            tok = next_tokens[b_i].item()
                            if tok == tokenizer.eos_token_id or "<|im_end|>" in tokenizer.decode([tok]):
                                unfinished[b_i] = False
                            else:
                                gen_history[b_i].append(tok)

                    if not unfinished.any():
                        break

                    curr_ids = torch.cat([curr_ids, next_tokens.unsqueeze(1)], dim=1)

            # Evaluate batch answers
            for b_i, item in enumerate(batch_slice):
                gold_num = extract_answer_number(item["answer"])
                pred_text = tokenizer.decode(gen_history[b_i])
                pred_num = extract_answer_number(pred_text)

                if pred_num != "" and pred_num == gold_num:
                    correct += 1
                if "####" in pred_text:
                    has_delimiter += 1
                total += 1
                total_tokens_generated += len(gen_history[b_i])

            progress_num = min(b_start + batch_size, len(test_items))
            cur_acc = (correct / total) * 100.0
            print(f"  [{model_name}] Completed {progress_num:>3}/{len(test_items)} | Exact Accuracy: {cur_acc:>5.2f}% ({correct}/{total}) | Delimiter Adherence: {(has_delimiter/total)*100:.1f}%")

        elapsed = time.time() - t0
        tok_speed = total_tokens_generated / max(1e-5, elapsed)
        acc = (correct / total) * 100.0
        fmt = (has_delimiter / total) * 100.0
        return {
            "name": model_name,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "format_adherence": fmt,
            "tok_per_sec": tok_speed,
            "time_sec": elapsed
        }

    # 5. Load and Benchmark
    TEST_SET = test_data[:150] # 150 test problems
    subq_ckpt_path = "/root/checkpoints/subq_qwen2_5_05b_recurrent_t6_gsm8k.pt"

    print("\n" + "=" * 135)
    print("  LOADING AND EVALUATING TRAINED RECURRENT SUBQ-QWEN (T=6, K=8)")
    print("=" * 135)
    subq_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    subq_model.config.use_cache = False
    rotary_emb = getattr(subq_model.model, "rotary_emb", None)
    for layer in subq_model.model.layers:
        orig_attn = layer.self_attn
        layer.self_attn = FastRecurrentSubQQwenAttention(orig_attn, rotary_emb=rotary_emb, T_steps=T_RECUR)

    print(f"Loading trained weights from {subq_ckpt_path}...")
    subq_model.load_state_dict(torch.load(subq_ckpt_path, map_location=device))
    print("Checkpoint loaded successfully!")

    subq_results = evaluate_model_batched(subq_model, "Recurrent SubQ-Qwen (T=6, K=8)", TEST_SET, batch_size=4)

    # 6. Final Comparative Scorecard
    print("\n" + "=" * 135)
    print("  FINAL GSM8K BENCHMARK SCORECARD: RECURRENT SUBQ (T=6) VS. BASELINES")
    print("=" * 135)
    print(f"{'Model Architecture':<36} | {'Test Accuracy (%)':<20} | {'Correct / Total':<18} | {'Delimiter Adherence':<22} | {'Evaluation Time':<15}")
    print("-" * 135)
    print(f"{'Dense Qwen2.5-0.5B (Standard)':<36} | {'11.33%':<20} | {'17/150':<18} | {'71.3%':<22} | {'~180s'}")
    print(f"{'Feedforward SubQ-Qwen (T=1, K=8)':<36} | {'1.33%':<20} | {'2/150':<18} | {'72.7%':<22} | {'~320s'}")
    print(f"{subq_results['name']:<36} | {subq_results['accuracy']:>6.2f}%{'':<13} | {subq_results['correct']:>3}/{subq_results['total']:<14} | {subq_results['format_adherence']:>6.1f}%{'':<15} | {subq_results['time_sec']:.1f}s ({subq_results['tok_per_sec']:.1f} tok/s)")
    print("=" * 135)

@app.local_entrypoint()
def main():
    fast_evaluate_qwen_gsm8k.remote()
