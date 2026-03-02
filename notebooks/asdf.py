# %%
import re
import os

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig


# %%
# ── Precision toggle ──────────────────────────────────────────────────────────
# RTX 5070 Ti (Blackwell) has native FP8 tensor cores → best speed + VRAM ratio.
LOAD_IN_FP8  = True    # ← Blackwell sm_120: requires CUDA 12.8+ and unsloth-nightly
LOAD_IN_4BIT = False   # fallback if FP8 errors; flip to True and set FP8=False

# ── Training config ───────────────────────────────────────────────────────────
MODEL_NAME  = "HuggingFaceTB/SmolLM-135M-Instruct"
G           = 8        # completions per prompt — 8 is sufficient for 135M, was 16
MAX_SEQ     = 2048     # max sequence length (prompt + completion); was 8192
MAX_NEW     = 512      # max new tokens per completion; GSM8K rarely needs >300
LR          = 1e-4
EPOCHS      = 1
CLIP_EPS    = 0.2
TEMPERATURE = 0.9
TRAIN_SIZE  = 500
EVAL_SIZE   = 50

LORA_R      = 8
LORA_ALPHA  = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

# ── Env vars for Unsloth + vLLM ───────────────────────────────────────────────
# UNSLOTH_VLLM_STANDBY=1 keeps the vLLM engine resident between training steps
# (swap mode) so generation starts immediately — eliminates the 3-min idle time.
os.environ["UNSLOTH_VLLM_STANDBY"] = "0"

# Clear any expandable_segments setting — incompatible with Unsloth CuMemAllocator.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | FP8={LOAD_IN_FP8} | 4-bit={LOAD_IN_4BIT}")


# %%
# ── Dataset helpers ───────────────────────────────────────────────────────────
def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K solution string.
    GSM8K solutions end with '#### <number>'."""
    match = re.search(r"####\s*([\d,\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve problems step by step. "
    "At the end, write your final answer after '####'."
)


def build_prompt(question: str) -> list[dict]:
    """Return a chat messages list for a GSM8K question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]


# %%
# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading GSM8K dataset...")
raw = load_dataset("openai/gsm8k", "main")


def make_hf_dataset(split: str, size: int) -> Dataset:
    rows = []
    for ex in raw[split].select(range(size)):
        ans = extract_answer(ex["answer"])
        if ans is not None:
            rows.append({"prompt": build_prompt(ex["question"]), "answer": ans})
    return Dataset.from_list(rows)


train_ds = make_hf_dataset("train", TRAIN_SIZE)
eval_ds  = make_hf_dataset("test",  EVAL_SIZE)
print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")


# %%
# ── Reward function ───────────────────────────────────────────────────────────
def reward_fn(completions: list[str], *, answer: list[str], **kwargs) -> list[float]:
    """Rubric reward for GSM8K.

    Correctness:
      +1.00  exact match on the '#### <n>' final answer
      +0.50  gold number appears as a whole word anywhere in the completion

    Format / structure:
      +0.20  uses '#### <answer>' format (even if answer is wrong)
      +0.15  shows arithmetic work: expression like '3 * 4' or '12 / 3'
      +0.15  shows equation results: writes '= <number>' at least once
      +0.10  answer within 2× of the gold value (order-of-magnitude credit)
      +0.05  has 3+ distinct numeric values (multi-step reasoning indicator)
      +0.05  contains any digit (minimum partial credit)

    Penalty:
      -0.20 max  length penalty scales linearly with word count up to MAX_NEW
    """
    rewards = []
    for completion, gold_answer in zip(completions, answer):
        points = 0.0
        # normalize: GRPO passes list-of-dicts; direct eval calls pass plain strings
        if isinstance(completion, list):
            text = completion[0]["content"] if isinstance(completion[0], dict) else str(completion[0])
        else:
            text = completion
        print(text)
        pred = extract_answer(text)

        if pred is not None and pred == gold_answer:
            points += 1.0
        if re.search(rf"\b{re.escape(gold_answer)}\b", text):
            points += 0.5

        if pred is not None:
            points += 0.2
        if re.search(r"\d+\s*[+\-*/×÷]\s*\d+", text):
            points += 0.15
        if re.search(r"=\s*\d+", text):
            points += 0.15

        try:
            gold_val = float(gold_answer)
            pred_val = float(pred) if pred is not None else None
            if pred_val is not None and gold_val != 0 and 0.5 <= pred_val / gold_val <= 2.0:
                points += 0.10
        except (ValueError, ZeroDivisionError):
            pass

        distinct_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        if len(distinct_nums) >= 3:
            points += 0.05

        if re.search(r"\d", text):
            points += 0.05

        num_words = len(text.split())
        length_penalty = 0.2 * (num_words / MAX_NEW)
        rewards.append(max(0.0, min(points, 1.0) - length_penalty))
    return rewards


print("Reward function defined.")


# %%
# ── Model ─────────────────────────────────────────────────────────────────────
# SmolLM-135M is a standard LlamaForCausalLM → fast_inference=True is safe.
# This enables Unsloth's vLLM engine path, which eliminates the startup idle
# caused by lazy vLLM init during the first training step.
#
# max_seq_length matches MAX_SEQ (2048) so KV cache isn't over-allocated.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ,
    load_in_4bit   = LOAD_IN_4BIT,
    load_in_8bit   = LOAD_IN_FP8,
    fast_inference = True,   # enables vLLM engine; was False (only needed for Granite)
    max_lora_rank  = LORA_R,
)
tokenizer.pad_token = tokenizer.eos_token

print(f"Loaded: {MODEL_NAME}")
print(f"Dtype:  {next(model.parameters()).dtype}")

# %% Pre-train eval
FastLanguageModel.for_inference(model)
sample = eval_ds[0]
input_ids = tokenizer.apply_chat_template(
    sample["prompt"], add_generation_prompt=True, return_tensors="pt"
).to(DEVICE).repeat_interleave(G, dim=0)  # GRPO group size G
with torch.inference_mode():
    out = model.generate(
        input_ids        = input_ids,
        max_new_tokens   = MAX_NEW,
        do_sample        = True,
        pad_token_id     = tokenizer.eos_token_id,
    )
for i in range(G):
    response = tokenizer.decode(out[i, input_ids.shape[1]:], skip_special_tokens=True)
    gold     = sample["answer"]
    print(f"Sample {i+1} | Gold {gold!r}:\n{response}\n{'─'*70}")
    pre_r    = reward_fn([response], answer=[gold])[0]
    print(f"Sample {i+1} | Gold {gold!r} Reward {pre_r:.2f}:\n{response}\n{'─'*70}")


# %%
# ── LoRA adapter ──────────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r                         = LORA_R,
    lora_alpha                = LORA_ALPHA,
    target_modules            = LORA_TARGET_MODULES,
    lora_dropout              = 0,
    bias                      = "none",
    use_gradient_checkpointing = "unsloth",
    random_state              = 42,
)
model.print_trainable_parameters()


# %%
# ── GRPOTrainer ───────────────────────────────────────────────────────────────
# Key changes vs original:
#   • use_vllm=True  — vLLM handles generation; 3-5× faster than HF generate
#   • vllm_gpu_memory_utilization=0.4 — enough for 135M; leaves headroom for training
#   • per_device_train_batch_size=8, grad_accum=1 — higher throughput, 135M is tiny
#   • num_generations=G (8 vs 16) — halves generation cost
#   • max_completion_length=MAX_NEW (512) — matches max_seq_length budget
#   • dataloader_num_workers=4 — overlaps data loading with GPU work
training_args = GRPOConfig(
    output_dir                  = "./grpo-lora",
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 1,
    num_generations             = G,
    max_completion_length       = MAX_NEW + 10,  # +10 buffer for special tokens
    temperature                 = TEMPERATURE,
    learning_rate               = LR,
    optim                       = "adamw_8bit",
    beta                        = 0.01,
    epsilon                     = CLIP_EPS,
    logging_steps               = 1,
    save_strategy               = "no",
    report_to                   = "none",
    # ── vLLM generation (the big speedup) ────────────────────────────────────
    use_vllm                    = True,
    vllm_gpu_memory_utilization = 0.4,   # tune up if you have VRAM headroom
    # ── Data loading ──────────────────────────────────────────────────────────
    dataloader_num_workers      = 4,
    dataloader_pin_memory       = True,
)

trainer = GRPOTrainer(
    model         = model,
    tokenizer     = tokenizer,
    reward_funcs  = [reward_fn],
    args          = training_args,
    train_dataset = train_ds,
)

print("Starting GRPO training...")
trainer.train()
print("Training complete.")


# %%
# ── Post-training eval ────────────────────────────────────────────────────────
FastLanguageModel.for_inference(model)

total_reward = 0.0
for item in eval_ds:
    input_ids = tokenizer.apply_chat_template(
        item["prompt"], add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():
        out = model.generate(
            input_ids      = input_ids,
            max_new_tokens = MAX_NEW,
            do_sample      = False,
            pad_token_id   = tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    r = reward_fn([completion], answer=[item["answer"]])[0]
    total_reward += r
    print(f"  [r={r:.2f}] gold={item['answer']!r} | {completion[:80]!r}")

post_acc = total_reward / len(eval_ds)
print(f"Post-train mean reward: {post_acc:.4f}")


# %%
# ── Save ──────────────────────────────────────────────────────────────────────
model.save_pretrained("./grpo-lora")
tokenizer.save_pretrained("./grpo-lora")
print("Saved LoRA adapter to ./grpo-lora")



