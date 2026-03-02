# train_grpo.py
# pip install transformers trl datasets torch accelerate

import math
import os

# Prevent double-vLLM OOM: fast_inference=False below means no Unsloth vLLM
# instance is pre-loaded, so we also keep use_vllm=False in GRPOConfig.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

MODEL_ID = "unsloth/gemma-3-270m-it"
TARGET_TOKENS = 20
SIGMA = 5

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_ID,
    max_seq_length = 2048,
    load_in_4bit   = False,
    load_in_8bit   = False,
    fast_inference = False,  # True pre-loads a vLLM engine; conflicts with GRPOTrainer's own vLLM
    max_lora_rank  = 8,
)
tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r                   = 4,
    target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    lora_alpha          = 16,
    lora_dropout        = 0,
    bias                = "none",
    use_gradient_checkpointing = "unsloth",
    random_state        = 42,
)


def reward_fn(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        tokens = tokenizer(completion, return_tensors="pt")["input_ids"][0]
        n = len(tokens)
        reward = math.exp(-((n - TARGET_TOKENS) ** 2) / (2 * SIGMA ** 2))
        rewards.append(float(reward))
    return rewards

dataset = load_dataset("openai/gsm8k", "main", split="train").select(range(500))
dataset = dataset.rename_column("question", "prompt")

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
    num_train_epochs            = 1,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 1,
    num_generations             = 4,
    max_prompt_length           = 512,
    max_completion_length       = 2048 - 512,
    temperature                 = 0.7,
    learning_rate               = 1e-4,
    optim                       = "adamw_8bit",
    beta                        = 0.01,
    epsilon                     = 1e-6,
    logging_steps               = 1,
    save_strategy               = "no",
    report_to                   = "wandb",
    # ── vLLM generation ──────────────────────────────────────────────────────
    # use_vllm=False: avoids double-vLLM OOM (fast_inference=True + use_vllm=True
    # each load a separate vLLM engine, exhausting the 15 GB VRAM budget).
    use_vllm                    = False,
    # ── Data loading ──────────────────────────────────────────────────────────
    dataloader_num_workers      = 4,
    dataloader_pin_memory       = True,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=[reward_fn],
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./grpo-gemma-final")
print("Done!")