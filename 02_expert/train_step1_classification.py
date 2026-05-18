#!/usr/bin/env python3
"""专家模型 Stage1：分类 SFT（Llama-3-8B LoRA）。仅供学术防御研究，禁止用于非法用途。"""

from __future__ import annotations

import sys
from pathlib import Path
import types
from importlib.machinery import ModuleSpec
import torch
from datasets import load_dataset

if not hasattr(torch, "_inductor"):
    torch._inductor = types.SimpleNamespace()
if not hasattr(torch._inductor, "config"):
    dummy_config = types.ModuleType("torch._inductor.config")
    dummy_config.__file__ = __file__
    torch._inductor.config = dummy_config

for missing in ("int1", "int2", "int3", "int4", "int5", "int6", "int7"):
    if not hasattr(torch, missing):
        setattr(torch, missing, torch.int8)

try:
    import torchao
except Exception:
    torchao = None

if torchao is None:
    torchao_stub = types.ModuleType("torchao")
    torchao_stub.__spec__ = ModuleSpec("torchao", loader=None)
    torchao_stub.__version__ = "0.0.0"
    quant_stub = types.ModuleType("torchao.quantization")
    quant_stub.__spec__ = ModuleSpec("torchao.quantization", loader=None)

    class _DummyConfig:
        pass

    quant_stub.Int4WeightOnlyConfig = _DummyConfig
    torchao_stub.quantization = quant_stub
    sys.modules["torchao"] = torchao_stub
    sys.modules["torchao.quantization"] = quant_stub
else:
    if not hasattr(torchao, "__version__"):
        torchao.__version__ = "0.0.0"

from unsloth import FastLanguageModel
import transformers
from transformers import TrainingArguments
from transformers.models.auto.processing_auto import AutoProcessor as _AutoProcessor
transformers.AutoProcessor = getattr(transformers, "AutoProcessor", _AutoProcessor)
from trl import SFTTrainer


MODEL_PATH = Path("meta-llama/Meta-Llama-3-8B-Instruct")
DATASET_PATH = Path("data/dataset_scamnet_5000.json")
OUTPUT_DIR = Path("outputs/expert/scamnet_step1_model")

MAX_SEQ_LENGTH = 2048

NUM_TRAIN_EPOCHS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
LOGGING_STEPS = 10
SAVE_STEPS = 50
WARMUP_STEPS = 20
LORA_RANK = 16
LORA_ALPHA = 32


def ensure_path(path: Path, kind: str) -> None:
    if not path.exists():
        print(f"[ERROR] {kind} Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)


def format_sample(sample: dict) -> dict:
    """拼接 # Pred: 与标签，Step1 为 completion 格式（非 chat template）。"""
    formatted_text = (
        f"{sample['input']}\n"
        f"# Pred:\n"
        f"{sample['output']}"
        f"<|eot_id|>"
    )
    
    return {"text": formatted_text}


def main() -> None:
    print("=" * 60)
    print("ScamNet Step 1: Initial Fine-Tuning (Classification Only)")
    print("=" * 60)
    
    ensure_path(MODEL_PATH, "Base Model")
    ensure_path(DATASET_PATH, "Dataset")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    print(f"\nLoading Model from: {MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_PATH),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Add LoRA Adapters
    print(f"\nAdding LoRA Adapters (Rank={LORA_RANK}, Alpha={LORA_ALPHA})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0, 
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # Load Dataset
    print(f"\nLoading Dataset: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=str(DATASET_PATH))["train"]
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Apply Formatting
    print("Formatting dataset for ScamNet task...")
    dataset = dataset.map(format_sample)
    # Print a sample to verify format
    print("\n[Sample Prompt]:")
    print("-" * 40)
    print(dataset[0]["text"])
    print("-" * 40)
    
    # Training Config
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none", # Disable wandb
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
    
    print("\nStarting Training (Step 1)...")
    trainer_stats = trainer.train()
    
    print("\nSaving Step 1 Model...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"✓ Saved to {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("Running Inference Test (Classification)")
    print("=" * 60)
    
    FastLanguageModel.for_inference(model)
    
    # Test Input (No CoT, just raw format)
    test_prompt = (
        "# Information:\n"
        "## URL:\nhttp://verify-apple-id-security.com/login\n"
        "## Content:\nApple ID Login. Please verify your identity to continue using iCloud services. Enter username and password.\n"
        "## External Links:\nNone\n"
        "\n"
        "# Pred:\n"
    )
    
    print(f"Test Prompt:\n{test_prompt}")
    
    inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=20, # We expect a very short answer: "Label: scam"
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.batch_decode(outputs)
    print("\nGenerated Output:")
    print(result[0])
    print("-" * 60)

if __name__ == "__main__":
    main()