#!/usr/bin/env python3
"""专家模型 Stage2：解释 SFT。仅供学术防御研究，禁止用于非法用途。"""

from __future__ import annotations
import sys
from pathlib import Path
import types
from importlib.machinery import ModuleSpec
import torch
from datasets import load_dataset

for missing in ("int1", "int2", "int3", "int4", "int5", "int6", "int7"):
    if not hasattr(torch, missing):
        setattr(torch, missing, torch.int8)

try:
    import torchao  # type: ignore
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

if not hasattr(torch, "_inductor"):
    torch._inductor = types.SimpleNamespace()
if not hasattr(torch._inductor, "config"):
    dummy_config = types.ModuleType("torch._inductor.config")
    dummy_config.__file__ = __file__
    torch._inductor.config = dummy_config

from unsloth import FastLanguageModel

import transformers
from transformers import TrainingArguments
from transformers.models.auto.processing_auto import AutoProcessor as _AutoProcessor
transformers.AutoProcessor = getattr(transformers, "AutoProcessor", _AutoProcessor)

from trl import SFTTrainer

MODEL_PATH = Path("outputs/expert/scamnet_step1_model")
DATASET_PATH = Path("data/dataset_explainable_200.json")

OUTPUT_DIR = Path("outputs/expert/scamnet_final_model")
MAX_SEQ_LENGTH = 2048

NUM_TRAIN_EPOCHS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
LOGGING_STEPS = 5
WARMUP_STEPS = 10
LORA_RANK = 32
LORA_ALPHA = 64

def format_sample(sample: dict) -> dict:
    formatted_text = (
        f"{sample['input']}\n"
        f"# Pred:\n"
        f"{sample['output']}"
        f"<|eot_id|>"
    )
    return {"text": formatted_text}

def main():
    print("=" * 60)
    print("ScamNet Step 2: Explainable Fine-Tuning")
    print("=" * 60)
    
    # Load Merged Model
    print(f"\nLoading Merged Model from: {MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_PATH),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype="bfloat16",    # 直接 16bit 加载，避免 4bit 编译路径异常
        load_in_4bit=False,
    )
    
    # Add NEW LoRA Adapters for Step 2
    print(f"\nAdding New LoRA Adapters (Rank={LORA_RANK}, LR={LEARNING_RATE})...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK, # Rank 32
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    except TypeError as e:
        # 如果基座已带 LoRA（Step1 适配器），则跳过新增适配器，直接继续训练现有 LoRA
        print(f"[Warning] Skip adding new LoRA because base already has adapters: {e}")
    
    dataset = load_dataset("json", data_files=str(DATASET_PATH))["train"]
    dataset = dataset.map(format_sample)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE, # 1e-5
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
    
    print("\nStarting Training (Step 2)...")
    trainer.train()
    
    print("\nSaving Final ScamNet Model...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"✓ Saved to {OUTPUT_DIR}")

    # Inference Test (Checking for Explanations)
    print("\n" + "=" * 60)
    print("Running Inference Test (Explanation Generation)")
    print("=" * 60)
    
    FastLanguageModel.for_inference(model)
    test_prompt = (
        "# Information:\n"
        "## URL:\nhttp://suspicious-bank-login.com\n"
        "## Content:\nVerify your account immediately.\n"
        "## External Links:\nNone\n"
        "\n"
        "# Pred:\n"
    )
    
    inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128, # 需要更长的 token 来生成解释
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    print(tokenizer.batch_decode(outputs)[0])

if __name__ == "__main__":
    main()