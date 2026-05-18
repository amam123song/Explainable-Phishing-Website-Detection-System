#!/usr/bin/env python3
"""单体学生 Stage2：解释微调。仅供学术研究，禁止用于非法用途。"""

from __future__ import annotations

import argparse
import json
import os

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

DEFAULT_STUDENT_BASE = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_LORA_STEP1 = "outputs/monolithic/step1_cls"
DEFAULT_DATA = "data/dataset_explainable_200.json"
DEFAULT_OUTPUT = "outputs/monolithic/step2_explain"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ScamNet 蒸馏4 — 可解释阶段（Mistral）")
    p.add_argument("--student_base", type=str, default=DEFAULT_STUDENT_BASE)
    p.add_argument("--student_lora_init", type=str, default=DEFAULT_LORA_STEP1)
    p.add_argument("--data_path", type=str, default=DEFAULT_DATA)
    p.add_argument("--max_samples", type=int, default=None, help="仅用前 N 条（调试/试跑）")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_input_chars", type=int, default=12000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device_map", type=str, default="auto")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExplainDatasetScamNet(Dataset):
    """与 step2_train_explainable 一致：input + # Pred:\\n + output。"""

    def __init__(self, data: list, tokenizer, max_length: int, max_input_chars: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_input_chars = max_input_chars

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        raw_in = item["input"][: self.max_input_chars]
        raw_out = item["output"]

        # 长样本截断后若 prompt 占满 max_length，会导致无 label、loss 为 NaN。
        # 通过缩短 raw_in，保证至少保留 1 个输出 token 的监督。
        for _ in range(32):
            prompt = f"{raw_in}\n# Pred:\n"
            full = prompt + raw_out

            enc = self.tokenizer(
                full,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - 1,
            )["input_ids"]
            prompt_len = len(prompt_ids)

            labels = input_ids.clone()
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100

            valid = ((labels != -100) & attention_mask.bool()).sum().item()
            if valid == 0:
                raw_in = raw_in[: max(len(raw_in) * 9 // 10, 500)]
                continue
            break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("蒸馏4 — Step2 可解释微调（Mistral + Step1 LoRA）")
    print("=" * 60)

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(args.data_path)
    if not os.path.exists(args.student_lora_init):
        raise FileNotFoundError(
            f"请先完成 Step1 或指定 --student_lora_init，缺失: {args.student_lora_init}"
        )

    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if args.max_samples is not None:
        raw_data = raw_data[: args.max_samples]
    print(f"样本数: {len(raw_data)}")

    bnb = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.student_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载学生基座 + Step1 LoRA…")
    base = AutoModelForCausalLM.from_pretrained(
        args.student_base,
        quantization_config=bnb,
        device_map=args.device_map,
    )
    student = PeftModel.from_pretrained(
        base,
        args.student_lora_init,
        is_trainable=True,
    )
    # 显存优化：训练时关闭 KV cache + 开启梯度检查点
    student.config.use_cache = False
    if hasattr(student, "enable_input_require_grads"):
        student.enable_input_require_grads()
    elif hasattr(student, "get_input_embeddings"):
        student.get_input_embeddings().weight.requires_grad_(True)
    student.gradient_checkpointing_enable()
    student.print_trainable_parameters()
    device = next(student.parameters()).device

    dataset = ExplainDatasetScamNet(
        raw_data, tokenizer, args.max_length, args.max_input_chars
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    updates_per_epoch = (len(loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    total_steps = max(updates_per_epoch * args.epochs, 1)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    student.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_finite = 0
        skipped = 0
        bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        opt.zero_grad(set_to_none=True)
        for batch in bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue

            loss_to_backward = loss / args.gradient_accumulation_steps
            loss_to_backward.backward()

            step_idx = bar.n + 1
            do_update = (step_idx % args.gradient_accumulation_steps == 0) or (step_idx == len(loader))
            if do_update:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            n_finite += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / max(n_finite, 1)
        if skipped:
            print(f"  Epoch {epoch + 1}: 跳过非有限 loss 的 step 数: {skipped}")
        print(f"  Epoch {epoch + 1} 平均 loss: {avg:.4f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
