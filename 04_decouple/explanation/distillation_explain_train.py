#!/usr/bin/env python3
"""解释 LoRA 蒸馏（CE on Label+解释）。仅供学术防御研究，禁止用于非法用途。"""

import os
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import PeftModel
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="ScamNet 可解释性蒸馏训练")

    p.add_argument(
        "--student_base",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="学生基座模型（HF repo id 或本地目录）",
    )
    p.add_argument(
        "--student_lora_init",
        type=str,
        default=str(REPO_ROOT / "outputs/decouple/cls_adapter"),
        help="第一阶段分类蒸馏得到的学生 LoRA 目录，作为初始化",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default=str(REPO_ROOT / "data/dataset_explainable_200.json"),
        help="包含 Label+解释 的 JSON 数据",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "outputs/decouple/explain_adapter"),
        help="可解释性蒸馏后的学生 LoRA 输出目录",
    )
    p.add_argument("--max_length", type=int, default=1024, help="最大序列长度（prompt+输出）")
    p.add_argument("--batch_size", type=int, default=2, help="batch size")
    p.add_argument("--epochs", type=int, default=3, help="训练轮数")
    p.add_argument("--lr", type=float, default=5e-6, help="学习率")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--device_map", type=str, default="auto", help="transformers device_map")
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExplainDataset(Dataset):
    """
    使用 dataset_explainable_200.json:
    - input: 原始 "input" 字段（含 URL/Content/External Links）
    - target: 完整 "output"（从 Label 到解释结束）
    prompt 形式：
      Website: <input[:N]>
    模型需要生成整个 output 文本。
    """

    def __init__(self, data, tokenizer, max_length: int = 1024, max_input_chars: int = 800):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_input_chars = max_input_chars

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_input = item["input"]
        raw_output = item["output"]

        # 构造 prompt & target 文本
        website_snippet = raw_input[: self.max_input_chars]
        prompt = f"Website: {website_snippet}\n"
        full_text = prompt + raw_output

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # prompt 长度，用于 mask 掉 prompt loss（只在 Label+解释上算 loss）
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_ids), self.max_length)

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # prompt 不参与 loss
        labels[attention_mask == 0] = -100  # padding 不参与 loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 50)
    print("ScamNet 第二阶段：可解释性蒸馏")
    print("=" * 50)

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据集不存在: {args.data_path}")
    if not os.path.exists(args.student_lora_init):
        raise FileNotFoundError(f"学生初始 LoRA 不存在: {args.student_lora_init}")
    print(f"数据集: {args.data_path}")
    print(f"学生初始 LoRA: {args.student_lora_init}")

    print("\n[1/4] 加载解释数据...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"  共 {len(raw_data)} 条样本")

    print("\n[2/4] 加载学生模型 (Mistral-7B + 第一阶段 LoRA)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.student_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student_base = AutoModelForCausalLM.from_pretrained(
        args.student_base,
        quantization_config=bnb,
        device_map=args.device_map,
    )
    # 从第一阶段 LoRA 初始化，并显式设为可训练
    student = PeftModel.from_pretrained(
        student_base,
        args.student_lora_init,
        is_trainable=True,
    )
    student.print_trainable_parameters()
    device = student.device
    print(f"  学生 device: {device}")

    print("\n[3/4] 准备训练数据...")
    dataset = ExplainDataset(raw_data, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"  数据集大小: {len(dataset)}, batches: {len(dataloader)}")

    print("\n[4/4] 开始解释蒸馏训练...")
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    student.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_loss_steps = 0
        skipped_nan_steps = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss

            if not torch.isfinite(loss):
                skipped_nan_steps += 1
                optimizer.zero_grad(set_to_none=True)
                progress.set_postfix({"loss": "nan(skip)"})
                continue

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += float(loss.item())
            total_loss_steps += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        denom = max(total_loss_steps, 1)
        avg = total_loss / denom
        if skipped_nan_steps > 0:
            print(f"  [警告] Epoch {epoch+1} 跳过 NaN/Inf steps: {skipped_nan_steps}/{len(dataloader)}")
        print(f"  Epoch {epoch+1} 平均损失(有限loss均值): {avg:.4f}")

    print("\n保存学生可解释性 LoRA...")
    os.makedirs(args.output_dir, exist_ok=True)
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  已保存到: {args.output_dir}")

    print("\n" + "=" * 50)
    print("可解释性蒸馏完成")
    print("=" * 50)


if __name__ == "__main__":
    main()

