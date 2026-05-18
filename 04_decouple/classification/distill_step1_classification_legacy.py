#!/usr/bin/env python3
"""分类蒸馏 legacy（全词表 KL，需 teacher/student 词表一致）。仅供学术研究，禁止用于非法用途。"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step1 分类蒸馏训练脚本")

    # 数据
    p.add_argument(
        "--data_path",
        type=str,
        default="data/dataset_scamnet_5000.json",
        help="训练数据 JSON（list[{'input','output'}]）路径（例如 dataset_scamnet_5000.json）",
    )
    p.add_argument("--max_samples", type=int, default=None, help="只取前 N 条训练（可用于快速验证）")
    p.add_argument("--shuffle_seed", type=int, default=3407, help="采样前是否打乱（None 表示不打乱）")

    # 教师/学生
    p.add_argument(
        "--base_model_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="本地基础模型目录（teacher/student 共用时可启用 KL 蒸馏）",
    )
    p.add_argument(
        "--teacher_lora_path",
        type=str,
        default="outputs/expert/scamnet_final_model",
        help="教师 LoRA 适配器目录（含 adapter_config.json）",
    )
    p.add_argument(
        "--student_base_model_path",
        type=str,
        default=None,
        help="学生基础模型（可与 base_model_path 不同；不同则可能词表不一致，KL 将自动禁用）",
    )

    # 输出
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/decouple/cls_adapter",
        help="保存学生 LoRA 适配器与 tokenizer 的目录",
    )

    # 训练超参
    p.add_argument("--temperature", type=float, default=4.0, help="蒸馏温度 T")
    p.add_argument("--alpha", type=float, default=0.7, help="硬损失占比 alpha（0~1）")
    p.add_argument("--lora_r", type=int, default=8, help="学生 LoRA rank")
    p.add_argument("--batch_size", type=int, default=2, help="batch size")
    p.add_argument("--epochs", type=int, default=2, help="训练轮数")
    p.add_argument("--lr", type=float, default=1e-4, help="学习率")
    p.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup 比例（总步数的比例）")
    p.add_argument("--max_length", type=int, default=512, help="最大序列长度（只用于分类蒸馏，不必太大）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")

    # 类别权重（可选）
    p.add_argument("--w_legit", type=float, default=1.0, help="legit(0) 的 CE 权重")
    p.add_argument("--w_scam", type=float, default=1.0, help="scam(1) 的 CE 权重")

    # 环境/调试
    p.add_argument(
        "--tiny_test",
        action="store_true",
        help="CPU + tiny-gpt2 跑通流程（不加载 LoRA/4bit），只用于验证代码能运行",
    )

    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_exists(path: str, desc: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} 不存在: {path}")


@dataclass
class Example:
    input_text: str
    label_word: str  # "scam" / "legit"
    class_id: int  # 1 / 0


def _extract_label(output_text: str) -> tuple[str, int]:
    first_line = (output_text or "").split("\n")[0].strip().lower()
    if "scam" in first_line:
        return "scam", 1
    if "legit" in first_line:
        return "legit", 0
    # 兜底
    return "scam", 1


class ScamnetClsDistillDataset(Dataset):
    """
    输入样本格式（与你现有 dataset_scamnet_5000.json 对齐）:
      - sample["input"]: 以 "# Information:" 开头的长文本
      - sample["output"]: "Label: scam" 或 "Label: legit"（可含后续解释，但第一行必须是 Label）

    构造训练序列:
      prompt = sample["input"] + "\n# Pred:\n"
      completion = "Label: scam" / "Label: legit" + "<|eot_id|>"（如果 tokenizer 有该 token）

    只在 completion token 上计算 loss（prompt 与 padding 的 label 置为 -100）。
    """

    def __init__(self, raw: list[dict], tokenizer, max_length: int):
        self.examples: list[Example] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for item in raw:
            in_txt = item.get("input", "")
            out_txt = item.get("output", "")
            label_word, class_id = _extract_label(out_txt)
            self.examples.append(Example(input_text=in_txt, label_word=label_word, class_id=class_id))

        # 预先缓存常用 token id（可能不存在）
        self._eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if self._eot_id is None or self._eot_id == tokenizer.unk_token_id:
            self._eot_id = None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]

        prompt = f"{ex.input_text}\n# Pred:\n"
        completion = f"Label: {ex.label_word}"

        # 注意：很多 tokenizer 会在输入超过 model_max_length 时打印 warning。
        # 这里显式启用 truncation，避免日志噪音；最终长度仍由后续 keep_prompt 逻辑严格控制。
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max(self.max_length * 8, 4096),
        )["input_ids"]
        completion_ids = self.tokenizer(
            completion,
            add_special_tokens=False,
            truncation=True,
            max_length=max(self.max_length, 64),
        )["input_ids"]
        if len(completion_ids) == 0:
            completion_ids = [self.tokenizer.eos_token_id]

        # 添加 eot（如果存在）
        if self._eot_id is not None:
            completion_ids = completion_ids + [self._eot_id]

        # 裁剪 prompt，使总长度不超过 max_length
        keep_prompt = max(self.max_length - len(completion_ids), 0)
        prompt_ids = prompt_ids[:keep_prompt]

        input_ids_list = prompt_ids + completion_ids

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        attention_mask_list = [1] * len(input_ids_list)
        if len(input_ids_list) < self.max_length:
            pad_len = self.max_length - len(input_ids_list)
            input_ids_list = input_ids_list + [pad_id] * pad_len
            attention_mask_list = attention_mask_list + [0] * pad_len
        else:
            input_ids_list = input_ids_list[: self.max_length]
            attention_mask_list = attention_mask_list[: self.max_length]

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), self.max_length)
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "class_id": torch.tensor(ex.class_id, dtype=torch.long),
        }


def load_json_list(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"数据格式不对：期望 JSON 为 list，但得到 {type(data)}: {path}")
    return data


def build_lora(model, r: int, tiny_test: bool) -> torch.nn.Module:
    if tiny_test:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=r * 2,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, cfg)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("两阶段蒸馏 Step1：分类蒸馏")
    print("=" * 60)

    # 0) 路径检查
    print("\n[0/6] 检查路径...")
    assert_exists(args.data_path, "训练数据集")
    if not args.tiny_test:
        assert_exists(args.base_model_path, "基础模型目录")
        assert_exists(args.teacher_lora_path, "教师 LoRA 目录")
        assert_exists(os.path.join(args.teacher_lora_path, "adapter_config.json"), "教师 adapter_config.json")
        if args.student_base_model_path is not None:
            assert_exists(args.student_base_model_path, "学生基础模型目录")
    print("  路径检查通过")

    # 1) 加载数据
    print("\n[1/6] 加载数据...")
    raw = load_json_list(args.data_path)
    if args.shuffle_seed is not None:
        rng = random.Random(args.shuffle_seed)
        raw = raw.copy()
        rng.shuffle(raw)
        print(f"  已按 seed={args.shuffle_seed} 打乱")
    if args.max_samples is not None:
        raw = raw[: args.max_samples]
        print(f"  使用样本数: {len(raw)}")
    else:
        print(f"  样本数: {len(raw)}")

    # 2) 加载 tokenizer（默认跟随学生基座；若 student_base_model_path 为空则用 base_model_path）
    print("\n[2/6] 加载 tokenizer...")
    if args.tiny_test:
        tok_name = "sshleifer/tiny-gpt2"
    else:
        tok_name = args.student_base_model_path or args.base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  tokenizer OK")

    # 3) 加载 teacher/student
    print("\n[3/6] 加载模型...")
    if args.tiny_test:
        teacher = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        teacher.eval()
        student = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        student = build_lora(student, r=args.lora_r, tiny_test=True)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True)

        # teacher = base + lora
        base_teacher = AutoModelForCausalLM.from_pretrained(
            args.base_model_path, device_map="auto", quantization_config=bnb
        )
        teacher = PeftModel.from_pretrained(base_teacher, args.teacher_lora_path)
        teacher.eval()

        # student = (student_base or base) + new lora
        student_base = args.student_base_model_path or args.base_model_path
        base_student = AutoModelForCausalLM.from_pretrained(
            student_base, device_map="auto", quantization_config=bnb
        )
        student = build_lora(base_student, r=args.lora_r, tiny_test=False)

    student.print_trainable_parameters()
    print("  模型加载完成")

    # 4) dataloader
    print("\n[4/6] 准备数据 loader...")
    ds = ScamnetClsDistillDataset(raw, tokenizer, max_length=args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    print(f"  dataset={len(ds)}, batches={len(dl)}")

    # 5) 训练
    print("\n[5/6] 开始训练...")
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    total_steps = max(len(dl) * args.epochs, 1)
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    temperature = float(args.temperature)
    alpha = float(args.alpha)
    w_legit = float(args.w_legit)
    w_scam = float(args.w_scam)

    # 只提示一次
    warned_vocab = False

    for ep in range(args.epochs):
        running = 0.0
        pbar = tqdm(dl, desc=f"Epoch {ep + 1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(student.device)
            attention_mask = batch["attention_mask"].to(student.device)
            labels = batch["labels"].to(student.device)
            class_id = batch["class_id"].to(student.device)  # [B]

            # forward student
            out_s = student(input_ids=input_ids, attention_mask=attention_mask)
            s_logits = out_s.logits  # [B, L, V]

            # forward teacher (no grad)
            with torch.no_grad():
                out_t = teacher(input_ids=input_ids, attention_mask=attention_mask)
                t_logits = out_t.logits

            # soft loss（仅 labels != -100 的位置；若词表不一致则禁用）
            if s_logits.size(-1) != t_logits.size(-1):
                soft_loss = torch.tensor(0.0, device=student.device)
                if not warned_vocab:
                    warned_vocab = True
                    print(
                        f"\n  [提示] teacher/student 词表大小不一致，禁用 KL 蒸馏: "
                        f"{t_logits.size(-1)} vs {s_logits.size(-1)}。将退化为纯硬监督。"
                    )
            else:
                mask = labels.ne(-100)  # [B, L]
                if mask.any():
                    s_flat = s_logits[mask]
                    t_flat = t_logits[mask]
                    soft_loss = F.kl_div(
                        F.log_softmax(s_flat / temperature, dim=-1),
                        F.softmax(t_flat / temperature, dim=-1),
                        reduction="batchmean",
                    ) * (temperature**2)
                else:
                    soft_loss = torch.tensor(0.0, device=student.device)

            # hard loss（token 级别 CE -> per-sample mean -> 类别加权）
            vocab = s_logits.size(-1)
            token_loss = F.cross_entropy(
                s_logits.view(-1, vocab),
                labels.view(-1),
                reduction="none",
            ).view_as(labels)  # [B, L]
            mask = labels.ne(-100)
            token_loss = token_loss * mask
            token_count = mask.sum(dim=1).clamp_min(1)
            loss_per_sample = token_loss.sum(dim=1) / token_count
            class_w = torch.where(class_id == 1, torch.tensor(w_scam, device=student.device), torch.tensor(w_legit, device=student.device))
            hard_loss = (loss_per_sample * class_w).mean()

            loss = alpha * hard_loss + (1.0 - alpha) * soft_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "hard": f"{hard_loss.item():.4f}", "soft": f"{soft_loss.item():.4f}"})

        print(f"  Epoch {ep + 1} 平均 loss: {running / max(len(dl), 1):.4f}")

    # 6) 保存
    print("\n[6/6] 保存学生 LoRA...")
    os.makedirs(args.output_dir, exist_ok=True)
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  已保存到: {args.output_dir}")

    print("\n完成。")


if __name__ == "__main__":
    main()

