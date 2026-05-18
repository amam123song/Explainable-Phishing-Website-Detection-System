#!/usr/bin/env python3
"""分类 LoRA 二类 logits 蒸馏（Mistral 学生，teacher/student 词表不一致时用 2 维 KL）。仅供学术研究，禁止用于非法用途。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

warnings.filterwarnings("ignore")

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from config.paths import hf_local_files_only

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step1 分类蒸馏（Mistral 学生）")

    # 数据
    p.add_argument("--data_path", type=str, default=str(REPO_ROOT / "data/dataset_scamnet_5000.json"))
    p.add_argument(
        "--expect_soft2_field",
        type=str,
        default=None,
        help="若提供（如 teacher_soft2），则从数据里读取该字段作为 teacher 概率，训练时不会加载 teacher 模型。",
    )
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--shuffle_seed", type=int, default=3407)

    # teacher
    p.add_argument(
        "--teacher_base_model_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    p.add_argument(
        "--teacher_lora_path",
        type=str,
        default="outputs/expert/scamnet_final_model",
    )

    # student
    p.add_argument(
        "--student_base",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="学生基座（Hub ID 或本地目录）",
    )
    p.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "outputs/decouple/cls_adapter"))

    # distill
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.7)

    # train
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    # 分类蒸馏只需要较短上下文即可（越长越吃显存且两次前向更重）
    p.add_argument("--max_prompt_chars", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)

    # debug
    p.add_argument("--tiny_test", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_exists(path: str, desc: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} 不存在: {path}")


def load_json_list(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 list JSON，实际为 {type(data)}: {path}")
    return data


def extract_hard_label(output_text: str) -> int:
    head = (output_text or "").split("\n", 1)[0].lower()
    if "scam" in head:
        return 1
    if "legit" in head:
        return 0
    return 1


@dataclass
class Example:
    prompt: str
    label_id: int  # 1 scam, 0 legit


class PromptDataset(Dataset):
    def __init__(self, raw: list[dict], max_prompt_chars: int, soft2_field: str | None):
        self.items: List[Example] = []
        self.soft2: List[List[float] | None] = []
        for it in raw:
            inp = it.get("input", "")
            out = it.get("output", "")
            label_id = extract_hard_label(out)
            # 训练与评估对齐：以 "# Pred:" 结尾
            prompt = inp[:max_prompt_chars].replace("# Pred:", "").strip()
            if "# Information:" not in prompt:
                prompt = f"# Information:\n{prompt}"
            prompt = f"{prompt}\n\n# Pred:\n"
            self.items.append(Example(prompt=prompt, label_id=label_id))
            if soft2_field is not None:
                val = it.get(soft2_field, None)
                if (
                    isinstance(val, list)
                    and len(val) == 2
                    and all(isinstance(x, (int, float)) for x in val)
                ):
                    self.soft2.append([float(val[0]), float(val[1])])
                else:
                    self.soft2.append(None)
            else:
                self.soft2.append(None)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        ex = self.items[idx]
        item = {"prompt": ex.prompt, "label_id": ex.label_id}
        if self.soft2[idx] is not None:
            item["teacher_soft2"] = self.soft2[idx]
        return item


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


def _seq_logprob_of_completion(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device,
) -> torch.Tensor:
    """
    返回 log P(completion | prompt) 的可微标量（对 student 反传；teacher 用 no_grad）。
    实现：拼接后前向一次，累加 completion token 的 logprob。
    """
    # 显式截断，避免 tokenizer 在超长输入时 warning/越界（尤其 tiny_test 的 GPT2 上限 1024）
    model_max = getattr(tokenizer, "model_max_length", None)
    if model_max is None or model_max > 100000:
        model_max = 4096

    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=model_max)["input_ids"]
    comp_ids = tokenizer(completion, add_special_tokens=False, truncation=True, max_length=64)["input_ids"]
    if len(comp_ids) == 0:
        return torch.tensor(float("-inf"), device=device)

    # 二次裁剪：确保拼接后不超过 model_max（保留 completion，裁 prompt）
    if len(prompt_ids) + len(comp_ids) > model_max:
        keep = max(model_max - len(comp_ids), 0)
        prompt_ids = prompt_ids[-keep:]

    input_ids = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids, device=device)
    # 训练时不要开 KV cache（否则显存暴涨）
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits  # [1, L, V]
    logp = torch.log_softmax(logits, dim=-1)

    prompt_len = len(prompt_ids)
    total = torch.zeros((), device=device)
    for i, tok_id in enumerate(comp_ids):
        pos = prompt_len + i
        if pos == 0:
            continue
        total = total + logp[0, pos - 1, tok_id]
    return total


def two_class_logits(model, tokenizer, prompt: str, device) -> torch.Tensor:
    """
    返回 shape=[2] 的 logits，顺序为 [legit, scam]。
    """
    s_legit = _seq_logprob_of_completion(model, tokenizer, prompt, "Label: legit", device)
    s_scam = _seq_logprob_of_completion(model, tokenizer, prompt, "Label: scam", device)
    return torch.stack([s_legit, s_scam], dim=0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("Step1 分类蒸馏（学生=Mistral 7B 4bit）")
    print("=" * 60)

    assert_exists(args.data_path, "训练数据集")
    use_precomputed = args.expect_soft2_field is not None
    if not args.tiny_test and not use_precomputed:
        assert_exists(args.teacher_base_model_path, "教师基座目录")
        assert_exists(args.teacher_lora_path, "教师 LoRA 目录")
        assert_exists(os.path.join(args.teacher_lora_path, "adapter_config.json"), "教师 adapter_config.json")

    raw = load_json_list(args.data_path)
    if args.shuffle_seed is not None:
        rng = random.Random(args.shuffle_seed)
        raw = raw.copy()
        rng.shuffle(raw)
    if args.max_samples is not None:
        raw = raw[: args.max_samples]
    print(f"训练样本数: {len(raw)}")

    ds = PromptDataset(raw, max_prompt_chars=args.max_prompt_chars, soft2_field=args.expect_soft2_field)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # load teacher
    teacher = None
    if use_precomputed:
        print(f"[INFO] 使用预计算 soft2 字段: {args.expect_soft2_field}，跳过加载 teacher。")
    elif args.tiny_test:
        teacher_name = "sshleifer/tiny-gpt2"
        t_tok = AutoTokenizer.from_pretrained(teacher_name, local_files_only=False)
        if t_tok.pad_token is None:
            t_tok.pad_token = t_tok.eos_token
        teacher = AutoModelForCausalLM.from_pretrained(teacher_name).to("cpu")
        teacher.eval()
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True)
        t_lfo = hf_local_files_only(args.teacher_base_model_path)
        t_tok = AutoTokenizer.from_pretrained(args.teacher_base_model_path, local_files_only=t_lfo)
        if t_tok.pad_token is None:
            t_tok.pad_token = t_tok.eos_token
        t_base = AutoModelForCausalLM.from_pretrained(
            args.teacher_base_model_path,
            quantization_config=bnb,
            device_map="auto",
            local_files_only=t_lfo,
        )
        teacher = PeftModel.from_pretrained(t_base, args.teacher_lora_path)
        teacher.eval()

    # load student
    if args.tiny_test:
        s_name = "sshleifer/tiny-gpt2"
        s_tok = AutoTokenizer.from_pretrained(s_name, local_files_only=False)
        if s_tok.pad_token is None:
            s_tok.pad_token = s_tok.eos_token
        s_base = AutoModelForCausalLM.from_pretrained(s_name).to("cpu")
        student = build_lora(s_base, r=args.lora_r, tiny_test=True)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True)
        s_lfo = hf_local_files_only(args.student_base)
        s_tok = AutoTokenizer.from_pretrained(args.student_base, local_files_only=s_lfo)
        if s_tok.pad_token is None:
            s_tok.pad_token = s_tok.eos_token
        s_base = AutoModelForCausalLM.from_pretrained(
            args.student_base,
            quantization_config=bnb,
            device_map="auto",
            local_files_only=s_lfo,
        )
        # 训练阶段关闭 cache，显著降低显存占用
        if hasattr(s_base, "config"):
            s_base.config.use_cache = False
        student = build_lora(s_base, r=args.lora_r, tiny_test=False)

    student.print_trainable_parameters()
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    total_steps = max((len(dl) * args.epochs) // max(args.grad_accum, 1), 1)
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    T = float(args.temperature)
    alpha = float(args.alpha)

    step = 0
    optimizer.zero_grad(set_to_none=True)

    for ep in range(args.epochs):
        running = 0.0
        pbar = tqdm(dl, desc=f"Epoch {ep + 1}/{args.epochs}")
        for batch in pbar:
            prompts: List[str] = batch["prompt"]
            labels: torch.Tensor = torch.tensor(batch["label_id"], dtype=torch.long, device=student.device)

            # 逐条算 2 类 logits（实现简单可靠；batch_size 建议 1）
            s_logits_list: List[torch.Tensor] = []
            t_probs_list: List[torch.Tensor] = []

            for prompt in prompts:
                # student 2-class logits (grad)
                s2 = two_class_logits(student, s_tok, prompt, student.device)  # [2]
                s_logits_list.append(s2)

                # teacher probs
                if not use_precomputed:
                    with torch.no_grad():
                        t2 = two_class_logits(teacher, t_tok, prompt, teacher.device)  # [2]
                        t_probs = F.softmax(t2 / T, dim=-1)
                    t_probs_list.append(t_probs.to(student.device))

            s2logits = torch.stack(s_logits_list, dim=0)  # [B,2] [legit,scam]
            if use_precomputed:
                if "teacher_soft2" not in batch:
                    raise ValueError(
                        f"数据中未发现字段 {args.expect_soft2_field}（或某些样本缺失）。"
                    )
                # batch["teacher_soft2"] 形如 [[pL,pS], ...]
                t_probs = torch.tensor(batch["teacher_soft2"], dtype=torch.float32, device=student.device)
            else:
                t_probs = torch.stack(t_probs_list, dim=0)    # [B,2]

            hard_ce = F.cross_entropy(s2logits, labels)
            s_log_probs = F.log_softmax(s2logits / T, dim=-1)
            soft_kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T**2)
            loss = alpha * hard_ce + (1.0 - alpha) * soft_kl

            (loss / max(args.grad_accum, 1)).backward()
            running += float(loss.item())

            if (step + 1) % max(args.grad_accum, 1) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "hard": f"{hard_ce.item():.4f}", "soft": f"{soft_kl.item():.4f}"})

        print(f"  Epoch {ep + 1} 平均 loss: {running / max(len(dl), 1):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    student.save_pretrained(args.output_dir)
    s_tok.save_pretrained(args.output_dir)
    print(f"已保存学生 LoRA 到: {args.output_dir}")


if __name__ == "__main__":
    main()

