#!/usr/bin/env python3
"""离线生成教师二类 soft label（字段 teacher_soft2）。仅供学术研究，禁止用于非法用途。"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

warnings.filterwarnings("ignore")

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from config.paths import hf_local_files_only

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成 Teacher 二类 soft label")
    p.add_argument("--data_path", type=str, default="data/dataset_scamnet_5000.json")
    p.add_argument("--out_path", type=str, default="data/dataset_scamnet_5000_soft2_T4.json")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--shuffle_seed", type=int, default=None)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--max_prompt_chars", type=int, default=8000)

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
    return p.parse_args()


def assert_exists(path: str, desc: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} 不存在: {path}")


def load_json_list(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 list JSON，实际为 {type(data)}: {path}")
    return data


def format_prompt(inp: str, max_prompt_chars: int) -> str:
    clean = (inp or "")[:max_prompt_chars].replace("# Pred:", "").strip()
    if "# Information:" not in clean:
        clean = f"# Information:\n{clean}"
    return f"{clean}\n\n# Pred:\n"


def _seq_logprob(model, tok, prompt: str, completion: str, device) -> torch.Tensor:
    model_max = getattr(tok, "model_max_length", None)
    if model_max is None or model_max > 100000:
        model_max = 4096
    prompt_ids = tok(prompt, add_special_tokens=False, truncation=True, max_length=model_max)["input_ids"]
    comp_ids = tok(completion, add_special_tokens=False, truncation=True, max_length=64)["input_ids"]
    if len(comp_ids) == 0:
        return torch.tensor(float("-inf"), device=device)
    if len(prompt_ids) + len(comp_ids) > model_max:
        keep = max(model_max - len(comp_ids), 0)
        prompt_ids = prompt_ids[-keep:]
    input_ids = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids, device=device)
    out = model(input_ids=input_ids, attention_mask=attn)
    logp = torch.log_softmax(out.logits, dim=-1)
    prompt_len = len(prompt_ids)
    total = torch.zeros((), device=device)
    for i, tok_id in enumerate(comp_ids):
        pos = prompt_len + i
        if pos == 0:
            continue
        total = total + logp[0, pos - 1, tok_id]
    return total


def main() -> None:
    args = parse_args()
    assert_exists(args.data_path, "输入数据集")
    assert_exists(args.teacher_base_model_path, "教师基座目录")
    assert_exists(args.teacher_lora_path, "教师 LoRA 目录")
    assert_exists(os.path.join(args.teacher_lora_path, "adapter_config.json"), "教师 adapter_config.json")

    data = load_json_list(args.data_path)
    if args.shuffle_seed is not None:
        rng = random.Random(args.shuffle_seed)
        data = data.copy()
        rng.shuffle(data)
    if args.max_samples is not None:
        data = data[: args.max_samples]
    print(f"样本数: {len(data)}")

    bnb = BitsAndBytesConfig(load_in_4bit=True)
    t_lfo = hf_local_files_only(args.teacher_base_model_path)
    tok = AutoTokenizer.from_pretrained(args.teacher_base_model_path, local_files_only=t_lfo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.teacher_base_model_path,
        quantization_config=bnb,
        device_map="auto",
        local_files_only=t_lfo,
    )
    teacher = PeftModel.from_pretrained(base, args.teacher_lora_path)
    teacher.eval()

    T = float(args.temperature)

    out: List[dict] = []
    for item in tqdm(data, total=len(data)):
        prompt = format_prompt(item.get("input", ""), args.max_prompt_chars)
        with torch.no_grad():
            l_legit = _seq_logprob(teacher, tok, prompt, "Label: legit", teacher.device)
            l_scam = _seq_logprob(teacher, tok, prompt, "Label: scam", teacher.device)
            logits2 = torch.stack([l_legit, l_scam], dim=0)
            probs2 = F.softmax(logits2 / T, dim=-1).detach().float().cpu().tolist()  # [p_legit, p_scam]
        new_item = dict(item)
        new_item["teacher_soft2"] = probs2
        out.append(new_item)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已写出: {out_path}")


if __name__ == "__main__":
    main()

