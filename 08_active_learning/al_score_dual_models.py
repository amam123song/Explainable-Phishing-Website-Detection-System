#!/usr/bin/env python3
"""对抗池主动学习打分（不确定性 U，可选 hidden 特征供 K-means）。仅供学术防御研究，禁止用于非法用途。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# 与脚本同目录的 al_common
_AL_DIR = Path(__file__).resolve().parent
if str(_AL_DIR) not in sys.path:
    sys.path.insert(0, str(_AL_DIR))

import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer

from al_common import build_pred_prompt, label_from_output, load_causal_lm_4bit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _seq_logprob_sum(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
    model_max: int,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=model_max)["input_ids"]
    comp_ids = tokenizer(completion, add_special_tokens=False, truncation=True, max_length=64)["input_ids"]
    if not comp_ids:
        return float("-inf")
    if len(prompt_ids) + len(comp_ids) > model_max:
        keep = max(model_max - len(comp_ids), 0)
        prompt_ids = prompt_ids[-keep:]
    input_ids = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logp = out.logits.float().log_softmax(dim=-1)
    prompt_len = len(prompt_ids)
    total = 0.0
    for i, tok_id in enumerate(comp_ids):
        pos = prompt_len + i
        if pos == 0:
            continue
        total += float(logp[0, pos - 1, tok_id].item())
    return total


def _prompt_hidden_mean(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    enc = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        out = model(
            **enc,
            output_hidden_states=True,
            use_cache=False,
        )
        h = out.hidden_states[-1][0]
        m = enc.attention_mask[0].float().unsqueeze(-1)
        return (h * m).sum(dim=0) / m.sum().clamp(min=1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对抗池主动学习：分类学生不确定性打分")
    p.add_argument(
        "--pool-json",
        type=Path,
        default=REPO_ROOT / "data" / "dataset_adversarial_pool.json",
        help="对抗样本 JSON（list，含 input / output）",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "results" / "al_scores.json",
        help="输出打分结果 JSON",
    )
    p.add_argument(
        "--cls-base",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Mistral 基座（与 06_inference/run_benchmark.py --student-cls-base 一致）",
    )
    p.add_argument(
        "--cls-lora",
        type=str,
        default=str(REPO_ROOT / "outputs/decouple/cls_adapter"),
        help="分类 LoRA（与 dual_cls / dual_full 分类阶段一致）",
    )
    p.add_argument("--max-samples", type=int, default=None, help="仅处理前 N 条（调试用）")
    p.add_argument("--max-prompt-chars", type=int, default=8000, help="与评测类似的 input 截断字符数")
    p.add_argument(
        "--tokenizer-max-length",
        type=int,
        default=4096,
        help="tokenizer 截断上限（用于 logprob 与 hidden）",
    )
    p.add_argument(
        "--no-feat",
        action="store_true",
        help="不算 hidden 特征（更快；后续选样需用 --mode random 或仅用 U 排序）",
    )
    p.add_argument(
        "--alpha-h",
        type=float,
        default=0.4,
        help="U 中熵项权重（对二元熵已除以 log2）",
    )
    p.add_argument(
        "--alpha-m",
        type=float,
        default=0.3,
        help="U 中 (1-margin) 权重",
    )
    p.add_argument(
        "--alpha-lc",
        type=float,
        default=0.3,
        help="U 中 least-confidence 权重",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pool_json.is_file():
        print(f"[ERROR] 找不到对抗池: {args.pool_json}", file=sys.stderr)
        sys.exit(1)

    with open(args.pool_json, "r", encoding="utf-8") as f:
        pool = json.load(f)
    if not isinstance(pool, list):
        print("[ERROR] 对抗池 JSON 应为 list", file=sys.stderr)
        sys.exit(1)
    n_total = len(pool)
    if args.max_samples is not None:
        pool = pool[: args.max_samples]

    from config.paths import hf_local_files_only

    cls_lfo = hf_local_files_only(args.cls_base)
    tokenizer = AutoTokenizer.from_pretrained(args.cls_base, local_files_only=cls_lfo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = load_causal_lm_4bit(args.cls_base, local_files_only=cls_lfo)
    if hasattr(base, "config"):
        base.config.use_cache = False
    model = PeftModel.from_pretrained(base, args.cls_lora)
    model.eval()
    device = next(model.parameters()).device

    model_max = args.tokenizer_max_length
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max and tok_max < 100000:
        model_max = min(model_max, tok_max)

    records = []
    for i, item in enumerate(tqdm(pool, desc="AL-score")):
        inp = item.get("input", "")
        out = item.get("output", "")
        y_true = label_from_output(out)
        prompt = build_pred_prompt(inp, args.max_prompt_chars)

        s_legit = _seq_logprob_sum(model, tokenizer, prompt, "Label: legit", device, model_max)
        s_scam = _seq_logprob_sum(model, tokenizer, prompt, "Label: scam", device, model_max)
        logits = torch.tensor([s_legit, s_scam], dtype=torch.float32)
        probs = F.softmax(logits, dim=0)
        p_legit, p_scam = float(probs[0].item()), float(probs[1].item())
        H = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
        h_norm = H / math.log(2.0) if H > 0 else 0.0
        sorted_p = sorted([p_legit, p_scam], reverse=True)
        margin = sorted_p[0] - sorted_p[1]
        least_conf = 1.0 - max(p_legit, p_scam)
        U = args.alpha_h * h_norm + args.alpha_m * (1.0 - margin) + args.alpha_lc * least_conf

        rec = {
            "index": i,
            "y_true": y_true,
            "s_legit": s_legit,
            "s_scam": s_scam,
            "p_legit": round(p_legit, 6),
            "p_scam": round(p_scam, 6),
            "entropy": round(H, 6),
            "entropy_norm": round(h_norm, 6),
            "margin": round(margin, 6),
            "least_confidence": round(least_conf, 6),
            "U": round(U, 6),
        }
        if not args.no_feat:
            feat_t = _prompt_hidden_mean(model, tokenizer, prompt, device, model_max)
            rec["feat"] = [round(float(x), 6) for x in feat_t.cpu().tolist()]

        records.append(rec)

    payload = {
        "pool_json": str(args.pool_json.resolve()),
        "pool_size_total": n_total,
        "scored_count": len(records),
        "cls_base": args.cls_base,
        "cls_lora": args.cls_lora,
        "max_prompt_chars": args.max_prompt_chars,
        "tokenizer_max_length": args.tokenizer_max_length,
        "weights_U": {"H_norm": args.alpha_h, "one_minus_margin": args.alpha_m, "least_conf": args.alpha_lc},
        "note": "dual_full 与 dual_cls 共用同一分类 LoRA；主动学习不确定性以此为准。",
        "scores": records,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] 已写入 {args.out_json}（{len(records)} 条）")


if __name__ == "__main__":
    main()
