#!/usr/bin/env python3
"""
动态路由阈值实验脚本（双学生：分类 LoRA + 解释 LoRA）

功能：
1) 同时加载干净测试集与对抗测试集（可分别限制样本数）
2) 先对全部样本执行分类，得到：
   - 预测标签
   - 二分类置信度 p_max
   - 分类耗时
3) 对全部样本执行一次解释生成，得到：
   - 解释文本
   - 解释耗时
4) 对阈值列表逐一离线汇总动态路由指标，无需重复跑模型：
   - 解释触发率
   - 平均延迟（分类 + 触发解释）
   - 分类 Accuracy / F1 / Precision / Recall（阈值下不变，保留用于完整性）
   - 漏解释风险（错误样本中未触发解释比例）
   - 置信度与正确性相关性（Spearman）

说明：不做解释质量自动打分；样本默认固定取前 N 条。
Research use only — 仅供学术防御研究，禁止用于非法用途。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.paths import (  # noqa: E402
    DATASET_ADV_POOL,
    DATASET_TEST,
    MISTRAL_BASE,
    OUTPUT_DECOUPLE_CLS,
    OUTPUT_DECOUPLE_EXPLAIN,
    RESULTS_DIR,
    hf_local_files_only,
)


def _set_torch_alloc_env() -> None:
    import os

    val = "expandable_segments:True"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", val)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", val)


def _extract_url(input_text: str) -> str:
    m = re.search(r"## URL:\s*\n(.+?)(?:\n|$)", input_text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _jsonl_append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def _jsonl_load_map(path: Path, key: str = "index") -> Dict[int, dict]:
    if not path.exists():
        return {}
    out: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = int(rec[key])
            out[idx] = rec
    return out


def _compute_resume_key(args: argparse.Namespace, n_clean: int, n_adv: int) -> str:
    payload = {
        "clean_data_path": str(args.clean_data_path),
        "adv_data_path": str(args.adv_data_path),
        "max_samples_clean": n_clean,
        "max_samples_adv": n_adv,
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "student_cls_base": args.student_cls_base,
        "student_cls_lora": args.student_cls_lora,
        "student_explain_lora": args.student_explain_lora,
        "dual_full_max_new_tokens_explain": args.dual_full_max_new_tokens_explain,
        "dual_full_max_seq_len_explain": args.dual_full_max_seq_len_explain,
        "dual_full_light_explain": bool(args.dual_full_light_explain),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def load_json_dataset(
    path: Path,
    max_samples: Optional[int],
    source_name: str,
    sample_mode: str,
    rng: random.Random,
) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples is not None:
        k = max(0, max_samples)
        if sample_mode == "head":
            data = data[:k]
        elif sample_mode == "random":
            if k < len(data):
                idx = sorted(rng.sample(range(len(data)), k))
                data = [data[i] for i in idx]
            else:
                data = list(data)
        else:
            raise ValueError(f"不支持的 sample_mode: {sample_mode}")
    out: List[dict] = []
    for i, item in enumerate(data):
        inp = item.get("input", "")
        out_raw = (item.get("output") or "").strip().lower()
        y_true = 1 if "scam" in out_raw.split("\n")[0] else 0
        out.append(
            {
                "global_index": -1,  # 稍后统一赋值
                "source": source_name,
                "source_index": i,
                "input": inp,
                "url": _extract_url(inp),
                "y_true": y_true,
                "output_raw": item.get("output", ""),
            }
        )
    return out


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def spearman_confidence_correctness(confidences: List[float], correctness: List[int]) -> float:
    from scipy.stats import spearmanr

    if len(confidences) < 2:
        return float("nan")
    corr, _ = spearmanr(confidences, correctness)
    if corr is None or math.isnan(corr):
        return float("nan")
    return float(corr)


def _from_pretrained_4bit_causal_lm(model_id: str, *, local_files_only: bool = True, fallback_label: str = "模型"):
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

    config = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
    load_kwargs = {
        "device_map": "auto",
        "local_files_only": local_files_only,
        "attn_implementation": "sdpa",
    }
    if getattr(config, "quantization_config", None) is None:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as e:
        print(f"[警告] {fallback_label} sdpa 加载失败 ({e})，回退 eager")
        load_kwargs["attn_implementation"] = "eager"
        return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)


@dataclass
class ClsRecord:
    y_pred: int
    score_scam: float
    score_legit: float
    p_scam: float
    p_legit: float
    p_max: float
    cls_latency_s: float


def run_classification_all(
    data: List[dict],
    cls_base: str,
    cls_lora: str,
    resume_cls_path: Optional[Path] = None,
) -> List[ClsRecord]:
    import torch
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoTokenizer

    lfo = hf_local_files_only(cls_base)
    tokenizer = AutoTokenizer.from_pretrained(cls_base, local_files_only=lfo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = _from_pretrained_4bit_causal_lm(cls_base, local_files_only=lfo, fallback_label="分类基座")
    model = PeftModel.from_pretrained(base, cls_lora)
    model.eval()
    device = next(model.parameters()).device

    def score_candidate(prompt: str, completion: str) -> float:
        with torch.no_grad():
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if not comp_ids:
                return float("-inf")
            input_ids = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device=device)
            attn = torch.ones_like(input_ids, device=device)
            out = model(input_ids=input_ids, attention_mask=attn)
            logp = out.logits.log_softmax(dim=-1)
            prompt_len = len(prompt_ids)
            total = 0.0
            for i, tok_id in enumerate(comp_ids):
                pos = prompt_len + i
                if pos == 0:
                    continue
                total += float(logp[0, pos - 1, tok_id].item())
            return total

    n = len(data)
    done_map = _jsonl_load_map(resume_cls_path) if resume_cls_path is not None else {}
    out_records: List[Optional[ClsRecord]] = [None for _ in range(n)]
    for idx, rec in done_map.items():
        if 0 <= idx < n:
            out_records[idx] = ClsRecord(
                y_pred=int(rec["y_pred"]),
                score_scam=float(rec["score_scam"]),
                score_legit=float(rec["score_legit"]),
                p_scam=float(rec["p_scam"]),
                p_legit=float(rec["p_legit"]),
                p_max=float(rec["p_max"]),
                cls_latency_s=float(rec["cls_latency_s"]),
            )

    for i, item in enumerate(tqdm(data, desc="Classification")):
        if out_records[i] is not None:
            continue
        prompt = f"{item['input']}\n# Pred:\n"
        t0 = time.perf_counter()
        s_scam = score_candidate(prompt, "Label: scam")
        s_legit = score_candidate(prompt, "Label: legit")
        t1 = time.perf_counter()

        # 对两个候选分数做 softmax，得到置信度
        m = max(s_scam, s_legit)
        exp_scam = math.exp(s_scam - m)
        exp_legit = math.exp(s_legit - m)
        denom = exp_scam + exp_legit + 1e-12
        p_scam = exp_scam / denom
        p_legit = exp_legit / denom

        y_pred = 1 if s_scam >= s_legit else 0
        p_max = max(p_scam, p_legit)
        one = ClsRecord(
            y_pred=y_pred,
            score_scam=s_scam,
            score_legit=s_legit,
            p_scam=p_scam,
            p_legit=p_legit,
            p_max=p_max,
            cls_latency_s=(t1 - t0),
        )
        out_records[i] = one
        if resume_cls_path is not None:
            _jsonl_append(
                resume_cls_path,
                {
                    "index": i,
                    "y_pred": one.y_pred,
                    "score_scam": one.score_scam,
                    "score_legit": one.score_legit,
                    "p_scam": one.p_scam,
                    "p_legit": one.p_legit,
                    "p_max": one.p_max,
                    "cls_latency_s": one.cls_latency_s,
                },
            )

    del model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    missing = [i for i, x in enumerate(out_records) if x is None]
    if missing:
        raise RuntimeError(f"[resume] 分类阶段缺失记录: {len(missing)}, first={missing[:5]}")
    return [x for x in out_records if x is not None]


def run_explanations_all(
    data: List[dict],
    cls_preds: List[int],
    cls_base: str,
    explain_lora: str,
    max_new_tokens_explain: int,
    max_seq_len_explain: int,
    light_explain: bool,
    resume_exp_path: Optional[Path] = None,
) -> Tuple[List[str], List[float]]:
    import torch
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoTokenizer

    lfo = hf_local_files_only(cls_base)
    tokenizer = AutoTokenizer.from_pretrained(cls_base, local_files_only=lfo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = _from_pretrained_4bit_causal_lm(cls_base, local_files_only=lfo, fallback_label="解释基座")
    if torch.cuda.is_available() and hasattr(base, "config"):
        base.config.use_cache = False
    model = PeftModel.from_pretrained(base, explain_lora)
    model.eval()
    device = next(model.parameters()).device

    def format_prompt_explain(inp: str) -> str:
        clean = inp.replace("# Pred:", "").strip()
        if "# Information:" not in clean:
            clean = f"# Information:\n{clean}"
        return f"{clean}\n\n# Pred:\n"

    n = len(data)
    done_map = _jsonl_load_map(resume_exp_path) if resume_exp_path is not None else {}
    texts: List[str] = ["" for _ in range(n)]
    lats: List[float] = [-1.0 for _ in range(n)]
    for idx, rec in done_map.items():
        if 0 <= idx < n:
            texts[idx] = str(rec["generated_explanation"])
            lats[idx] = float(rec["exp_latency_s"])

    for i, item in enumerate(tqdm(data, desc="Explanation(all cached)")):
        if texts[i] and lats[i] >= 0:
            continue
        label_str = "scam" if cls_preds[i] == 1 else "legit"
        base_prompt = format_prompt_explain(item["input"])
        if light_explain:
            hint = (
                "Write a concise reason only. Use 1-2 short sentences, focus on strongest phishing cues, "
                "avoid background and repetition, and keep total explanation under 160 tokens."
            )
            prompt_exp = f"{base_prompt}Label: {label_str}\n\n## Reason:\n{hint}\n"
        else:
            prompt_exp = f"{base_prompt}Label: {label_str}\n\n## Reason:\n"

        inputs = tokenizer(
            prompt_exp,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len_explain,
        ).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens_explain,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
        t1 = time.perf_counter()
        gen_explain = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        full_response = f"Label: {label_str}\n\n## Reason:\n{gen_explain}"

        one_lat = t1 - t0
        texts[i] = full_response
        lats[i] = one_lat
        if resume_exp_path is not None:
            _jsonl_append(
                resume_exp_path,
                {
                    "index": i,
                    "generated_explanation": full_response,
                    "exp_latency_s": one_lat,
                },
            )

    del model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    missing = [i for i, (t, l) in enumerate(zip(texts, lats)) if (not t) or l < 0]
    if missing:
        raise RuntimeError(f"[resume] 解释阶段缺失记录: {len(missing)}, first={missing[:5]}")
    return texts, lats


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = (len(values_sorted) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return values_sorted[lo]
    frac = idx - lo
    return values_sorted[lo] * (1 - frac) + values_sorted[hi] * frac


def parse_thresholds(raw: str) -> List[float]:
    vals: List[float] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        if v <= 0 or v >= 1:
            raise ValueError(f"阈值需在 (0,1) 内，收到: {v}")
        vals.append(v)
    if not vals:
        raise ValueError("至少提供一个阈值")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="动态路由阈值扫描实验脚本")
    parser.add_argument(
        "--clean-data-path",
        type=Path,
        default=DATASET_TEST,
        help="干净测试集路径（默认 data/dataset_test_strict.json）",
    )
    parser.add_argument(
        "--adv-data-path",
        type=Path,
        default=DATASET_ADV_POOL,
        help="对抗测试集路径（默认 data/dataset_adversarial_pool.json，需自行生成）",
    )
    parser.add_argument("--max-samples-clean", type=int, default=None, help="干净集最多测试样本数（默认全量）")
    parser.add_argument("--max-samples-adv", type=int, default=None, help="对抗集最多测试样本数（默认全量）")
    parser.add_argument(
        "--max-samples-each",
        type=int,
        default=None,
        help="干净/对抗统一样本上限；若设置，则覆盖 --max-samples-clean 与 --max-samples-adv",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["head", "random"],
        default="head",
        help="样本截取模式：head=固定前N条，random=随机抽样N条",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（仅 sample-mode=random 时生效）",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.99",
        help="阈值列表，逗号分隔",
    )
    parser.add_argument(
        "--student-cls-base",
        type=str,
        default=MISTRAL_BASE,
        help="分类/解释共同基座（Hub ID 或本地目录）",
    )
    parser.add_argument(
        "--student-cls-lora",
        type=str,
        default=str(OUTPUT_DECOUPLE_CLS),
        help="分类 LoRA 路径",
    )
    parser.add_argument(
        "--student-explain-lora",
        type=str,
        default=str(OUTPUT_DECOUPLE_EXPLAIN),
        help="解释 LoRA 路径",
    )
    parser.add_argument("--dual-full-max-new-tokens-explain", type=int, default=224)
    parser.add_argument("--dual-full-max-seq-len-explain", type=int, default=4096)
    parser.add_argument(
        "--dual-full-light-explain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否使用轻量解释模式",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "dynamic_threshold",
        help="结果输出目录（默认 results/dynamic_threshold/）",
    )
    parser.add_argument(
        "--resume-progress-dir",
        type=Path,
        default=None,
        help="断点续传目录（jsonl）。设置后将按样本写入分类/解释进度，重启自动跳过已完成样本。",
    )
    parser.add_argument(
        "--resume-tag",
        type=str,
        default=None,
        help="可选：自定义断点续传标识。默认根据运行参数自动生成。",
    )
    args = parser.parse_args()

    _set_torch_alloc_env()
    thresholds = parse_thresholds(args.thresholds)
    rng = random.Random(args.seed)

    if args.max_samples_each is not None:
        max_clean = args.max_samples_each
        max_adv = args.max_samples_each
    else:
        max_clean = args.max_samples_clean
        max_adv = args.max_samples_adv

    if not args.clean_data_path.exists():
        raise FileNotFoundError(f"干净测试集不存在: {args.clean_data_path}")
    if not args.adv_data_path.exists():
        raise FileNotFoundError(f"对抗测试集不存在: {args.adv_data_path}")
    if not Path(args.student_cls_lora).exists():
        raise FileNotFoundError(f"分类 LoRA 不存在: {args.student_cls_lora}")
    if not Path(args.student_explain_lora).exists():
        raise FileNotFoundError(f"解释 LoRA 不存在: {args.student_explain_lora}")

    clean_data = load_json_dataset(args.clean_data_path, max_clean, "clean", args.sample_mode, rng)
    adv_data = load_json_dataset(args.adv_data_path, max_adv, "adversarial", args.sample_mode, rng)
    data = clean_data + adv_data
    for i, item in enumerate(data):
        item["global_index"] = i

    if not data:
        raise RuntimeError("没有可测试样本，请检查样本数参数")

    n_clean = len(clean_data)
    n_adv = len(adv_data)
    n_total = len(data)
    print(f"[Data] clean={n_clean}, adversarial={n_adv}, total={n_total}")

    resume_cls_path = None
    resume_exp_path = None
    if args.resume_progress_dir is not None:
        args.resume_progress_dir.mkdir(parents=True, exist_ok=True)
        resume_key = args.resume_tag or _compute_resume_key(args, n_clean=n_clean, n_adv=n_adv)
        resume_cls_path = args.resume_progress_dir / f"{resume_key}_cls.jsonl"
        resume_exp_path = args.resume_progress_dir / f"{resume_key}_exp.jsonl"
        print(f"[Resume] key={resume_key}")
        print(f"[Resume] cls={resume_cls_path}")
        print(f"[Resume] exp={resume_exp_path}")

    print("[Step1] 运行全量分类（一次）...")
    cls_records = run_classification_all(
        data,
        args.student_cls_base,
        args.student_cls_lora,
        resume_cls_path=resume_cls_path,
    )
    y_true = [x["y_true"] for x in data]
    y_pred = [r.y_pred for r in cls_records]
    confs = [r.p_max for r in cls_records]
    correctness = [1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]
    cls_lats = [r.cls_latency_s for r in cls_records]

    print("[Step2] 运行全量解释缓存（一次）...")
    exp_texts, exp_lats = run_explanations_all(
        data=data,
        cls_preds=y_pred,
        cls_base=args.student_cls_base,
        explain_lora=args.student_explain_lora,
        max_new_tokens_explain=args.dual_full_max_new_tokens_explain,
        max_seq_len_explain=args.dual_full_max_seq_len_explain,
        light_explain=bool(args.dual_full_light_explain),
        resume_exp_path=resume_exp_path,
    )

    base_metrics = compute_binary_metrics(y_true, y_pred)
    sp = spearman_confidence_correctness(confs, correctness)

    # 按阈值离线汇总
    threshold_rows: List[dict] = []
    for delta in thresholds:
        trigger_mask = [c < delta for c in confs]
        triggered_idx = [i for i, m in enumerate(trigger_mask) if m]
        not_triggered_idx = [i for i, m in enumerate(trigger_mask) if not m]

        coverage = len(triggered_idx) / n_total
        total_lat = 0.0
        lat_per_sample = []
        for i in range(n_total):
            one = cls_lats[i] + (exp_lats[i] if trigger_mask[i] else 0.0)
            total_lat += one
            lat_per_sample.append(one)
        mean_lat = total_lat / n_total
        p50 = percentile(lat_per_sample, 0.5)
        p95 = percentile(lat_per_sample, 0.95)

        wrong_idx = [i for i in range(n_total) if y_true[i] != y_pred[i]]
        miss_risk = 0.0
        if wrong_idx:
            missed_wrong = sum(1 for i in wrong_idx if i in not_triggered_idx)
            miss_risk = missed_wrong / len(wrong_idx)

        row = {
            "delta": delta,
            "n_total": n_total,
            "triggered_explanations": len(triggered_idx),
            "coverage": coverage,
            "latency_mean_s": mean_lat,
            "latency_p50_s": p50,
            "latency_p95_s": p95,
            "accuracy": base_metrics["accuracy"],
            "precision": base_metrics["precision"],
            "recall": base_metrics["recall"],
            "f1": base_metrics["f1"],
            "miss_risk": miss_risk,
            "spearman_confidence_correctness": sp,
            "avg_confidence_triggered": (
                statistics.mean([confs[i] for i in triggered_idx]) if triggered_idx else float("nan")
            ),
            "avg_confidence_not_triggered": (
                statistics.mean([confs[i] for i in not_triggered_idx]) if not_triggered_idx else float("nan")
            ),
        }
        threshold_rows.append(row)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"dynamic_threshold_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) 阈值汇总 CSV
    csv_path = run_dir / "threshold_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_rows)

    # 2) 全量样本明细 JSONL（含分类置信度、分类耗时、解释耗时、解释文本）
    sample_jsonl = run_dir / "sample_level_records.jsonl"
    with open(sample_jsonl, "w", encoding="utf-8") as f:
        for i, item in enumerate(data):
            rec = {
                "index": i,
                "source": item["source"],
                "source_index": item["source_index"],
                "url": item["url"],
                "y_true": item["y_true"],
                "y_pred": y_pred[i],
                "correct": correctness[i],
                "confidence_pmax": confs[i],
                "p_scam": cls_records[i].p_scam,
                "p_legit": cls_records[i].p_legit,
                "score_scam": cls_records[i].score_scam,
                "score_legit": cls_records[i].score_legit,
                "cls_latency_s": cls_lats[i],
                "exp_latency_s": exp_lats[i],
                "generated_explanation": exp_texts[i],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3) 运行配置与总览 JSON
    summary = {
        "run_params": {
            "clean_data_path": str(args.clean_data_path),
            "adv_data_path": str(args.adv_data_path),
            "max_samples_clean": max_clean,
            "max_samples_adv": max_adv,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "thresholds": thresholds,
            "student_cls_base": args.student_cls_base,
            "student_cls_lora": args.student_cls_lora,
            "student_explain_lora": args.student_explain_lora,
            "dual_full_max_new_tokens_explain": args.dual_full_max_new_tokens_explain,
            "dual_full_max_seq_len_explain": args.dual_full_max_seq_len_explain,
            "dual_full_light_explain": bool(args.dual_full_light_explain),
            "resume_progress_dir": str(args.resume_progress_dir) if args.resume_progress_dir else None,
            "resume_tag": args.resume_tag,
        },
        "n_clean": n_clean,
        "n_adv": n_adv,
        "n_total": n_total,
        "classification_metrics": base_metrics,
        "spearman_confidence_correctness": sp,
        "outputs": {
            "threshold_metrics_csv": str(csv_path),
            "sample_level_jsonl": str(sample_jsonl),
        },
    }
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Done] 输出目录: {run_dir}")
    print(f"  - 阈值汇总: {csv_path}")
    print(f"  - 样本明细: {sample_jsonl}")
    print(f"  - 运行总览: {summary_path}")


if __name__ == "__main__":
    main()

