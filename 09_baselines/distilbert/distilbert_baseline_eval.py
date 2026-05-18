import argparse
import json
import re
import time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_test_data(path: Path, max_samples: int | None = None) -> Tuple[List[dict], np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[:max_samples]
    y_true: List[int] = []
    for item in data:
        out = (item.get("output") or "").strip().lower()
        first_line = out.split("\n")[0]
        y_true.append(1 if "scam" in first_line else 0)
    return data, np.asarray(y_true, dtype=np.int64)


def build_text_batch(examples: List[dict]) -> List[str]:
    return [item.get("input", "") for item in examples]


def _extract_url(input_text: str) -> str:
    m = re.search(r"## URL:\s*\n(.+?)(?:\n|$)", input_text, re.DOTALL)
    return m.group(1).strip() if m else ""


def build_bad_cases(
    data: List[dict],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> List[dict]:
    y_pred = (y_prob >= threshold).astype(int)
    bad: List[dict] = []
    for i, (t, p) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
        if int(t) != int(p):
            inp = data[i].get("input", "")
            bad.append(
                {
                    "index": int(i),
                    "url": _extract_url(inp),
                    "true_label": "scam" if int(t) == 1 else "legit",
                    "pred_label": "scam" if int(p) == 1 else "legit",
                    "p_scam": float(y_prob[i]),
                }
            )
    return bad


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")
    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "roc_auc": float(roc) if not np.isnan(roc) else None,
        "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else None,
    }


def evaluate_distilbert(
    model_dir: Path,
    data_path: Path,
    max_samples: int | None = None,
    batch_size: int = 16,
    device: str | None = None,
    max_length: int = 512,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data, y_true = load_test_data(data_path, max_samples)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), num_labels=2)
    model.to(device)
    model.eval()

    texts = build_text_batch(data)
    all_probs: List[float] = []

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    t0 = time.perf_counter()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            all_probs.extend(probs.detach().cpu().tolist())

    t1 = time.perf_counter()
    time_per_sample = (t1 - t0) / max(1, len(texts))

    if device == "cuda":
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / 1024**3
    else:
        max_mem_gb = 0.0

    y_prob = np.asarray(all_probs, dtype=np.float32)
    metrics = compute_metrics(y_true, y_prob)
    bad_cases = build_bad_cases(data, y_true, y_prob, threshold=0.5)
    metrics.update(
        {
            "time_per_sample_s": round(float(time_per_sample), 6),
            "max_memory_gb": round(float(max_mem_gb), 4),
            "num_samples": int(len(texts)),
            "device": device,
            "model_dir": str(model_dir),
            "data_path": str(data_path),
            "num_bad_cases": int(len(bad_cases)),
            "bad_cases": bad_cases,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="DistilBERT（英文二分类）钓鱼检测基线评测脚本")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "09_baselines/distilbert/distilbert_finetuned_phishing",
        help="本地 DistilBERT 二分类模型目录（微调后输出目录）",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data/dataset_test_strict_10000.json",
        help="测试集 JSON 路径",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="最大评测样本数（默认全量）")
    parser.add_argument("--batch-size", type=int, default=16, help="推理 batch size")
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="tokenize 最大长度（默认 512）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device, 例如 cuda 或 cpu（默认自动检测）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="若提供则将指标写入该 JSON 文件",
    )
    args = parser.parse_args()

    metrics = evaluate_distilbert(
        model_dir=args.model_dir,
        data_path=args.data_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
    )

    print("DistilBERT 评测结果：")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"结果已写入: {args.output}")


if __name__ == "__main__":
    main()

