#!/usr/bin/env python3
"""随机森林基线评测。仅供学术防御研究，禁止用于非法用途。"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_test_data(path: Path, max_samples: Optional[int] = None) -> Tuple[List[Dict], np.ndarray]:
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


def build_text_batch(examples: List[Dict]) -> List[str]:
    return [e.get("input", "") for e in examples]


def _extract_url(input_text: str) -> str:
    m = re.search(r"## URL:\s*\n(.+?)(?:\n|$)", input_text, re.DOTALL)
    return m.group(1).strip() if m else ""


def build_bad_cases(
    data: List[Dict],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> List[Dict]:
    y_pred = (y_prob >= threshold).astype(int)
    bad: List[Dict] = []
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
        "roc_auc": float(roc) if np.isfinite(roc) else None,
        "pr_auc": float(pr_auc) if np.isfinite(pr_auc) else None,
    }


def evaluate_rf(
    model_path: Path,
    data_path: Path,
    max_samples: Optional[int] = None,
    threshold: float = 0.5,
) -> dict:
    data, y_true = load_test_data(data_path, max_samples)
    texts = build_text_batch(data)

    model = joblib.load(model_path)

    t0 = time.perf_counter()
    y_prob = model.predict_proba(texts)[:, 1]
    t1 = time.perf_counter()

    time_per_sample = (t1 - t0) / max(1, len(texts))
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    bad_cases = build_bad_cases(data, y_true, y_prob, threshold=threshold)
    metrics.update(
        {
            "time_per_sample_s": round(float(time_per_sample), 6),
            "max_memory_gb": 0.0,
            "num_samples": int(len(texts)),
            "device": "cpu",
            "model_path": str(model_path),
            "data_path": str(data_path),
            "threshold": float(threshold),
            "num_bad_cases": int(len(bad_cases)),
            "bad_cases": bad_cases,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest + TF-IDF 钓鱼检测基线评测脚本")
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "09_baselines/random_forest/rf_tfidf.joblib",
        help="训练好的 RF 模型(joblib)路径",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data/dataset_test_strict_10000.json",
        help="测试集 JSON 路径",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大评测样本数（默认全量）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="阈值（prob>=threshold 判为 scam）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="若提供则将指标写入该 JSON 文件",
    )

    args = parser.parse_args()
    metrics = evaluate_rf(
        model_path=args.model,
        data_path=args.data_path,
        max_samples=args.max_samples,
        threshold=args.threshold,
    )

    print("RandomForest 评测结果：")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"结果已写入: {args.output}")


if __name__ == "__main__":
    main()

