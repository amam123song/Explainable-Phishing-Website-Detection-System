import argparse
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
from typing import List, Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def load_json_data(path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[:max_samples]
    return data


def build_label(example: Dict) -> int:
    out = (example.get("output") or "").strip().lower()
    first = out.split("\n")[0]
    return 1 if "scam" in first else 0


def build_text(example: Dict) -> str:
    # 传统基线：直接用 input 文本做 TF-IDF（包含 URL 与上下文信息）
    return example.get("input", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest + TF-IDF 钓鱼检测基线训练脚本")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=REPO_ROOT / "data/dataset_scamnet_5000.json",
        help="训练用 JSON 数据路径",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多使用多少条样本训练（默认全量）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "09_baselines/random_forest/rf_tfidf.joblib",
        help="输出模型（joblib）路径",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="RF 树数量",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="RF 最大深度（默认不限制）",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="RF 叶子最小样本数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    args = parser.parse_args()

    raw = load_json_data(args.train_data, args.max_samples)
    texts = [build_text(e) for e in raw]
    labels = np.asarray([build_label(e) for e in raw], dtype=np.int64)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=200_000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            ("rf", clf),
        ]
    )

    print(f"载入训练数据: {args.train_data}  样本数: {len(texts)}")
    print("开始训练 Random Forest + TF-IDF ...")
    pipe.fit(texts, labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.output)
    print(f"训练完成，模型已保存: {args.output}")


if __name__ == "__main__":
    main()

