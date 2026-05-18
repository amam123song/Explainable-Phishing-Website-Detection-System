import argparse
import inspect
import json
import re
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, List, Optional

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


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


def build_text(example: Dict, text_mode: str = "full_input") -> str:
    input_text = example.get("input", "")
    if text_mode == "url_only":
        m = re.search(r"## URL:\s*\n(.+?)(?:\n|$)", input_text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return input_text


def make_splits(
    examples: List[Dict], train_ratio: float = 0.9, text_mode: str = "full_input"
) -> DatasetDict:
    texts = [build_text(e, text_mode=text_mode) for e in examples]
    labels = [build_label(e) for e in examples]
    n = len(texts)
    n_train = int(n * train_ratio)
    indices = np.arange(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    def subset(idxs: np.ndarray) -> Dataset:
        return Dataset.from_dict(
            {
                "text": [texts[i] for i in idxs],
                "label": [int(labels[i]) for i in idxs],
            }
        )

    return DatasetDict(
        train=subset(train_idx),
        validation=subset(val_idx) if len(val_idx) > 0 else subset(train_idx),
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1_pr, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    f1_bin = f1_score(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1_bin if np.isfinite(f1_bin) else f1_pr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="在钓鱼检测数据上微调 DistilBERT（英文二分类）")
    parser.add_argument(
        "--model-name-or-dir",
        type=str,
        default="distilbert-base-uncased",
        help="预训练 DistilBERT 模型（如 distilbert-base-uncased）或本地目录",
    )
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
        help="最多使用多少条样本进行微调（默认全量）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "09_baselines/distilbert/distilbert_finetuned_phishing",
        help="微调后模型输出目录",
    )
    parser.add_argument("--num-epochs", type=float, default=3.0, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="训练与评估 batch size")
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="tokenize 最大长度（默认 256，更省显存/内存）",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="梯度累积步数（显存/内存不足时可调大）",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="在无 GPU/加速器环境下禁用 dataloader pin_memory（更稳）",
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="学习率")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="warmup 比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--text-mode",
        type=str,
        default="full_input",
        choices=["full_input", "url_only"],
        help="文本构建模式：full_input 使用完整 input，url_only 仅使用 URL",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备：auto/cpu/cuda（默认 auto）",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"载入训练数据: {args.train_data}")
    raw_data = load_json_data(args.train_data, args.max_samples)
    dataset_dict = make_splits(raw_data, text_mode=args.text_mode)

    print(f"载入模型与 tokenizer: {args.model_name_or_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_dir, num_labels=2)

    def tokenize_function(examples):
        enc = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=args.max_length,
        )
        enc["labels"] = examples["label"]
        return enc

    tokenized = dataset_dict.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    params = inspect.signature(TrainingArguments.__init__).parameters
    train_args_kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": 0.01,
        "logging_steps": 50,
        "save_total_limit": 2,
        "seed": args.seed,
        "report_to": [],
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
    }
    if "eval_strategy" in params:
        train_args_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        train_args_kwargs["evaluation_strategy"] = "epoch"

    if "save_strategy" in params:
        train_args_kwargs["save_strategy"] = "epoch"

    if args.no_pin_memory and "dataloader_pin_memory" in params:
        train_args_kwargs["dataloader_pin_memory"] = False

    if args.device == "cpu":
        # transformers 不同版本对“强制 CPU”的参数名不同，做兼容处理
        if "use_cpu" in params:
            train_args_kwargs["use_cpu"] = True
        elif "no_cuda" in params:
            train_args_kwargs["no_cuda"] = True
    elif args.device == "cuda":
        # 若用户显式选择 cuda，确保不被 no_cuda/use_cpu 覆盖
        if "use_cpu" in params:
            train_args_kwargs["use_cpu"] = False
        if "no_cuda" in params:
            train_args_kwargs["no_cuda"] = False

    train_args_kwargs = {k: v for k, v in train_args_kwargs.items() if k in params}
    training_args = TrainingArguments(**train_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("开始微调 DistilBERT 基线 ...")
    trainer.train()
    print("训练完成，保存最佳模型到:", args.output_dir)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()

