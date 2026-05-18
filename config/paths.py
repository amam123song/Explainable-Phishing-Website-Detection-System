"""路径与环境变量（使用相对路径或环境变量，勿硬编码 API Key 或个人目录）。"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── 数据 ──────────────────────────────────────────────────────────────
DATA_DIR = REPO_ROOT / "data"
DATASET_TRAIN = DATA_DIR / "dataset_scamnet_5000.json"
DATASET_EXPLAIN_200 = DATA_DIR / "dataset_explainable_200.json"
DATASET_TEST = DATA_DIR / "dataset_test_strict.json"
DATASET_TEST_LARGE = DATA_DIR / "dataset_test_strict_10000.json"
DATASET_ADV_POOL = DATA_DIR / "dataset_adversarial_pool.json"
DATASET_SOFT2 = DATA_DIR / "dataset_scamnet_5000_soft2_T4.json"

# ── 训练产出（本仓库不提供预训练权重，见 05_weights/README.md）────────
OUTPUT_DIR = REPO_ROOT / "outputs"
RESULTS_DIR = REPO_ROOT / "results"
EVAL_EXPLANATION_OUT = RESULTS_DIR / "explanation_compare_out"

OUTPUT_EXPERT_STEP1 = OUTPUT_DIR / "expert" / "scamnet_step1_model"
OUTPUT_EXPERT_FINAL = OUTPUT_DIR / "expert" / "scamnet_final_model"
OUTPUT_MONOLITHIC_CLS = OUTPUT_DIR / "monolithic" / "step1_cls"
OUTPUT_MONOLITHIC_EXPLAIN = OUTPUT_DIR / "monolithic" / "step2_explain"
OUTPUT_DECOUPLE_CLS = OUTPUT_DIR / "decouple" / "cls_adapter"
OUTPUT_DECOUPLE_EXPLAIN = OUTPUT_DIR / "decouple" / "explain_adapter"

# 推理脚本默认 LoRA 路径（需先完成训练）
CLS_ADAPTER = OUTPUT_DECOUPLE_CLS
EXPLAIN_ADAPTER = OUTPUT_DECOUPLE_EXPLAIN

# ── 基线模型产出（09_baselines 下相对路径）────────────────────────────
BASELINES_DIR = REPO_ROOT / "09_baselines"
RF_MODEL = BASELINES_DIR / "random_forest" / "rf_tfidf.joblib"
URLNET_CKPT = BASELINES_DIR / "urlnet" / "urlnet_charcnn_best.pth"
DISTILBERT_MODEL = BASELINES_DIR / "distilbert" / "distilbert_finetuned_phishing"
BERT_MODEL = BASELINES_DIR / "bert" / "bert_finetuned_phishing"

# ── HuggingFace 模型 ID（优先用 Hub ID，避免泄露本机 cache 路径）────────
MISTRAL_BASE = os.environ.get("MISTRAL_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
LLAMA_EXPERT_BASE = os.environ.get("LLAMA_EXPERT_BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
LLAMA_EXPERT_LORA = Path(os.environ.get("LLAMA_EXPERT_LORA_PATH", str(OUTPUT_EXPERT_FINAL)))
BERT_BASE = os.environ.get("BERT_BASE_MODEL", "bert-base-uncased")
DISTILBERT_BASE = os.environ.get("DISTILBERT_BASE_MODEL", "distilbert-base-uncased")

# ── API（仅环境变量，无默认值密钥）────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
)
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
JUDGE_BASE_URL = os.environ.get(
    "JUDGE_BASE_URL",
    os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
EXPLAIN_GEN_MODEL = os.environ.get("EXPLAIN_GEN_MODEL", "gpt-4o-mini")


def hf_local_files_only(model_id_or_path: str) -> bool:
    """本机目录/文件存在则仅离线加载；否则允许从 HuggingFace Hub 下载。"""
    p = Path(model_id_or_path).expanduser()
    return p.exists() and (p.is_dir() or p.is_file())
