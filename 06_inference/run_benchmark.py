#!/usr/bin/env python3
"""
多模型钓鱼网站检测统一评测。

--run：expert | dual_cls | dual_full | student_single | all

需自备测试集与 LoRA 权重，见 05_weights/README.md。
Research use only — 仅供学术防御研究，禁止用于非法用途。
"""

from __future__ import annotations

import argparse
import json
import hashlib
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from config.paths import hf_local_files_only

# 评估结果与错误样本保存目录
OUTPUT_DIR = REPO_ROOT / "results"
DEFAULT_CLS_LORA = REPO_ROOT / "outputs/decouple/cls_adapter"
DEFAULT_EXPLAIN_LORA = REPO_ROOT / "outputs/decouple/explain_adapter"


def _set_torch_alloc_env() -> None:
    """新版 PyTorch 推荐 PYTORCH_ALLOC_CONF；同时保留 PYTORCH_CUDA_ALLOC_CONF 以兼容旧版。"""
    import os

    val = "expandable_segments:True"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", val)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", val)


def _from_pretrained_4bit_causal_lm(
    model_id: str,
    *,
    local_files_only: bool | None = None,
    attn_implementation: str = "sdpa",
    fallback_label: str = "模型",
):
    if local_files_only is None:
        local_files_only = hf_local_files_only(model_id)
    """
    统一 4bit 因果模型加载：若 checkpoint 的 config 已含 quantization_config，则不再传入 BnB，
    避免 transformers 的重复 quantization_config 警告。
    """
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

    config = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
    load_kwargs: dict = {
        "device_map": "auto",
        "local_files_only": local_files_only,
        "attn_implementation": attn_implementation,
    }
    if getattr(config, "quantization_config", None) is None:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as e:
        if attn_implementation == "sdpa":
            print(f"  [警告] {fallback_label} sdpa 加载失败 ({e})，回退 eager")
            load_kwargs["attn_implementation"] = "eager"
            return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        raise e


# 公共：测试集加载与指标
def load_test_data(data_path: Path, max_samples: Optional[int]) -> Tuple[List[dict], List[int]]:
    """加载测试 JSON，返回 [{"input","output"}, ...] 与 y_true (0/1)。
    若指定 max_samples：固定取前 max_samples 条，不随机抽取。"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[:max_samples]  # 固定前 N 条，不随机
    y_true = []
    for item in data:
        out = (item.get("output") or "").strip().lower()
        if "scam" in out.split("\n")[0]:
            y_true.append(1)
        else:
            y_true.append(0)
    return data, y_true


def compute_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        accuracy_score,
        confusion_matrix,
    )
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def _extract_url(input_text: str) -> str:
    """从 input 中解析 URL（用于错误样本记录）。"""
    m = re.search(r"## URL:\s*\n(.+?)(?:\n|$)", input_text, re.DOTALL)
    return m.group(1).strip() if m else ""


def build_bad_cases(
    data: List[dict],
    y_true: List[int],
    y_pred: List[int],
) -> List[dict]:
    """根据 y_true / y_pred 收集预测错误的样本，便于写入文件。"""
    bad = []
    for i, (true_l, pred_l) in enumerate(zip(y_true, y_pred)):
        if true_l != pred_l:
            inp = data[i].get("input", "")
            bad.append({
                "index": i,
                "url": _extract_url(inp),
                "true_label": "scam" if true_l == 1 else "legit",
                "pred_label": "scam" if pred_l == 1 else "legit",
            })
    return bad


def _run_params_to_basename(args) -> str:
    """由运行参数生成文件名前缀（不含编号、不含扩展名）。"""
    run = getattr(args, "run", "all")
    max_s = getattr(args, "max_samples", None)
    max_str = str(max_s) if max_s is not None else "full"
    data_stem = args.data_path.stem
    suffix = ""
    # 避免 dual_full 轻量解释模式与普通模式结果文件同名混淆
    if run in ("dual_full", "all") and getattr(args, "dual_full_light_explain", False):
        suffix = "_dualfull_light"
    return f"run_{run}_samples_{max_str}_data_{data_stem}{suffix}"


def _next_available_path(dir_path: Path, base_name: str, ext: str = ".json") -> Path:
    """若 base_name+ext 已存在，则返回 base_name_1+ext, base_name_2+ext, ... 直到不重复。"""
    dir_path.mkdir(parents=True, exist_ok=True)
    candidate = dir_path / f"{base_name}{ext}"
    if not candidate.exists():
        return candidate
    k = 1
    while True:
        candidate = dir_path / f"{base_name}_{k}{ext}"
        if not candidate.exists():
            return candidate
        k += 1


def _jsonl_append(path: Path, obj: dict) -> None:
    """Append one JSON object as one line; flush for crash-safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def _jsonl_load_index_map(path: Path, *, index_key: str = "index") -> dict[int, dict]:
    """Load progress file into {index -> record}."""
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = int(rec[index_key])
            out[idx] = rec
    return out


def _compute_resume_key(
    model_kind: str,
    args: argparse.Namespace,
    *,
    overrides: Optional[dict] = None,
) -> str:
    """Compute a stable key for resume files from model+params."""
    payload = {
        "model_kind": model_kind,
        "data_path": str(getattr(args, "data_path", "")),
        "max_samples": getattr(args, "max_samples", None),
        "run": getattr(args, "run", None),
        "save_explanations": bool(getattr(args, "save_explanations", False)),
        # dual_full params
        "dual_full_light_explain": bool(getattr(args, "dual_full_light_explain", False)),
        "dual_full_max_new_tokens_explain": int(getattr(args, "dual_full_max_new_tokens_explain", 224)),
        "dual_full_max_seq_len_explain": int(getattr(args, "dual_full_max_seq_len_explain", 4096)),
        # model paths (used to make key stable per checkpoint)
        "expert_model": str(getattr(args, "expert_model", "")),
        "student_cls_base": str(getattr(args, "student_cls_base", "")),
        "student_cls_lora": str(getattr(args, "student_cls_lora", "")),
        "student_explain_lora": str(getattr(args, "student_explain_lora", "")),
        "student_single_model": str(getattr(args, "student_single_model", "")),
    }
    if overrides:
        payload.update(overrides)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def run_scamnet(
    data: List[dict],
    model_path: Path,
    max_seq_length: int = 8192,
    input_char_limit: int = 8000,
    max_new_tokens_pass1: int = 60,
    max_new_tokens_pass2: int = 512,
    save_explanations: bool = False,
    resume_progress_path: Optional[Path] = None,
) -> Tuple[List[int], float, float, Optional[List[str]]]:
    """ScamNet 专家模型：Transformers + BitsAndBytes 4bit 基座 + Peft LoRA，attn=sdpa（不用 Unsloth，与蒸馏侧对齐便于公平对比）。
    逻辑仍对齐两段生成与后处理（evaluate_step2_fixed3 风格）。
    返回 (y_pred, 单样本耗时秒, 显存GB, 若 save_explanations 则每样本的完整生成文本列表否则 None)。"""
    import os

    import torch
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _set_torch_alloc_env()

    # 与 evaluate_step2_fixed3.py 一致
    def truncate_input(text: str, max_chars: int = input_char_limit) -> str:
        if len(text) <= max_chars:
            return text
        return text[: int(max_chars * 0.8)] + "\n...[Content Truncated]..."

    def format_prompt(sample_input: str) -> str:
        clean = sample_input.replace("# Pred:", "").strip()
        if "# Information:" not in clean:
            clean = f"# Information:\n{clean}"
        return f"{clean}\n\n# Pred:\n"

    def extract_label(text: str) -> int:
        head = (text.strip() or "").split("\n")[0].lower()
        if "label: scam" in head or head.startswith("scam"):
            return 1
        if "label: legit" in head or head.startswith("legit"):
            return 0
        head_loose = (text[:50] or "").lower()
        if "scam" in head_loose:
            return 1
        return 0

    def check_has_explanation(text: str) -> bool:
        """是否有任意 ## 开头的小节（与 fixed3 一致，用于兼容）。"""
        return "##" in text or len(text.split("\n")) > 2

    def check_has_reason(text: str) -> bool:
        """是否包含 ## Reason: 小节（强制要求有理由说明）。"""
        return "## reason" in (text or "").lower()

    def deduplicate_reason_section(text: str) -> str:
        """后处理：移除 ## Reason: 内的重复句式，减轻循环重复与冗余。"""
        lower = text.lower()
        idx = lower.find("## reason")
        if idx < 0:
            return text
        before = text[:idx]
        rest = text[idx:]
        colon = rest.find(":")
        head_end = rest.find("\n", colon + 1) + 1 if colon >= 0 else len(rest)
        reason_head = rest[:head_end]
        after_header = rest[head_end:]
        if "\n## " in after_header:
            reason_body, tail_part = after_header.split("\n## ", 1)
            tail = "\n## " + tail_part
        else:
            reason_body = after_header
            tail = ""
        reason_body = reason_body.strip()
        if not reason_body:
            return text
        # 按句号分句
        sentences = []
        for part in reason_body.replace(".\n", ". ").split(". "):
            s = part.strip()
            if not s:
                continue
            if not s.endswith("."):
                s = s + "."
            sentences.append(s)
        # 去重：连续相同句只保留一句；与前句开头 40 字符相同视为重复
        seen_starts = set()
        out = []
        prev = ""
        for s in sentences:
            if s == prev:
                continue
            start_key = (s + " ")[:40].strip()
            if start_key in seen_starts:
                continue
            seen_starts.add(start_key)
            prev = s
            out.append(s)
        new_body = " ".join(out)
        return before + reason_head + new_body + tail

    def smart_truncate(text: str, max_length: int = 600) -> str:
        """智能截断：在完整句子边界处截断，避免句中硬切。优先级质量>速度。"""
        if len(text) <= max_length:
            return text
        truncated = text[:max_length]
        last_period = truncated.rfind("。")
        if last_period == -1:
            last_period = truncated.rfind(". ")
        if last_period == -1:
            last_period = truncated.rfind(".")
        if last_period == -1:
            last_period = truncated.rfind("!")
        if last_period == -1:
            last_period = truncated.rfind("?")
        if last_period > max_length * 0.8:
            return text[: last_period + 1]
        return text[:max_length].rstrip() + "..."

    def clean_explanation(text: str, max_len: int = 600, min_len: int = 50) -> str:
        """后处理：移除代码块、References、免责声明，并用智能截断控制长度。仅用于原始专家模型输出。"""
        if not text or not text.strip():
            return text
        original = text
        # 1. 移除 ```...``` 代码块（含 ```python 等）
        text = re.sub(r"```[\s\S]*?```", "", text)
        # 2. 移除 ## References: 整段（至下一小节或结尾）
        text = re.sub(r"\n?## References:\s*[\s\S]*?(?=\n\n## |\n\n# |\Z)", "", text, flags=re.IGNORECASE | re.DOTALL)
        # 3. 移除免责声明类句式
        text = re.sub(r"Please note that[^\n]*(?=\n|$)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"I do not claim any responsibility[^\n]*(?=\n|$)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Disclaimer[^\n]*(?=\n|$)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"This (?:is )?not[^\n]*(?:professional|legal|financial) advice[^\n]*(?=\n|$)", "", text, flags=re.IGNORECASE)
        # 4. 合并多余空行并首尾去空
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        # 5. 智能截断（在句号等处截断，上限放宽至 600 字）
        if len(text) > max_len:
            text = smart_truncate(text, max_length=max_len)
        # 6. 若清理后过短则保留截断后的原文（至少保留 Label 与简短说明）
        if len(text) < min_len and len(original) > min_len:
            fallback = re.sub(r"```[\s\S]*?```", "", original)
            fallback = re.sub(r"\n?## References:\s*[\s\S]*?(?=\n\n## |\n\n# |\Z)", "", fallback, flags=re.IGNORECASE | re.DOTALL)
            fallback = fallback.strip()
            if len(fallback) > max_len:
                fallback = smart_truncate(fallback, max_length=max_len)
            if len(fallback) >= min_len:
                text = fallback
        return text

    # 抑制重复：生成时使用 repetition_penalty。质量优先：第二段统一放宽 token 上限，避免生成时被截断
    repetition_penalty = 1.2
    pass2_tokens = max(512, max_new_tokens_pass2)

    print("  [加载] expert: Transformers 4bit + LoRA + sdpa（无 Unsloth）")
    adapter_cfg = model_path / "adapter_config.json"
    if not adapter_cfg.is_file():
        raise FileNotFoundError(f"专家目录需含 adapter_config.json（LoRA）: {model_path}")
    with open(adapter_cfg, "r", encoding="utf-8") as f:
        acfg = json.load(f)
    base_path = acfg.get("base_model_name_or_path")
    if not base_path:
        raise ValueError("adapter_config.json 缺少 base_model_name_or_path")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=hf_local_files_only(str(model_path))
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = _from_pretrained_4bit_causal_lm(
        str(base_path),
        local_files_only=hf_local_files_only(str(base_path)),
        fallback_label="expert 基座",
    )
    model = PeftModel.from_pretrained(base_model, str(model_path))
    model.eval()
    device = next(model.parameters()).device

    n = len(data)
    resume_enabled = resume_progress_path is not None
    done_map = _jsonl_load_index_map(resume_progress_path) if resume_enabled else {}

    # 断点续传：y_pred/response 只在缺失 index 时重新计算
    y_pred: List[int] = [-1 for _ in range(n)]
    if save_explanations:
        responses: Optional[List[str]] = ["" for _ in range(n)]
    else:
        responses = None
    for idx, rec in done_map.items():
        if 0 <= idx < n:
            y_pred[idx] = int(rec["y_pred"])
            if responses is not None and "response" in rec:
                responses[idx] = str(rec["response"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    pass2_used = 0
    gen1_time_sum = 0.0
    gen2_time_sum = 0.0
    for i, item in enumerate(tqdm(data, desc="ScamNet")):
        if y_pred[i] != -1:
            continue
        input_text = truncate_input(item["input"])
        prompt = format_prompt(input_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            t_gen1 = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens_pass1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=repetition_penalty,
            )
            gen1_time_sum += time.perf_counter() - t_gen1
        gen_text_pass1 = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        # 强制生成 ## Reason:：仅当第一段已包含 ## Reason: 才接受，否则第二段续写并加简洁/去重提示
        if not check_has_reason(gen_text_pass1):
            pass2_used += 1
            reason_hint = "\n\n## Reason:\n(Be concise. Do not repeat the same phrase. Focus on this URL only.)\n"
            continuation_prompt = prompt + gen_text_pass1 + reason_hint
            inputs_2 = tokenizer(continuation_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                t_gen2 = time.perf_counter()
                outputs_2 = model.generate(
                    **inputs_2,
                    max_new_tokens=pass2_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    repetition_penalty=repetition_penalty,
                )
                gen2_time_sum += time.perf_counter() - t_gen2
            gen_text_pass2 = tokenizer.decode(
                outputs_2[0][inputs_2["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            final_gen_text = gen_text_pass1 + "\n\n## Reason:\n" + gen_text_pass2
        else:
            final_gen_text = gen_text_pass1
        final_gen_text = deduplicate_reason_section(final_gen_text)
        final_gen_text = clean_explanation(final_gen_text)
        pred_i = extract_label(final_gen_text)
        y_pred[i] = pred_i
        if responses is not None:
            responses[i] = final_gen_text

        if resume_enabled:
            rec = {"index": i, "y_pred": pred_i}
            if responses is not None:
                rec["response"] = final_gen_text
            _jsonl_append(resume_progress_path, rec)
    t_end = time.perf_counter()
    time_per_sample = (t_end - t_start) / n if n else 0.0
    pass2_rate = pass2_used / n if n else 0.0
    print(
        f"  [统计] pass2触发: {pass2_used}/{n} ({pass2_rate:.1%}); "
        f"gen1时间总计={gen1_time_sum:.2f}s; gen2时间总计={gen2_time_sum:.2f}s"
    )

    # 保证断点续传后数据完整
    missing = [j for j, v in enumerate(y_pred) if v == -1]
    if missing:
        raise RuntimeError(f"[resume] ScamNet: missing predictions for {len(missing)} samples, first={missing[:5]}")
    if save_explanations and responses is not None:
        missing_resp = [j for j, v in enumerate(responses) if not v]
        if missing_resp:
            raise RuntimeError(
                f"[resume] ScamNet: missing explanations for {len(missing_resp)} samples, first={missing_resp[:5]}"
            )

    gpu_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    return y_pred, time_per_sample, gpu_gb, responses


def run_student_single(
    data: List[dict],
    student_model: str,
    max_seq_length: int = 8192,
    input_char_limit: int = 8000,
    max_new_tokens_pass1: int = 60,
    max_new_tokens_pass2: int = 512,
    save_explanations: bool = False,
) -> Tuple[List[int], float, float, Optional[List[str]]]:
    """蒸馏单模型评测：与 expert 一致使用 Transformers + sdpa（LoRA=4bit 基座+Peft；merged=整模 fp16/bf16 或 bnb）。"""
    import os
    from pathlib import Path

    import torch
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _set_torch_alloc_env()

    # 与 run_scamnet 对齐
    def truncate_input(text: str, max_chars: int = input_char_limit) -> str:
        if len(text) <= max_chars:
            return text
        return text[: int(max_chars * 0.8)] + "\n...[Content Truncated]..."

    def format_prompt(sample_input: str) -> str:
        clean = sample_input.replace("# Pred:", "").strip()
        if "# Information:" not in clean:
            clean = f"# Information:\n{clean}"
        return f"{clean}\n\n# Pred:\n"

    def extract_label(text: str) -> int:
        head = (text.strip() or "").split("\n")[0].lower()
        if "label: scam" in head or head.startswith("scam"):
            return 1
        if "label: legit" in head or head.startswith("legit"):
            return 0
        head_loose = (text[:50] or "").lower()
        if "scam" in head_loose:
            return 1
        return 0

    def check_has_reason(text: str) -> bool:
        return "## reason" in (text or "").lower()

    def deduplicate_reason_section(text: str) -> str:
        """后处理：移除 ## Reason: 内的重复句式，减轻循环重复与冗余。"""
        lower = text.lower()
        idx = lower.find("## reason")
        if idx < 0:
            return text
        before = text[:idx]
        rest = text[idx:]
        colon = rest.find(":")
        head_end = rest.find("\n", colon + 1) + 1 if colon >= 0 else len(rest)
        reason_head = rest[:head_end]
        after_header = rest[head_end:]
        if "\n## " in after_header:
            reason_body, tail_part = after_header.split("\n## ", 1)
            tail = "\n## " + tail_part
        else:
            reason_body = after_header
            tail = ""
        reason_body = reason_body.strip()
        if not reason_body:
            return text
        sentences = []
        for part in reason_body.replace(".\n", ". ").split(". "):
            s = part.strip()
            if not s:
                continue
            if not s.endswith("."):
                s = s + "."
            sentences.append(s)
        seen_starts = set()
        out = []
        prev = ""
        for s in sentences:
            if s == prev:
                continue
            start_key = (s + " ")[:40].strip()
            if start_key in seen_starts:
                continue
            seen_starts.add(start_key)
            prev = s
            out.append(s)
        new_body = " ".join(out)
        return before + reason_head + new_body + tail

    def smart_truncate(text: str, max_length: int = 600) -> str:
        if len(text) <= max_length:
            return text
        truncated = text[:max_length]
        last_period = truncated.rfind("。")
        if last_period == -1:
            last_period = truncated.rfind(". ")
        if last_period == -1:
            last_period = truncated.rfind(".")
        if last_period == -1:
            last_period = truncated.rfind("!")
        if last_period == -1:
            last_period = truncated.rfind("?")
        if last_period > max_length * 0.8:
            return text[: last_period + 1]
        return text[:max_length].rstrip() + "..."

    def clean_explanation(text: str, max_len: int = 600, min_len: int = 50) -> str:
        """后处理：移除代码块/References/免责声明，并做智能截断。"""
        if not text or not text.strip():
            return text
        original = text
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(
            r"\n?## References:\s*[\s\S]*?(?=\n\n## |\n\n# |\Z)",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"Please note that[^\n]*(?=\n|$)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"I do not claim any responsibility[^\n]*(?=\n|$)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Disclaimer[^\n]*(?=\n|$)", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r"This (?:is )?not[^\n]*(?:professional|legal|financial) advice[^\n]*(?=\n|$)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > max_len:
            text = smart_truncate(text, max_length=max_len)
        if len(text) < min_len and len(original) > min_len:
            fallback = re.sub(r"```[\s\S]*?```", "", original)
            fallback = re.sub(
                r"\n?## References:\s*[\s\S]*?(?=\n\n## |\n\n# |\Z)",
                "",
                fallback,
                flags=re.IGNORECASE | re.DOTALL,
            )
            fallback = fallback.strip()
            if len(fallback) > max_len:
                fallback = smart_truncate(fallback, max_length=max_len)
            if len(fallback) >= min_len:
                text = fallback
        return text

    # 抑制重复（与 run_scamnet 对齐）
    repetition_penalty = 1.2
    pass2_tokens = max(512, max_new_tokens_pass2)

    sp = Path(student_model)
    print("  [加载] student_single: Transformers + sdpa（无 Unsloth）")
    tokenizer = AutoTokenizer.from_pretrained(str(sp), local_files_only=hf_local_files_only(str(sp)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if (sp / "adapter_config.json").is_file():
        with open(sp / "adapter_config.json", "r", encoding="utf-8") as f:
            acfg = json.load(f)
        base_path = acfg.get("base_model_name_or_path")
        if not base_path:
            raise ValueError("adapter_config.json 缺少 base_model_name_or_path")
        base_model = _from_pretrained_4bit_causal_lm(
            str(base_path),
            local_files_only=hf_local_files_only(str(base_path)),
            fallback_label="student_single LoRA 基座",
        )
        model = PeftModel.from_pretrained(base_model, str(sp))
        model.eval()
    else:
        cfg_path = sp / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"非 LoRA 目录需含 config.json: {sp}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            mcfg = json.load(f)
        qcfg = mcfg.get("quantization_config") or {}
        quant_4bit = bool(qcfg.get("load_in_4bit") or qcfg.get("_load_in_4bit"))
        if quant_4bit:
            model = _from_pretrained_4bit_causal_lm(
                str(sp),
                local_files_only=hf_local_files_only(str(sp)),
                fallback_label="student_single merged 4bit",
            )
        else:
            ds = str(mcfg.get("dtype") or "float16").lower()
            td = torch.bfloat16 if "bfloat" in ds or ds == "bf16" else torch.float16
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(sp),
                    torch_dtype=td,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    local_files_only=hf_local_files_only(str(sp)),
                    attn_implementation="sdpa",
                )
            except Exception as e:
                print(f"  [警告] sdpa 加载失败 ({e})，回退 attn_implementation=eager")
                model = AutoModelForCausalLM.from_pretrained(
                    str(sp),
                    torch_dtype=td,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    local_files_only=hf_local_files_only(str(sp)),
                    attn_implementation="eager",
                )
        model.eval()

    device = next(model.parameters()).device

    y_pred: List[int] = []
    responses: Optional[List[str]] = [] if save_explanations else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    pass2_used = 0
    gen1_time_sum = 0.0
    gen2_time_sum = 0.0
    for item in tqdm(data, desc="Student-Single"):
        input_text = truncate_input(item["input"])
        prompt = format_prompt(input_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            t_gen1 = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens_pass1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=repetition_penalty,
            )
            gen1_time_sum += time.perf_counter() - t_gen1
        gen_text_pass1 = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        if not check_has_reason(gen_text_pass1):
            pass2_used += 1
            reason_hint = "\n\n## Reason:\n(Be concise. Do not repeat the same phrase. Focus on this URL only.)\n"
            continuation_prompt = prompt + gen_text_pass1 + reason_hint
            inputs_2 = tokenizer(continuation_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                t_gen2 = time.perf_counter()
                outputs_2 = model.generate(
                    **inputs_2,
                    max_new_tokens=pass2_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    repetition_penalty=repetition_penalty,
                )
                gen2_time_sum += time.perf_counter() - t_gen2
            gen_text_pass2 = tokenizer.decode(
                outputs_2[0][inputs_2["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            final_gen_text = gen_text_pass1 + "\n\n## Reason:\n" + gen_text_pass2
        else:
            final_gen_text = gen_text_pass1

        final_gen_text = deduplicate_reason_section(final_gen_text)
        final_gen_text = clean_explanation(final_gen_text)
        y_pred.append(extract_label(final_gen_text))
        if responses is not None:
            responses.append(final_gen_text)
    t_end = time.perf_counter()

    n = len(data)
    time_per_sample = (t_end - t_start) / n if n else 0.0
    pass2_rate = pass2_used / n if n else 0.0
    print(
        f"  [统计] pass2触发: {pass2_used}/{n} ({pass2_rate:.1%}); "
        f"gen1时间总计={gen1_time_sum:.2f}s; gen2时间总计={gen2_time_sum:.2f}s"
    )
    gpu_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    return y_pred, time_per_sample, gpu_gb, responses


def run_dual_student_cls(
    data: List[dict],
    cls_base: str,
    cls_lora: str,
    resume_cls_progress_path: Optional[Path] = None,
    input_char_limit: int = 8000,
) -> Tuple[List[int], float, float]:
    """双学生仅分类（Mistral Step1）。
    支持样本级断点续传：resume 文件每行格式 {"index": i, "y_pred": 0/1}。
    返回 (y_pred, 单样本耗时秒, 显存GB)。"""
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(cls_base, local_files_only=hf_local_files_only(cls_base))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = _from_pretrained_4bit_causal_lm(
        cls_base, local_files_only=hf_local_files_only(cls_base), fallback_label="dual_cls 基座"
    )
    model = PeftModel.from_pretrained(base, cls_lora)
    model.eval()
    device = next(model.parameters()).device

    def truncate_input(text: str, max_chars: int = input_char_limit) -> str:
        if len(text) <= max_chars:
            return text
        return text[: int(max_chars * 0.8)] + "\n...[Content Truncated]..."

    def score_candidate(prompt: str, completion: str) -> float:
        with torch.no_grad():
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if not comp_ids:
                return float("-inf")
            input_ids = torch.tensor(
                [prompt_ids + comp_ids], dtype=torch.long, device=device
            )
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

    def predict(inp: str) -> int:
        prompt = f"{truncate_input(inp)}\n# Pred:\n"
        s_scam = score_candidate(prompt, "Label: scam")
        s_legit = score_candidate(prompt, "Label: legit")
        return 1 if s_scam >= s_legit else 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    n = len(data)
    resume_enabled = resume_cls_progress_path is not None
    done_map = _jsonl_load_index_map(resume_cls_progress_path) if resume_enabled else {}
    y_pred: List[int] = [-1 for _ in range(n)]
    for idx, rec in done_map.items():
        if 0 <= idx < n:
            y_pred[idx] = int(rec["y_pred"])

    for i, item in enumerate(tqdm(data, desc="DualStudent-Cls")):
        if y_pred[i] != -1:
            continue
        pred_i = predict(item["input"])
        y_pred[i] = pred_i
        if resume_enabled:
            _jsonl_append(resume_cls_progress_path, {"index": i, "y_pred": pred_i})
    t_end = time.perf_counter()
    time_per_sample = (t_end - t_start) / n if n else 0.0

    missing = [j for j, v in enumerate(y_pred) if v == -1]
    if missing:
        raise RuntimeError(
            f"[resume] DualCls: missing predictions for {len(missing)} samples, first={missing[:5]}"
        )

    gpu_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    del model, base
    torch.cuda.empty_cache()
    return y_pred, time_per_sample, gpu_gb


def run_dual_student_full(
    data: List[dict],
    cls_base: str,
    cls_lora: str,
    explain_lora: str,
    max_new_tokens_explain: int = 224,
    max_seq_len_explain: int = 4096,
    light_explain: bool = False,
    save_explanations: bool = False,
    resume_cls_progress_path: Optional[Path] = None,
    resume_exp_progress_path: Optional[Path] = None,
    input_char_limit: int = 8000,
    y_pred_external: Optional[List[int]] = None,
) -> Tuple[List[int], float, float, Optional[List[str]]]:
    """蒸馏双学生：先分类再解释。返回 (y_pred, 单样本总耗时秒, 显存峰值GB, 若 save_explanations 则每样本的完整生成文本列表否则 None)。

    y_pred_external:
        若提供与 ``data`` 等长的 0/1 列表，则跳过分类 LoRA，直接以该标签为先验进入解释阶段
        （PhishGuard 前端与论文「快速判别 → 可解释分析」对齐）。提供时 ``resume_cls_progress_path`` 不参与分类断点。
    """
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from tqdm import tqdm

    def _format_prompt_explain(inp: str) -> str:
        clean = inp.replace("# Pred:", "").strip()
        if "# Information:" not in clean:
            clean = f"# Information:\n{clean}"
        return f"{clean}\n\n# Pred:\n"

    def truncate_input(text: str, max_chars: int = input_char_limit) -> str:
        if len(text) <= max_chars:
            return text
        return text[: int(max_chars * 0.8)] + "\n...[Content Truncated]..."

    n = len(data)
    t_cls_start = time.perf_counter()
    peak_cls = 0.0

    if y_pred_external is not None:
        if len(y_pred_external) != n:
            raise ValueError(
                f"y_pred_external 长度必须与 data 一致: expected {n}, got {len(y_pred_external)}"
            )
        for j, y in enumerate(y_pred_external):
            if int(y) not in (0, 1):
                raise ValueError(f"y_pred_external[{j}]={y!r} 非法，仅允许 0 或 1")
        y_pred = [int(y) for y in y_pred_external]
        t_cls_end = time.perf_counter()
    else:
        # ---------- 阶段 1：分类 ----------
        tok_cls = AutoTokenizer.from_pretrained(cls_base, local_files_only=hf_local_files_only(cls_base))
        if tok_cls.pad_token is None:
            tok_cls.pad_token = tok_cls.eos_token
        base_cls = _from_pretrained_4bit_causal_lm(
            cls_base, local_files_only=hf_local_files_only(cls_base), fallback_label="dual_full 分类基座"
        )
        cls_model = PeftModel.from_pretrained(base_cls, cls_lora)
        cls_model.eval()
        device_cls = next(cls_model.parameters()).device

        def score_candidate(prompt: str, completion: str) -> float:
            with torch.no_grad():
                prompt_ids = tok_cls(prompt, add_special_tokens=False)["input_ids"]
                comp_ids = tok_cls(completion, add_special_tokens=False)["input_ids"]
                if not comp_ids:
                    return float("-inf")
                input_ids = torch.tensor(
                    [prompt_ids + comp_ids], dtype=torch.long, device=device_cls
                )
                attn = torch.ones_like(input_ids, device=device_cls)
                out = cls_model(input_ids=input_ids, attention_mask=attn)
                logp = out.logits.log_softmax(dim=-1)
                prompt_len = len(prompt_ids)
                total = 0.0
                for i, tok_id in enumerate(comp_ids):
                    pos = prompt_len + i
                    if pos == 0:
                        continue
                    total += float(logp[0, pos - 1, tok_id].item())
                return total

        def predict_cls(inp: str) -> int:
            prompt = f"{truncate_input(inp)}\n# Pred:\n"
            s_scam = score_candidate(prompt, "Label: scam")
            s_legit = score_candidate(prompt, "Label: legit")
            return 1 if s_scam >= s_legit else 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        resume_cls_enabled = resume_cls_progress_path is not None
        done_cls_map = _jsonl_load_index_map(resume_cls_progress_path) if resume_cls_enabled else {}
        y_pred = [-1 for _ in range(n)]
        for idx, rec in done_cls_map.items():
            if 0 <= idx < n:
                y_pred[idx] = int(rec["y_pred"])

        for i, item in enumerate(tqdm(data, desc="DualFull-Cls")):
            if y_pred[i] != -1:
                continue
            pred_i = predict_cls(item["input"])
            y_pred[i] = pred_i
            if resume_cls_enabled:
                _jsonl_append(resume_cls_progress_path, {"index": i, "y_pred": pred_i})

        t_cls_end = time.perf_counter()
        peak_cls = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )
        del cls_model, base_cls
        torch.cuda.empty_cache()

    # 阶段 2：解释生成
    tok_exp = AutoTokenizer.from_pretrained(cls_base, local_files_only=hf_local_files_only(cls_base))
    if tok_exp.pad_token is None:
        tok_exp.pad_token = tok_exp.eos_token
    base_exp = _from_pretrained_4bit_causal_lm(
        cls_base, local_files_only=hf_local_files_only(cls_base), fallback_label="dual_full 解释基座"
    )
    if torch.cuda.is_available() and hasattr(base_exp, "config"):
        base_exp.config.use_cache = False
    exp_model = PeftModel.from_pretrained(base_exp, explain_lora)
    exp_model.eval()
    device_exp = next(exp_model.parameters()).device

    t_exp_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    resume_exp_enabled = resume_exp_progress_path is not None and save_explanations
    done_exp_map = _jsonl_load_index_map(resume_exp_progress_path) if resume_exp_enabled else {}
    if save_explanations:
        responses: Optional[List[str]] = ["" for _ in range(n)]
        for idx, rec in done_exp_map.items():
            if 0 <= idx < n and "response" in rec:
                responses[idx] = str(rec["response"])
    else:
        responses = None

    for i in tqdm(range(n), total=n, desc="DualFull-Explain"):
        if save_explanations and i in done_exp_map:
            continue
        pred = y_pred[i]
        if pred == -1:
            raise RuntimeError(f"[resume] DualFull: missing cls prediction at index {i}")

        item = data[i]
        label_str = "scam" if pred == 1 else "legit"
        base_prompt = _format_prompt_explain(truncate_input(item["input"]))
        if light_explain:
            reason_hint = (
                "Write a concise reason only. Use 1-2 short sentences, focus on strongest phishing cues, "
                "avoid background and repetition, and keep total explanation under 160 tokens."
            )
            prompt_exp = f"{base_prompt}Label: {label_str}\n\n## Reason:\n{reason_hint}\n"
        else:
            prompt_exp = f"{base_prompt}Label: {label_str}\n\n## Reason:\n"
        inputs = tok_exp(
            prompt_exp,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len_explain,
        ).to(device_exp)
        with torch.no_grad():
            outputs = exp_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens_explain,
                use_cache=True,
                pad_token_id=tok_exp.eos_token_id,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
        gen_explain = tok_exp.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        full_response = f"Label: {label_str}\n\n## Reason:\n{gen_explain}"
        if responses is not None:
            responses[i] = full_response
        if resume_exp_enabled:
            _jsonl_append(resume_exp_progress_path, {"index": i, "response": full_response})

    t_exp_end = time.perf_counter()
    peak_exp = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    # 保证断点续传后数据完整
    missing_cls = [j for j, v in enumerate(y_pred) if v == -1]
    if missing_cls:
        raise RuntimeError(f"[resume] DualFull: missing cls predictions for {len(missing_cls)} samples, first={missing_cls[:5]}")
    if save_explanations and responses is not None:
        missing_resp = [j for j, v in enumerate(responses) if not v]
        if missing_resp:
            raise RuntimeError(
                f"[resume] DualFull: missing explanations for {len(missing_resp)} samples, first={missing_resp[:5]}"
            )

    del exp_model, base_exp
    torch.cuda.empty_cache()

    time_per_sample = (t_cls_end - t_cls_start + t_exp_end - t_exp_start) / n if n else 0.0
    gpu_gb = max(peak_cls, peak_exp)
    return y_pred, time_per_sample, gpu_gb, responses


def main():
    parser = argparse.ArgumentParser(description="多模型钓鱼检测统一评测")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "dataset_test_strict.json",
        help="统一测试集 JSON",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多评测样本数；限制时固定取测试集前 N 条，不随机抽取（默认全量）",
    )
    parser.add_argument(
        "--run",
        choices=["expert", "dual_full", "dual_cls", "student_single", "all"],
        default="dual_cls",
        help="评测模型；all=依次运行已就绪模型。轻量基线见 09_baselines/random_forest/",
    )
    parser.add_argument(
        "--expert-model",
        type=Path,
        default=Path("outputs/expert/scamnet_final_model"),
        help="ScamNet 专家模型路径",
    )
    parser.add_argument(
        "--student-cls-base",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="双学生分类基座路径",
    )
    parser.add_argument(
        "--student-cls-lora",
        type=str,
        default=str(DEFAULT_CLS_LORA),
        help="双学生分类 LoRA 路径（默认 outputs/decouple/cls_adapter）",
    )
    parser.add_argument(
        "--student-explain-lora",
        type=str,
        default=str(DEFAULT_EXPLAIN_LORA),
        help="双学生解释 LoRA 路径（仅 dual_full 使用）",
    )
    parser.add_argument(
        "--dual-full-max-new-tokens-explain",
        type=int,
        default=224,
        help="dual_full 解释阶段单样本生成 token 上限（默认 224）",
    )
    parser.add_argument(
        "--dual-full-max-seq-len-explain",
        type=int,
        default=4096,
        help="dual_full 解释阶段输入 max_length（默认 4096）",
    )
    parser.add_argument(
        "--dual-full-light-explain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用轻量解释模式：自动压缩 dual_full 解释长度（更快但解释更短）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="结果 JSON 输出路径（可选）",
    )
    parser.add_argument(
        "--save-explanations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否将生成解释写入结果文件（默认开启，可用 --no-save-explanations 关闭）",
    )

    parser.add_argument(
        "--resume-progress-dir",
        type=Path,
        default=None,
        help="开启真正断点续传：将样本级进度写入该目录（jsonl），重启后自动跳过已完成 index。",
    )
    parser.add_argument(
        "--resume-tag",
        type=str,
        default=None,
        help="可选：自定义断点续传key（避免因参数变化/命名导致无法续跑）。",
    )

    # 蒸馏单模型（最终 Step2 LoRA）
    parser.add_argument(
        "--student-single-model",
        type=str,
        default="outputs/monolithic/step2_explain",
        help="蒸馏单模型：含 adapter_config.json 则为 LoRA（4bit 基座+Peft）；否则按 merged 全量（fp16/bf16 或 config 内 4bit）加载。",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"[ERROR] 测试集不存在: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    data, y_true = load_test_data(args.data_path, args.max_samples)
    n = len(data)
    sample_note = f"（固定前 {n} 条）" if args.max_samples is not None else ""
    print(f"测试集: {args.data_path.name}, 样本数: {n} {sample_note}")
    print()

    results = []

    def add_result(
        name: str,
        y_pred: List[int],
        time_per_sample: float,
        gpu_gb: float,
        generated_responses: Optional[List[str]] = None,
    ):
        m = compute_metrics(y_true, y_pred)
        m["model_name"] = name
        m["time_per_sample_s"] = round(time_per_sample, 4)
        m["gpu_peak_gb"] = round(gpu_gb, 2)
        m["bad_cases"] = build_bad_cases(data, y_true, y_pred)
        m["bad_cases_count"] = len(m["bad_cases"])
        if generated_responses is not None and len(generated_responses) == len(data):
            m["generated_explanations"] = [
                {
                    "index": i,
                    "url": _extract_url(data[i].get("input", "")),
                    "true_label": "scam" if y_true[i] == 1 else "legit",
                    "pred_label": "scam" if y_pred[i] == 1 else "legit",
                    "generated_response": generated_responses[i],
                }
                for i in range(len(data))
            ]
        results.append(m)

    run_expert = args.run in ("expert", "all")
    run_dual_full = args.run in ("dual_full", "all")
    run_dual_cls = args.run in ("dual_cls", "all")
    run_student_single_flag = args.run in ("student_single", "all")

    resume_enabled = args.resume_progress_dir is not None
    resume_dir: Optional[Path] = args.resume_progress_dir

    if run_expert:
        print("[原始专家] ScamNet ...")
        if not args.expert_model.exists():
            print(f"  跳过：模型不存在 {args.expert_model}")
        else:
            expert_resume_path = None
            if resume_enabled and resume_dir is not None:
                resume_key = args.resume_tag or _compute_resume_key("expert", args)
                expert_resume_path = resume_dir / f"{resume_key}_expert.jsonl"
            y_pred, tps, gb, resp = run_scamnet(
                data,
                args.expert_model,
                save_explanations=args.save_explanations,
                resume_progress_path=expert_resume_path,
            )
            add_result("原始专家(ScamNet)", y_pred, tps, gb, generated_responses=resp)
            print(f"  Precision={results[-1]['precision']}, Recall={results[-1]['recall']}, F1={results[-1]['f1_score']}, 单样本={tps:.3f}s, 显存={gb:.2f}GB")

    if run_student_single_flag:
        print("[蒸馏单模型] Step2（与 expert 一致：Transformers+sdpa，无 Unsloth） ...")
        student_single_dir = Path(args.student_single_model) if args.student_single_model else None
        if not student_single_dir or not student_single_dir.exists():
            print(f"  跳过：模型不存在 {args.student_single_model}")
        elif not (student_single_dir / "adapter_config.json").exists():
            print(
                "  跳过：当前 student_single 仅支持 LoRA 目录（需包含 adapter_config.json）；"
                f"检测到不兼容目录: {student_single_dir}"
            )
        else:
            y_pred, tps, gb, resp = run_student_single(
                data=data,
                student_model=str(student_single_dir),
                save_explanations=args.save_explanations,
            )
            add_result("蒸馏单模型(Step2)", y_pred, tps, gb, generated_responses=resp)
            print(
                f"  Precision={results[-1]['precision']}, Recall={results[-1]['recall']}, "
                f"F1={results[-1]['f1_score']}, 单样本={tps:.3f}s, 显存={gb:.2f}GB"
            )

    if run_dual_full:
        print("[蒸馏双学生] 分类+解释 ...")
        if not Path(args.student_cls_lora).exists():
            print(f"  跳过：分类 LoRA 不存在 {args.student_cls_lora}")
        elif not Path(args.student_explain_lora).exists():
            print(f"  跳过：解释 LoRA 不存在 {args.student_explain_lora}")
        else:
            dual_full_model_name = "蒸馏双学生(分类+解释)"
            max_tokens_explain = args.dual_full_max_new_tokens_explain
            max_seq_len_explain = args.dual_full_max_seq_len_explain
            if getattr(args, "dual_full_light_explain", False):
                # 轻量模式：进一步压缩解释长度，加速评测
                max_tokens_explain = min(max_tokens_explain, 160)
                max_seq_len_explain = min(max_seq_len_explain, 2048)
                dual_full_model_name = "蒸馏双学生(分类+解释)[light]"

            dual_full_resume_cls_path = None
            dual_full_resume_exp_path = None
            if resume_enabled and resume_dir is not None:
                overrides = {
                    "dual_full_max_new_tokens_explain": int(max_tokens_explain),
                    "dual_full_max_seq_len_explain": int(max_seq_len_explain),
                }
                resume_key_cls = args.resume_tag or _compute_resume_key("dual_full_cls", args, overrides=overrides)
                resume_key_exp = args.resume_tag or _compute_resume_key("dual_full_exp", args, overrides=overrides)
                dual_full_resume_cls_path = resume_dir / f"{resume_key_cls}_dual_full_cls.jsonl"
                dual_full_resume_exp_path = resume_dir / f"{resume_key_exp}_dual_full_exp.jsonl"
            y_pred, tps, gb, resp = run_dual_student_full(
                data,
                args.student_cls_base,
                args.student_cls_lora,
                args.student_explain_lora,
                max_new_tokens_explain=max_tokens_explain,
                max_seq_len_explain=max_seq_len_explain,
                light_explain=bool(getattr(args, "dual_full_light_explain", False)),
                save_explanations=args.save_explanations,
                resume_cls_progress_path=dual_full_resume_cls_path,
                resume_exp_progress_path=dual_full_resume_exp_path,
            )
            add_result(dual_full_model_name, y_pred, tps, gb, generated_responses=resp)
            print(f"  Precision={results[-1]['precision']}, Recall={results[-1]['recall']}, F1={results[-1]['f1_score']}, 单样本={tps:.3f}s, 显存={gb:.2f}GB")

    if run_dual_cls:
        print("[蒸馏学生只分类] 不解释 ...")
        if not Path(args.student_cls_lora).exists():
            print(f"  跳过：LoRA 不存在 {args.student_cls_lora}")
        else:
            dual_cls_resume_path = None
            if resume_enabled and resume_dir is not None:
                resume_key = args.resume_tag or _compute_resume_key("dual_cls", args)
                dual_cls_resume_path = resume_dir / f"{resume_key}_dual_cls.jsonl"
            y_pred, tps, gb = run_dual_student_cls(
                data,
                args.student_cls_base,
                args.student_cls_lora,
                resume_cls_progress_path=dual_cls_resume_path,
            )
            add_result("蒸馏学生只分类不解释", y_pred, tps, gb)
            print(f"  Precision={results[-1]['precision']}, Recall={results[-1]['recall']}, F1={results[-1]['f1_score']}, 单样本={tps:.3f}s, 显存={gb:.2f}GB")

    # 汇总表
    print()
    print("=" * 80)
    print("汇总表 (Precision / Recall / F1 / 单样本耗时s / 显存GB)")
    print("=" * 80)
    for r in results:
        print(
            f"  {r['model_name']:24s}  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1_score']:.4f}  "
            f"t={r['time_per_sample_s']:.4f}s  mem={r['gpu_peak_gb']:.2f}GB"
        )
    print("=" * 80)

    # 写入对比实验目录：各项数值 + 错误样本，文件名由运行参数决定，同参数不覆盖而是加 _1,_2,...
    base_name = _run_params_to_basename(args)
    save_path = _next_available_path(OUTPUT_DIR, base_name, ".json")
    run_params = {
        "data_path": str(args.data_path),
        "max_samples": args.max_samples,
        "run": args.run,
        "dual_full_light_explain": bool(getattr(args, "dual_full_light_explain", False)),
        "dual_full_max_new_tokens_explain": int(getattr(args, "dual_full_max_new_tokens_explain", 224)),
        "dual_full_max_seq_len_explain": int(getattr(args, "dual_full_max_seq_len_explain", 4096)),
    }
    export = {
        "run_params": run_params,
        "n_samples": n,
        "results": results,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"结果已写入（指标+错误样本）: {save_path}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        print(f"同时写入指定路径: {args.output}")


if __name__ == "__main__":
    main()
