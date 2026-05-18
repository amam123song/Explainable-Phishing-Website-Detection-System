#!/usr/bin/env python3
"""GPT-4o-mini API 基线评测。仅供学术防御研究，禁止用于非法用途。"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.paths import DATASET_TEST, OPENAI_BASE_URL, RESULTS_DIR  # noqa: E402

OUTPUT_DIR = RESULTS_DIR / "gpt4o_mini"
DEFAULT_DATA_PATH = DATASET_TEST
DEFAULT_API_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    OPENAI_BASE_URL.rstrip("/") + "/chat/completions",
)
DEFAULT_MODEL = os.environ.get("GPT4O_MINI_MODEL", "gpt-4o-mini")


def _next_available_path(dir_path: Path, base_name: str, ext: str = ".json") -> Path:
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


def _params_to_basename(args: argparse.Namespace) -> str:
    max_s = getattr(args, "max_samples", None)
    max_str = str(max_s) if max_s is not None else "full"
    data_stem = args.data_path.stem
    return f"run_gpt4o-mini_samples_{max_str}_data_{data_stem}"


def load_test_data(
    data_path: Path,
    max_samples: Optional[int],
    sample_mode: str,
    seed: int,
) -> Tuple[List[dict], List[int], List[int]]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    indices = list(range(len(data)))
    if max_samples is not None:
        k = min(max_samples, len(indices))
        if sample_mode == "head":
            indices = indices[:k]
        else:
            rnd = random.Random(seed)
            indices = sorted(rnd.sample(indices, k))

    selected = [data[i] for i in indices]
    y_true = []
    for item in selected:
        out = (item.get("output") or "").strip().lower()
        if "scam" in out.split("\n")[0]:
            y_true.append(1)
        else:
            y_true.append(0)
    return selected, y_true, indices


def extract_url(input_text: str) -> str:
    m = re.search(r"## URL:\s*\n(.+?)(?:\n|$)", input_text, re.DOTALL)
    return m.group(1).strip() if m else ""


def format_prompt_fast(sample_input: str, input_char_limit: int = 8000, *, max_output_chars: int = 1200) -> str:
    if len(sample_input) > input_char_limit:
        sample_input = sample_input[: int(input_char_limit * 0.8)] + "\n...[Content Truncated]..."
    clean = sample_input.replace("# Pred:", "").strip()
    if "# Information:" not in clean:
        clean = f"# Information:\n{clean}"
    max_chars = int(max_output_chars) if max_output_chars and int(max_output_chars) > 0 else 1200
    return (
        f"{clean}\n\n# Pred:\n"
        "Please output exactly in this format:\n"
        "Label: scam OR Label: legit\n\n"
        f"Hard length limit: The entire output MUST be <= {max_chars} characters.\n"
        "Do NOT include any other sections, headings, or extra text.\n\n"
        "## Reason:\n"
    )


def extract_label(text: str) -> int:
    s = (text or "").strip().lower()
    if not s:
        return 0
    s = s.replace("诈骗", "scam").replace("正常", "legit").replace("安全", "legit")
    m = re.search(r"(?:label|pred|prediction)\s*[:：]\s*(scam|legit|legitimate)\b", s)
    if m:
        return 1 if m.group(1) == "scam" else 0
    first = s.splitlines()[0].strip() if s.splitlines() else s
    if re.search(r"\blegitimate\b", first):
        return 0
    if re.search(r"\bscam\b", first):
        return 1
    if re.search(r"\blegit\b", first):
        return 0
    if re.search(r"\b(phishing|credential theft|malicious)\b", s):
        return 1
    if re.search(r"\b(legitimate|benign|safe)\b", s):
        return 0
    if re.search(r"\bscam\b", s):
        return 1
    if re.search(r"\blegit\b", s):
        return 0
    return 0


def clean_explanation(text: str, max_len: int = 1200, min_len: int = 50) -> str:
    if not text or not text.strip():
        return text
    original = text
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"\n?## References:\s*[\s\S]*?(?=\n\n## |\n\n# |\Z)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"Please note that[^\n]*(?=\n|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"I do not claim any responsibility[^\n]*(?=\n|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Disclaimer[^\n]*(?=\n|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."
    if len(text) < min_len and len(original) > min_len:
        fallback = re.sub(r"```[\s\S]*?```", "", original).strip()
        if len(fallback) > max_len:
            fallback = fallback[:max_len].rstrip() + "..."
        if len(fallback) >= min_len:
            text = fallback
    return text


def _safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def call_openai_compatible_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 600,
    timeout_s: int = 120,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(base_url, headers=headers, json=body, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _coerce_generated_to_expert_style(text: str, *, max_explanation_chars: int = 1200) -> str:
    """补全 Reason 标题并调用 clean_explanation 截断与清洗。"""
    t = (text or "").strip()
    if not t:
        return t
    if "## reason" not in t.lower():
        t = "## Reason:\n" + t
    max_len = int(max_explanation_chars) if max_explanation_chars and int(max_explanation_chars) > 0 else 1200
    return clean_explanation(t, max_len=max_len)


def infer_one(
    sample_input: str,
    base_url: str,
    api_key: str,
    model: str,
    input_char_limit: int,
    max_explanation_chars: int,
    request_timeout_s: int = 45,
    request_max_tokens: int = 320,
    retries: int = 3,
) -> str:
    prompt = format_prompt_fast(
        sample_input,
        input_char_limit=input_char_limit,
        max_output_chars=max_explanation_chars,
    )
    max_chars = int(max_explanation_chars) if max_explanation_chars and int(max_explanation_chars) > 0 else 1200
    messages = [
        {
            "role": "system",
            "content": (
                "You are a phishing detection expert.\n"
                "Follow the required output format EXACTLY.\n"
                f"Hard length limit: Your entire output MUST be <= {max_chars} characters.\n"
                "If you cannot fit, shorten the reason while keeping the label correct.\n"
                "Output plain text only (no markdown fences)."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ret = call_openai_compatible_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=request_max_tokens,
                timeout_s=request_timeout_s,
            )
            return _coerce_generated_to_expert_style(ret, max_explanation_chars=max_explanation_chars)
        except Exception as e:
            last_err = e
            time.sleep(attempt)
    raise RuntimeError(f"调用失败: {last_err}")


def compute_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def build_bad_cases(data: List[dict], y_true: List[int], y_pred: List[int], source_indices: List[int]) -> List[dict]:
    bad = []
    for i, (true_l, pred_l) in enumerate(zip(y_true, y_pred)):
        if true_l != pred_l:
            inp = data[i].get("input", "")
            bad.append(
                {
                    "index": source_indices[i],
                    "url": extract_url(inp),
                    "true_label": "scam" if true_l == 1 else "legit",
                    "pred_label": "scam" if pred_l == 1 else "legit",
                }
            )
    return bad


def _read_existing_records_jsonl(jsonl_path: Path) -> Dict[int, Dict[str, Any]]:
    """按数据集 index 读取 jsonl 记录；解析失败的行跳过。"""
    existing: Dict[int, Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return existing
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _safe_json_loads(line)
            if not isinstance(obj, dict):
                continue
            idx = obj.get("index")
            try:
                idx_int = int(idx)
            except Exception:
                continue
            existing[idx_int] = obj
    return existing


def _append_record_jsonl(jsonl_path: Path, record: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="商用 gpt-4o-mini 钓鱼检测评测脚本")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="测试集 JSON 路径")
    parser.add_argument("--max-samples", type=int, default=50, help="最多评测样本数")
    parser.add_argument("--sample-mode", choices=["head", "random"], default="head", help="采样模式：head 或 random")
    parser.add_argument("--seed", type=int, default=42, help="随机采样种子")
    parser.add_argument("--base-url", type=str, default=DEFAULT_API_BASE_URL, help="OpenAI 兼容 chat completions 地址")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""), help="API Key，默认取环境变量 OPENAI_API_KEY")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="商用模型名")
    parser.add_argument("--input-char-limit", type=int, default=8000, help="输入截断字符数")
    parser.add_argument(
        "--max-explanation-chars",
        type=int,
        default=1200,
        help="生成解释的最大保留字符数（默认 1200，不影响标签解析）",
    )
    parser.add_argument("--sleep-s", type=float, default=0.0, help="每条样本之间额外 sleep 秒数")
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="输出文件名前缀；为空则按参数自动生成",
    )
    parser.add_argument("--output", type=Path, default=None, help="可选，额外写入的 json 路径")
    parser.add_argument(
        "--save-explanations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否保存每条样本的模型完整输出",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传：已存在 jsonl 时跳过已完成索引并追加",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发线程数，默认 1",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=int,
        default=45,
        help="单次 API 请求超时（秒），默认 45",
    )
    parser.add_argument(
        "--request-max-tokens",
        type=int,
        default=320,
        help="API max_tokens，默认 320",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=2,
        help="每个样本的请求重试次数，默认 2",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise SystemExit(f"测试集不存在: {args.data_path}")
    if not args.api_key:
        raise SystemExit("缺少 API Key。请通过 --api-key 传入，或设置环境变量 OPENAI_API_KEY。")

    data, y_true, selected_indices = load_test_data(
        data_path=args.data_path,
        max_samples=args.max_samples,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    n = len(data)
    if n == 0:
        raise SystemExit("没有可评测样本。")

    run_tag = (args.run_tag or "").strip() or _params_to_basename(args)
    default_json_path = OUTPUT_DIR / f"{run_tag}.json"
    default_jsonl_path = OUTPUT_DIR / f"{run_tag}.records.jsonl"

    if default_json_path.exists() and not args.resume and not args.run_tag:
        base_name = _params_to_basename(args)
        default_json_path = _next_available_path(OUTPUT_DIR, base_name, ".json")
        run_tag = default_json_path.stem
        default_jsonl_path = OUTPUT_DIR / f"{run_tag}.records.jsonl"

    existing = _read_existing_records_jsonl(default_jsonl_path) if args.resume else {}
    done_set = set(existing.keys())

    print(
        f"测试集: {args.data_path.name}, 样本数: {n}, 采样模式: {args.sample_mode}\n"
        f"输出: {default_json_path}\n"
        f"逐条记录: {default_jsonl_path}\n"
        f"断点续传: {'开启' if args.resume else '关闭'}（已完成 {len(done_set)}/{n}）"
    )

    y_pred: List[int] = [-1 for _ in range(n)]
    generated: List[str] = ["" for _ in range(n)]
    t0 = time.perf_counter()
    for local_i, item in enumerate(data, 0):
        dataset_idx = selected_indices[local_i]
        if dataset_idx not in done_set:
            continue
        rec = existing[dataset_idx]
        text = str(rec.get("generated_response") or "")
        pred_label = str(rec.get("pred_label") or "").strip().lower()
        pred = 1 if pred_label == "scam" else 0
        y_pred[local_i] = pred
        generated[local_i] = text

    pending_local_indices = [
        local_i for local_i in range(n) if selected_indices[local_i] not in done_set
    ]

    workers = max(1, int(args.workers))
    if workers == 1:
        pbar = tqdm(total=n, desc=f"{args.model}")
        pbar.update(len(done_set))
        for local_i in pending_local_indices:
            item = data[local_i]
            dataset_idx = selected_indices[local_i]
            text = infer_one(
                sample_input=item.get("input", ""),
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                input_char_limit=args.input_char_limit,
                max_explanation_chars=args.max_explanation_chars,
                request_timeout_s=args.request_timeout_s,
                request_max_tokens=args.request_max_tokens,
                retries=args.request_retries,
            )
            pred = extract_label(text)
            y_pred[local_i] = pred
            generated[local_i] = text

            true_label = "scam" if y_true[local_i] == 1 else "legit"
            pred_label = "scam" if pred == 1 else "legit"
            rec = {
                "index": dataset_idx,
                "url": extract_url(item.get("input", "")),
                "true_label": true_label,
                "pred_label": pred_label,
                "generated_response": text,
                "status": "ok",
            }
            _append_record_jsonl(default_jsonl_path, rec)
            done_set.add(dataset_idx)
            pbar.update(1)
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
        pbar.close()
    else:
        write_lock = threading.Lock()

        def _worker(local_i: int) -> Tuple[int, int, str, Dict[str, Any]]:
            item = data[local_i]
            dataset_idx = selected_indices[local_i]
            true_label = "scam" if y_true[local_i] == 1 else "legit"
            try:
                text = infer_one(
                    sample_input=item.get("input", ""),
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    input_char_limit=args.input_char_limit,
                    max_explanation_chars=args.max_explanation_chars,
                    request_timeout_s=args.request_timeout_s,
                    request_max_tokens=args.request_max_tokens,
                    retries=args.request_retries,
                )
                pred = extract_label(text)
                pred_label = "scam" if pred == 1 else "legit"
                rec = {
                    "index": dataset_idx,
                    "url": extract_url(item.get("input", "")),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "generated_response": text,
                    "status": "ok",
                }
                return local_i, pred, text, rec
            except Exception as e:
                text = f"Label: legit\n\n## Reason:\n[ERROR] request failed: {str(e)[:300]}"
                pred = 0
                rec = {
                    "index": dataset_idx,
                    "url": extract_url(item.get("input", "")),
                    "true_label": true_label,
                    "pred_label": "legit",
                    "generated_response": text,
                    "status": "error",
                    "error": str(e)[:500],
                }
                return local_i, pred, text, rec

        pbar = tqdm(total=n, desc=f"{args.model}(workers={workers})")
        pbar.update(len(done_set))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_worker, local_i): local_i for local_i in pending_local_indices}
            for fut in concurrent.futures.as_completed(fut_map):
                local_i, pred, text, rec = fut.result()
                y_pred[local_i] = pred
                generated[local_i] = text
                with write_lock:
                    _append_record_jsonl(default_jsonl_path, rec)
                    done_set.add(int(rec["index"]))
                pbar.update(1)
                if args.sleep_s > 0:
                    time.sleep(args.sleep_s)
        pbar.close()

    elapsed = time.perf_counter() - t0
    tps = elapsed / n

    missing = [i for i, v in enumerate(y_pred) if v == -1]
    if missing:
        raise RuntimeError(f"仍有未完成样本预测（可能是并发中断或异常）：count={len(missing)}, first={missing[:5]}")

    metrics = compute_metrics(y_true, y_pred)
    metrics["model_name"] = f"{args.model}"
    metrics["time_per_sample_s"] = round(tps, 4)
    metrics["gpu_peak_gb"] = 0.0
    metrics["bad_cases"] = build_bad_cases(data, y_true, y_pred, selected_indices)
    metrics["bad_cases_count"] = len(metrics["bad_cases"])
    if args.save_explanations and len(generated) == len(data):
        metrics["generated_explanations"] = [
            {
                "index": selected_indices[i],
                "url": extract_url(data[i].get("input", "")),
                "true_label": "scam" if y_true[i] == 1 else "legit",
                "pred_label": "scam" if y_pred[i] == 1 else "legit",
                "generated_response": generated[i],
            }
            for i in range(n)
        ]

    run_params: Dict[str, Any] = {
        "data_path": str(args.data_path),
        "max_samples": args.max_samples,
        "run": "gpt-4o-mini",
        "base_url": args.base_url,
        "model": args.model,
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "input_char_limit": args.input_char_limit,
        "max_explanation_chars": args.max_explanation_chars,
        "request_timeout_s": args.request_timeout_s,
        "request_max_tokens": args.request_max_tokens,
        "request_retries": args.request_retries,
        "sleep_s": args.sleep_s,
        "resume": bool(args.resume),
        "save_explanations": bool(args.save_explanations),
    }
    export = {
        "run_params": run_params,
        "n_samples": n,
        "results": [metrics],
    }

    with open(default_json_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        print(f"同时写入指定路径: {args.output}")

    print(
        f"Precision={metrics['precision']}, Recall={metrics['recall']}, "
        f"F1={metrics['f1_score']}, Accuracy={metrics['accuracy']}, "
        f"time/sample={metrics['time_per_sample_s']}s"
    )
    print(f"结果已写入: {default_json_path}")


if __name__ == "__main__":
    main()
