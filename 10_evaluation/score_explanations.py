#!/usr/bin/env python3
"""LLM-as-a-Judge 解释质量背靠背评测。API Key 仅通过环境变量配置。仅供学术研究，禁止用于非法用途。"""

import argparse
import ast
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EVAL_A = str(REPO_ROOT / "results" / "eval_dual_full.json")
DEFAULT_EVAL_B = str(REPO_ROOT / "results" / "eval_expert.json")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "results" / "explanation_compare_out")

DEFAULT_API_BASE_URL = os.environ.get(
    "JUDGE_BASE_URL",
    os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
).rstrip("/") + "/chat/completions"
DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")


COMPARE_RUBRIC_PROMPT = r"""
你是严格的解释质量评估器。你会同时看到同一个样本 index 的两段解释：
- A: dualfull 的解释
- B: expert 的解释

你还会看到证据包 evidence（url / content_excerpt / external_links / true_label）。

你需要：
1) 对 A 和 B 分别独立打分（Rubric 总分 10）
2) 做 A vs B 的对比结论（winner: A/B/tie）
3) 给出简短理由

Rubric：
- fact_consistency 0-3
- logic_clarity 0-3
- completeness 0-2（是否覆盖 URL / Content / External Links）
- professionalism 0-2

错误类型 errors（可空）：
- prompt_leakage: template_residual / instruction_exposure / training_prompt_replay
- logic_errors: label_contradiction / reasoning_jump / causal_inversion
- hallucination_generation: fact_fabrication / feature_attribution / brand_domain_mismatch

输出必须是严格 JSON（不要 markdown，不要额外文本）：
{
  "sample_index": number,
  "A": {
    "scores": {"fact_consistency":0-3,"logic_clarity":0-3,"completeness":0-2,"professionalism":0-2,"total":0-10},
    "coverage_flags": {"url": true/false, "content": true/false, "external_links": true/false},
    "label_in_explanation": "scam" | "legit" | null,
    "errors": [{"category":"...","subtype":"...","severity":"low|medium|high","evidence_quote":"...","why":"..."}]
  },
  "B": {
    "scores": {"fact_consistency":0-3,"logic_clarity":0-3,"completeness":0-2,"professionalism":0-2,"total":0-10},
    "coverage_flags": {"url": true/false, "content": true/false, "external_links": true/false},
    "label_in_explanation": "scam" | "legit" | null,
    "errors": [{"category":"...","subtype":"...","severity":"low|medium|high","evidence_quote":"...","why":"..."}]
  },
  "comparison": {
    "winner": "A" | "B" | "tie",
    "reason": "一句话比较理由"
  }
}
"""


def build_compare_rubric_prompt(eval_lang: str) -> str:
    lang_rules = {
        "zh": (
            "语言要求：除 JSON 字段名外，所有自然语言内容（如 comparison.reason、errors[].why、"
            "errors[].evidence_quote 的解释性部分）必须使用简体中文。不要输出英文句子。"
        ),
        "en": (
            "Language requirement: Except JSON field names, all natural language content "
            "(e.g., comparison.reason, errors[].why, explanatory parts) must be in English only. "
            "Do not output Chinese text."
        ),
    }
    extra = lang_rules.get(eval_lang, lang_rules["zh"])
    return COMPARE_RUBRIC_PROMPT.strip() + "\n\n" + extra


@dataclass
class Evidence:
    url: str
    content_excerpt: str
    external_links: List[str]
    true_label: str


def _safe_json_loads(s: str) -> Dict[str, Any]:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_run_tag(sample_mode: str, sample_count: int, seed: int) -> str:
    return f"n{sample_count}_mode-{sample_mode}_seed-{seed}"


def next_unique_output_path(output_dir: str, base_name: str, ext: str) -> str:
    # 例如 base_name=pairwise_scores_n100_mode-random_seed-42
    # 生成：
    # pairwise_scores_n100_mode-random_seed-42_1.jsonl
    # pairwise_scores_n100_mode-random_seed-42_2.jsonl
    i = 1
    while True:
        p = os.path.join(output_dir, f"{base_name}_{i}.{ext}")
        if not os.path.exists(p):
            return p
        i += 1


def allocate_paired_run_paths(output_dir: str, run_tag: str) -> Tuple[str, str, int]:
    # 为 scores 与 summary 分配同一个序号，避免中断后两个文件编号错位
    scores_base = f"pairwise_scores_{run_tag}"
    summary_base = f"pairwise_summary_{run_tag}"
    pattern = re.compile(
        rf"^(?:{re.escape(scores_base)}|{re.escape(summary_base)})_(\d+)\.(?:jsonl|json)$"
    )

    max_idx = 0
    for name in os.listdir(output_dir):
        m = pattern.match(name)
        if not m:
            continue
        try:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
        except Exception:
            continue

    run_idx = max_idx + 1
    out_jsonl = os.path.join(output_dir, f"{scores_base}_{run_idx}.jsonl")
    out_summary = os.path.join(output_dir, f"{summary_base}_{run_idx}.json")
    return out_jsonl, out_summary, run_idx


def parse_dataset_input_block(input_text: str) -> Tuple[str, str, List[str]]:
    url_m = re.search(r"## URL:\s*(.+?)\n", input_text)
    if not url_m:
        raise ValueError("Cannot parse URL from dataset input")
    url = url_m.group(1).strip()

    marker_content = "## Content:"
    marker_links = "## External Links:"
    idx_content = input_text.find(marker_content)
    idx_links = input_text.find(marker_links)
    if idx_content < 0 or idx_links < 0 or idx_links <= idx_content:
        raise ValueError("Cannot parse Content/External Links markers from dataset input")

    content = input_text[idx_content + len(marker_content):idx_links].strip()
    idx_after_links = idx_links + len(marker_links)
    bracket_start = input_text.find("[", idx_after_links)
    if bracket_start < 0:
        return url, content, []

    depth = 0
    bracket_end = -1
    for i in range(bracket_start, len(input_text)):
        ch = input_text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                bracket_end = i + 1
                break
    if bracket_end < 0:
        return url, content, []

    links_text = input_text[bracket_start:bracket_end].strip()
    try:
        links = ast.literal_eval(links_text)
    except Exception:
        links_clean = links_text.strip("[]")
        links = [x.strip().strip("'").strip('"') for x in links_clean.split(",") if x.strip()]
    if not isinstance(links, list):
        links = list(links)
    return url, content, [str(x) for x in links]


def build_content_excerpt(content: str, a_resp: str, b_resp: str, max_chars: int = 6500) -> str:
    keywords = [
        "login", "password", "cookie", "privacy", "terms", "secure", "account", "verify",
        "urgent", "suspended", "redirect", "oauth", "sign in", "register",
    ]
    for resp in (a_resp, b_resp):
        for u in re.findall(r"https?://[^\s\"')\]]+", resp)[:5]:
            host = re.sub(r"^https?://", "", u).strip()
            if host:
                keywords.append(host)
    keywords = list(dict.fromkeys(keywords))

    chunks: List[str] = []
    for kw in keywords:
        idx = content.lower().find(kw.lower())
        if idx >= 0:
            left = max(0, idx - 280)
            right = min(len(content), idx + 820)
            chunk = content[left:right].strip()
            if chunk and chunk not in chunks:
                chunks.append(chunk)
        if len(chunks) >= 8:
            break

    if not chunks:
        chunks = [content[:2400].strip(), content[-2400:].strip() if len(content) > 2400 else ""]

    excerpt = "\n\n---\n\n".join([c for c in chunks if c])
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 30] + "\n...[TRUNCATED]..."
    return excerpt


def resolve_dataset_path(args_dataset_json: Optional[str], eval_a_data: Dict[str, Any]) -> str:
    if args_dataset_json:
        return args_dataset_json
    ds = eval_a_data.get("run_params", {}).get("data_path")
    if not ds:
        raise ValueError("无法从 eval A 的 run_params.data_path 推断 dataset 路径，请显式传 --dataset_json")
    return ds


def flatten_explanations(eval_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    results = eval_data.get("results", [])
    if not results:
        return {}
    ge = results[0].get("generated_explanations", [])
    out: Dict[int, Dict[str, Any]] = {}
    for item in ge:
        idx = item.get("index")
        if isinstance(idx, int):
            out[idx] = item
    return out


def call_openai_compatible_chat(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]]) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 1800}
    resp = requests.post(base_url, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def score_pair_with_llm(
    base_url: str,
    api_key: str,
    model: str,
    eval_lang: str,
    sample_index: int,
    evidence: Evidence,
    exp_a: Dict[str, Any],
    exp_b: Dict[str, Any],
) -> Dict[str, Any]:
    external_links = evidence.external_links[:50]
    if len(evidence.external_links) > 50:
        external_links.append("...[TRUNCATED]...")

    user_payload = {
        "sample_index": sample_index,
        "evidence": {
            "url": evidence.url,
            "true_label": evidence.true_label,
            "content_excerpt": evidence.content_excerpt,
            "external_links": external_links,
        },
        "A_explanation": {
            "source": "dualfull",
            "pred_label": exp_a.get("pred_label"),
            "generated_response": exp_a.get("generated_response", ""),
        },
        "B_explanation": {
            "source": "expert",
            "pred_label": exp_b.get("pred_label"),
            "generated_response": exp_b.get("generated_response", ""),
        },
    }
    messages = [
        {"role": "system", "content": build_compare_rubric_prompt(eval_lang)},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            raw = call_openai_compatible_chat(base_url, api_key, model, messages)
            return _safe_json_loads(raw)
        except Exception as e:
            last_err = e
            messages[-1]["content"] += "\n\n请只输出严格 JSON 对象，不要附加解释。"
            time.sleep(attempt)
    raise RuntimeError(f"LLM 调用失败: {last_err}")


def normalize_score_block(block: Dict[str, Any]) -> Dict[str, Any]:
    scores = block.get("scores", {})
    return {
        "scores": {
            "fact_consistency": int(scores.get("fact_consistency", 0)),
            "logic_clarity": int(scores.get("logic_clarity", 0)),
            "completeness": int(scores.get("completeness", 0)),
            "professionalism": int(scores.get("professionalism", 0)),
            "total": int(scores.get("total", 0)),
        },
        "coverage_flags": block.get("coverage_flags", {"url": False, "content": False, "external_links": False}),
        "label_in_explanation": block.get("label_in_explanation"),
        "errors": block.get("errors", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_a", default=DEFAULT_EVAL_A)
    parser.add_argument("--eval_b", default=DEFAULT_EVAL_B)
    parser.add_argument("--dataset_json", default=None, help="可选；默认从 eval_a.run_params.data_path 自动读取")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_count", type=int, default=20, help="评估多少个 index")
    parser.add_argument("--sample_mode", choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep_s", type=float, default=0.0)
    parser.add_argument("--eval_lang", choices=["zh", "en"], default="zh", help="强制评估输出语言：zh=中文，en=英文")
    parser.add_argument("--base_url", default=DEFAULT_API_BASE_URL)
    parser.add_argument(
        "--api_key",
        default=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        help="勿写入代码；使用环境变量 JUDGE_API_KEY 或 OPENAI_API_KEY",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "缺少 API Key。请设置环境变量 JUDGE_API_KEY 或 OPENAI_API_KEY，"
            "或通过 --api_key 临时传入（勿提交到版本库）。"
        )

    ensure_dir(args.output_dir)
    run_tag = build_run_tag(args.sample_mode, args.sample_count, args.seed)
    out_jsonl, out_summary, run_idx = allocate_paired_run_paths(args.output_dir, run_tag)

    eval_a_data = load_json(args.eval_a)
    eval_b_data = load_json(args.eval_b)
    dataset_path = resolve_dataset_path(args.dataset_json, eval_a_data)
    dataset_data = load_json(dataset_path)

    a_by_idx = flatten_explanations(eval_a_data)
    b_by_idx = flatten_explanations(eval_b_data)
    common_indices = sorted(set(a_by_idx.keys()) & set(b_by_idx.keys()))
    if not common_indices:
        raise SystemExit("两个解释文件没有共同 index，无法对比评估。")

    if args.sample_mode == "head":
        selected = common_indices[: args.sample_count]
    else:
        rnd = random.Random(args.seed)
        k = min(args.sample_count, len(common_indices))
        selected = sorted(rnd.sample(common_indices, k))

    # 新规则：每次运行都输出新文件，不复用旧文件，因此不做“读取已有 jsonl 断点跳过”
    processed: set = set()

    scores_a: List[int] = []
    scores_b: List[int] = []
    wins = {"A": 0, "B": 0, "tie": 0}

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for i, idx in enumerate(selected, 1):
            exp_a = a_by_idx[idx]
            exp_b = b_by_idx[idx]
            cache_key = _hash_key(f"{idx}|{exp_a.get('generated_response','')}|{exp_b.get('generated_response','')}")
            if cache_key in processed:
                print(f"[{i}/{len(selected)}] skip index={idx}")
                continue

            # 先按 index 对齐 dataset，失败再按 URL 匹配兜底
            ds_item = dataset_data[idx] if 0 <= idx < len(dataset_data) else None
            if not ds_item:
                ds_item = {"input": f"## URL:\n{exp_a.get('url')}\n## Content:\n\n## External Links:\n[]"}

            try:
                url, content, links = parse_dataset_input_block(ds_item["input"])
            except Exception:
                url = exp_a.get("url", "")
                content = ""
                links = []

            evidence = Evidence(
                url=url or exp_a.get("url", ""),
                content_excerpt=build_content_excerpt(content, exp_a.get("generated_response", ""), exp_b.get("generated_response", "")),
                external_links=links,
                true_label=str(exp_a.get("true_label", "")),
            )

            print(f"[{i}/{len(selected)}] scoring index={idx}")
            llm_ret = score_pair_with_llm(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                eval_lang=args.eval_lang,
                sample_index=idx,
                evidence=evidence,
                exp_a=exp_a,
                exp_b=exp_b,
            )

            block_a = normalize_score_block(llm_ret.get("A", {}))
            block_b = normalize_score_block(llm_ret.get("B", {}))
            winner = llm_ret.get("comparison", {}).get("winner", "tie")
            if winner not in wins:
                winner = "tie"
            wins[winner] += 1
            scores_a.append(block_a["scores"]["total"])
            scores_b.append(block_b["scores"]["total"])

            rec = {
                "cache_key": cache_key,
                "index": idx,
                "url": exp_a.get("url"),
                "true_label": exp_a.get("true_label"),
                "A_source": "dualfull",
                "B_source": "expert",
                "A_pred_label": exp_a.get("pred_label"),
                "B_pred_label": exp_b.get("pred_label"),
                "provider": "gpt-compatible",
                "model": args.model,
                "base_url": args.base_url,
                "A": block_a,
                "B": block_b,
                "comparison": llm_ret.get("comparison", {"winner": "tie", "reason": ""}),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

    def _avg(vals: List[int]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "eval_a": args.eval_a,
        "eval_b": args.eval_b,
        "dataset_json": dataset_path,
        "sample_mode": args.sample_mode,
        "sample_count_requested": args.sample_count,
        "sample_indices": selected,
        "model": args.model,
        "eval_lang": args.eval_lang,
        "base_url": args.base_url,
        "avg_total_A_dualfull": _avg(scores_a),
        "avg_total_B_expert": _avg(scores_b),
        "wins": wins,
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Run index: {run_idx}")
    print(f"Output jsonl: {out_jsonl}")
    print(f"Output summary: {out_summary}")


if __name__ == "__main__":
    main()

