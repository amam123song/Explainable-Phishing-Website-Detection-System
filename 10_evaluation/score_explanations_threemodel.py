#!/usr/bin/env python3
"""三模型 LLM-as-a-Judge 解释质量评测。仅供学术研究，禁止用于非法用途。"""

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EVAL_A = str(REPO_ROOT / "results" / "eval_dual_full.json")
DEFAULT_EVAL_B = str(REPO_ROOT / "results" / "eval_expert.json")
DEFAULT_EVAL_C = str(REPO_ROOT / "results" / "eval_monolithic_student.json")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "results" / "explanation_compare_out")

DEFAULT_API_BASE_URL = os.environ.get(
    "JUDGE_BASE_URL",
    os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
).rstrip("/") + "/chat/completions"
DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")


COMPARE_RUBRIC_PROMPT = r"""
你是资深的网络安全专家，负责评估钓鱼网站检测模型的解释质量。
你会同时看到同一个样本 index 的两段解释：
- A模型的解释
- B模型的解释
你还会看到网页的【证据包】（URL / 网页文本摘录 / 外链列表 / 真实标签）。
【重要评估原则（请务必遵守！）】
1. 允许外部知识：不要将模型利用其预训练知识（如：识别某域名属于哪家知名公司、识别某网页布局属于典型金融网站、知道某个社交媒体账号是验证账号）视为“幻觉”。只要该外部知识在现实中是合理的，且能辅助判断，应予以加分而非扣分！
2. 抓主要矛盾：不需要模型像机器人一样机械地穷举 URL、内容和外链。如果仅凭 URL 或页面内容的某几个致命特征（如明显的错别字仿冒、钓鱼表单）就足以得出结论，只要论证一针见血，即使没有提到外链，也应该给高分！
3. 惩罚真正的错误：只严厉惩罚“逻辑自相矛盾（如：说域名没问题，最后结论却是诈骗）”和“严重的提示词/训练日志泄露（如：输出 Epoch、Loss、System prompt）”。
你需要：
1) 对 A 和 B 分别独立打分（Rubric 总分 10）
2) 做 A vs B 的对比结论（winner: A/B/tie）
3) 给出简短理由
【Rubric】
- fact_and_knowledge (0-3分)：事实与知识的准确性。与证据包不冲突，且引用的外部知识合理即可得满分。只有严重的无中生有（例如编造证据包中完全相反的信息）才扣分。
- logic_clarity (0-3分)：逻辑清晰度。推理链条是否连贯，结论是否由论据自然推导得出。
- insightfulness (0-2分)：洞察力。是否一针见血地指出了钓鱼/合法的核心关键点，而不是机械流水账。
- professionalism (0-2分)：专业性与流畅度。术语使用准确，无模板残留、无训练日志泄露。
【错误类型 errors（可空）】
- prompt_leakage: 严重的模板残留 / 训练日志回放
- logic_errors: 结论与论据冲突 / 极其离谱的推理跳跃
- fatal_hallucination: 与安全判定直接相关的严重事实编造（合理的背景知识拓展不算此列）
输出必须是严格 JSON（不要 markdown，不要额外文本）：
{
  "sample_index": number,
  "A": {
    "scores": {"fact_and_knowledge":0-3,"logic_clarity":0-3,"insightfulness":0-2,"professionalism":0-2,"total":0-10},
    "errors": [{"category":"...","evidence_quote":"...","why":"..."}]
  },
  "B": {
    "scores": {"fact_and_knowledge":0-3,"logic_clarity":0-3,"insightfulness":0-2,"professionalism":0-2,"total":0-10},
    "errors": [{"category":"...","evidence_quote":"...","why":"..."}]
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
            "source": "candidate_A",
            "pred_label": exp_a.get("pred_label"),
            "generated_response": exp_a.get("generated_response", ""),
        },
        "B_explanation": {
            "source": "candidate_B",
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
    # 兼容新旧两版 rubric 字段：
    # 新版: fact_and_knowledge / insightfulness
    # 旧版: fact_consistency / completeness
    fact_and_knowledge = int(
        scores.get("fact_and_knowledge", scores.get("fact_consistency", 0))
    )
    insightfulness = int(scores.get("insightfulness", scores.get("completeness", 0)))
    return {
        "scores": {
            "fact_and_knowledge": fact_and_knowledge,
            "logic_clarity": int(scores.get("logic_clarity", 0)),
            "insightfulness": insightfulness,
            "professionalism": int(scores.get("professionalism", 0)),
            "total": int(scores.get("total", 0)),
        },
        "coverage_flags": block.get("coverage_flags", {"url": False, "content": False, "external_links": False}),
        "label_in_explanation": block.get("label_in_explanation"),
        "errors": block.get("errors", []),
    }


def _winner_from_totals(a_total: int, b_total: int) -> str:
    if a_total > b_total:
        return "A"
    if b_total > a_total:
        return "B"
    return "tie"


def _format_hms(seconds: float) -> str:
    sec = max(0, int(seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


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
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1,
        help="每处理多少条刷新一次进度（默认1=每条都显示）",
    )
    parser.add_argument(
        "--blind_swap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否对每条样本随机交换A/B输入顺序以减少位置偏差（默认开启）",
    )
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
    # 维度均分（用于 summary）
    scores_a_fact_knowledge: List[int] = []
    scores_a_logic: List[int] = []
    scores_a_insight: List[int] = []
    scores_a_prof: List[int] = []
    scores_b_fact_knowledge: List[int] = []
    scores_b_logic: List[int] = []
    scores_b_insight: List[int] = []
    scores_b_prof: List[int] = []
    wins = {"A": 0, "B": 0, "tie": 0}
    t0 = time.time()
    total_n = len(selected)

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for i, idx in enumerate(selected, 1):
            exp_a_orig = a_by_idx[idx]
            exp_b_orig = b_by_idx[idx]

            # 盲评：随机交换给评审器的 A/B 位置，避免固定顺序偏置
            swap_ab = False
            if args.blind_swap:
                swap_ab = random.Random(args.seed + idx).random() < 0.5
            exp_a = exp_b_orig if swap_ab else exp_a_orig
            exp_b = exp_a_orig if swap_ab else exp_b_orig

            cache_key = _hash_key(
                f"{idx}|swap={int(swap_ab)}|{exp_a.get('generated_response','')}|{exp_b.get('generated_response','')}"
            )
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

            # 先读取“盲位”A/B评分
            block_a_blind = normalize_score_block(llm_ret.get("A", {}))
            block_b_blind = normalize_score_block(llm_ret.get("B", {}))

            # 再映射回“原始输入”A(=eval_a) / B(=eval_b)
            if swap_ab:
                block_a = block_b_blind
                block_b = block_a_blind
            else:
                block_a = block_a_blind
                block_b = block_b_blind

            # winner 统一按总分推导，保证与分数一致
            winner = _winner_from_totals(
                block_a["scores"]["total"], block_b["scores"]["total"]
            )
            wins[winner] += 1
            scores_a.append(block_a["scores"]["total"])
            scores_b.append(block_b["scores"]["total"])
            scores_a_fact_knowledge.append(block_a["scores"]["fact_and_knowledge"])
            scores_a_logic.append(block_a["scores"]["logic_clarity"])
            scores_a_insight.append(block_a["scores"]["insightfulness"])
            scores_a_prof.append(block_a["scores"]["professionalism"])
            scores_b_fact_knowledge.append(block_b["scores"]["fact_and_knowledge"])
            scores_b_logic.append(block_b["scores"]["logic_clarity"])
            scores_b_insight.append(block_b["scores"]["insightfulness"])
            scores_b_prof.append(block_b["scores"]["professionalism"])

            rec = {
                "cache_key": cache_key,
                "index": idx,
                "url": exp_a_orig.get("url"),
                "true_label": exp_a_orig.get("true_label"),
                "A_source": "eval_a",
                "B_source": "eval_b",
                "blind_swap_applied": bool(swap_ab),
                "A_pred_label": exp_a_orig.get("pred_label"),
                "B_pred_label": exp_b_orig.get("pred_label"),
                "provider": "gpt-compatible",
                "model": args.model,
                "base_url": args.base_url,
                "A": block_a,
                "B": block_b,
                "comparison": {
                    "winner": winner,
                    "reason": llm_ret.get("comparison", {}).get("reason", ""),
                },
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            # 单行实时进度：百分比 + 已完成 + 当前index + ETA
            if args.progress_every > 0 and (i % args.progress_every == 0 or i == total_n):
                elapsed = time.time() - t0
                per_item = elapsed / i if i else 0.0
                eta = per_item * (total_n - i)
                pct = (i / total_n * 100.0) if total_n else 100.0
                msg = (
                    f"\r进度 {i}/{total_n} ({pct:6.2f}%) | "
                    f"当前index={idx} | "
                    f"已耗时={_format_hms(elapsed)} | "
                    f"预计剩余={_format_hms(eta)}"
                )
                sys.stdout.write(msg)
                sys.stdout.flush()
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
    if total_n > 0:
        print()

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
        "blind_swap": bool(args.blind_swap),
        "avg_total_A_eval_a": _avg(scores_a),
        "avg_total_B_eval_b": _avg(scores_b),
        "avg_fact_and_knowledge_A_eval_a": _avg(scores_a_fact_knowledge),
        "avg_fact_and_knowledge_B_eval_b": _avg(scores_b_fact_knowledge),
        "avg_logic_clarity_A_eval_a": _avg(scores_a_logic),
        "avg_logic_clarity_B_eval_b": _avg(scores_b_logic),
        "avg_insightfulness_A_eval_a": _avg(scores_a_insight),
        "avg_insightfulness_B_eval_b": _avg(scores_b_insight),
        "avg_professionalism_A_eval_a": _avg(scores_a_prof),
        "avg_professionalism_B_eval_b": _avg(scores_b_prof),
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

