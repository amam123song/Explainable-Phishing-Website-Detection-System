#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""黑盒启发式对抗样本生成（仅 scam 扰动）。仅供学术防御研究，禁止用于非法用途。"""

import argparse
import json
import os
import random
import re
from urllib.parse import urlparse, urlunparse


def is_parseable_url(url: str) -> bool:
    """基于 urlparse 判断 URL 是否可解析且具备非空主机名（netloc）。"""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if not url or any(c.isspace() for c in url):
        return False
    normalized = url if "://" in url else f"http://{url}"
    try:
        parsed = urlparse(normalized)
    except ValueError:
        return False
    netloc = (parsed.netloc or "").strip()
    if not netloc:
        return False
    host = netloc.split("@")[-1].split(":")[0].strip()
    if not host or any(c.isspace() for c in host):
        return False
    if parsed.scheme and parsed.scheme.lower() not in ("http", "https", ""):
        return False
    return True


def mutate_url_deformation(url: str) -> str:
    """URL 变形：长路径填充、子域前缀或路径字母的十六进制编码。变异后须仍可通过 urlparse 校验。"""
    if not is_parseable_url(url):
        return url

    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc
    path = parsed.path

    choice = random.choice(["overflow", "subdomain", "hex"])

    if choice == "overflow":
        safe_pad = ("www-google-com-secure-auth-login-session-valid-") * 3
        new_path = f"/{safe_pad}{path}"
        candidate = urlunparse(parsed._replace(path=new_path))
    elif choice == "subdomain":
        fake_brands = ["paypal-secure", "apple-auth", "amazon-billing"]
        new_host = f"{random.choice(fake_brands)}.{host}"
        candidate = urlunparse(parsed._replace(netloc=new_host))
    else:
        safe_path = path if path else "/"
        encoded_path = "".join([f"%{ord(c):02X}" if c.isalpha() else c for c in safe_path])
        candidate = urlunparse(parsed._replace(path=encoded_path))
        if candidate == url:
            safe_pad = ("www-google-com-secure-auth-login-session-valid-") * 3
            new_path = f"/{safe_pad}{safe_path}"
            candidate = urlunparse(parsed._replace(path=new_path))

    return candidate if is_parseable_url(candidate) else url

def mutate_content_perturbation(content: str) -> str:
    """正文扰动：规则同义词替换与可选 HTML 注释插入。"""
    if not content: return content

    synonyms = {
        r'\blogin\b': 'sign-in',
        r'\bpassword\b': 'passcode',
        r'\baccount\b': 'profile',
        r'\bverify\b': 'authenticate',
        r'\bsuspended\b': 'on hold'
    }

    mutated_content = content
    for word, replacement in synonyms.items():
        mutated_content = re.sub(word, replacement, mutated_content, flags=re.IGNORECASE)

    if random.random() > 0.5:
        mutated_content = mutated_content.replace("sign-in", "sign<!-- tracker -->-in")

    if mutated_content == content:
        insert_at = min(50, len(mutated_content))
        mutated_content = mutated_content[:insert_at] + "\u200b" + mutated_content[insert_at:]

    return mutated_content

def mutate_visual_confusion(content: str, url: str) -> tuple:
    """西里尔同形字符替换 URL 与正文；若无替换则追加参数或插入零宽字符。"""
    homoglyphs = {'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с'}

    mutated_url = url
    mutated_content = content

    for en_char, cyrillic_char in homoglyphs.items():
        mutated_url = mutated_url.replace(en_char, cyrillic_char, 2)
        mutated_content = mutated_content.replace(en_char, cyrillic_char, 5)

    if mutated_url == url:
        sep = "&" if ("?" in url) else "?"
        mutated_url = f"{url}{sep}ref=а"
    if mutated_content == content and content:
        insert_at = min(30, len(mutated_content))
        mutated_content = mutated_content[:insert_at] + "\u200b" + mutated_content[insert_at:]

    return mutated_content, mutated_url


_RE_URL = re.compile(r"(## URL:\n)(.*?)(\n## Content:\n)", re.DOTALL)
_RE_CONTENT = re.compile(r"(\n## Content:\n)(.*?)(\n## External Links:\n)", re.DOTALL)
_RE_LABEL = re.compile(r"\bLabel:\s*(scam|legit)\b", re.IGNORECASE)


def _get_label(sample: dict) -> str | None:
    """读取标签：支持 label 字段、数值 0/1，或 output 中的 Label 行。"""
    label = sample.get("label")
    if isinstance(label, str):
        low = label.strip().lower()
        if low in ("scam", "legit"):
            return low
    if isinstance(label, (int, float)):
        return "scam" if int(label) == 1 else "legit"

    out = sample.get("output")
    if isinstance(out, str):
        m = _RE_LABEL.search(out)
        if m:
            return m.group(1).lower()
    return None


def _extract_url_content(sample: dict) -> tuple[str, str]:
    """从 input 解析 URL 与 Content；失败则退回 url、content 字段。"""
    text = sample.get("input")
    if isinstance(text, str):
        m_url = _RE_URL.search(text)
        m_c = _RE_CONTENT.search(text)
        if m_url and m_c:
            url = m_url.group(2).strip()
            content = m_c.group(2)
            return url, content

    return sample.get("url", "") or "", sample.get("content", "") or ""


def _write_back_url_content(sample: dict, new_url: str, new_content: str) -> dict:
    """写回变异结果：有结构化 input 时仅替换 URL/Content 段，否则写 url、content。"""
    out = sample.copy()
    text = out.get("input")
    if isinstance(text, str) and _RE_URL.search(text) and _RE_CONTENT.search(text):
        text = _RE_URL.sub(rf"\1{new_url}\3", text, count=1)
        text = _RE_CONTENT.sub(lambda m: f"{m.group(1)}{new_content}{m.group(3)}", text, count=1)
        out["input"] = text
    else:
        out["url"] = new_url
        out["content"] = new_content
    return out


def generate_blackbox_dataset(
    input_json: str,
    output_json: str,
    *,
    limit: int = 2100,
    seed: int = 42,
):
    """
    取前 limit 条；仅 scam 扰动；legit 不变。
    scam 类型比例：URL 40%，内容 40%，视觉 10%，组合 10%。
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = list(data)[:limit]

    rng = random.Random(seed)
    random.seed(seed)
    scam_indices = [i for i, d in enumerate(data) if _get_label(d) == "scam"]
    rng.shuffle(scam_indices)

    n_scam = len(scam_indices)
    n_url = int(n_scam * 0.40)
    n_content = int(n_scam * 0.40)
    n_visual = int(n_scam * 0.10)
    n_combo = n_scam - n_url - n_content - n_visual

    url_set = set(scam_indices[:n_url])
    content_set = set(scam_indices[n_url : n_url + n_content])
    visual_set = set(scam_indices[n_url + n_content : n_url + n_content + n_visual])
    combo_set = set(scam_indices[n_url + n_content + n_visual :])

    adv_dataset: list[dict] = []
    stats = {"scam": 0, "legit": 0, "URL变形": 0, "内容扰动": 0, "视觉混淆": 0, "组合攻击": 0}

    for idx, sample in enumerate(data):
        label = _get_label(sample)
        if label != "scam":
            adv_dataset.append(sample)
            stats["legit"] += 1 if label == "legit" else 0
            continue

        stats["scam"] += 1
        orig_url, orig_content = _extract_url_content(sample)

        if idx in url_set:
            new_url = mutate_url_deformation(orig_url)
            new_content = orig_content
            attack_type = "URL变形"
        elif idx in content_set:
            new_url = orig_url
            new_content = mutate_content_perturbation(orig_content)
            attack_type = "内容扰动"
        elif idx in visual_set:
            new_content, new_url = mutate_visual_confusion(orig_content, orig_url)
            attack_type = "视觉混淆"
        elif idx in combo_set:
            step1_url = mutate_url_deformation(orig_url)
            step2_content, step2_url = mutate_visual_confusion(
                mutate_content_perturbation(orig_content),
                step1_url,
            )
            new_url = step2_url
            new_content = step2_content
            attack_type = "组合攻击"
        else:
            new_url = orig_url
            new_content = orig_content
            attack_type = "未分配"

        if new_url != orig_url and not is_parseable_url(new_url):
            new_url = orig_url
        adv_sample = _write_back_url_content(sample, new_url, new_content)
        adv_sample["attack_type"] = attack_type
        adv_dataset.append(adv_sample)
        if attack_type in stats:
            stats[attack_type] += 1

    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(adv_dataset, f, ensure_ascii=False, indent=2)

    print(
        f"已基于前 {len(data)} 条样本生成对抗集并保存至 {output_json}\n"
        f"- scam: {stats['scam']}（URL变形 {stats['URL变形']} / 内容扰动 {stats['内容扰动']} / 视觉混淆 {stats['视觉混淆']} / 组合攻击 {stats['组合攻击']}）\n"
        f"- legit(原样): {stats['legit']}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="黑盒启发式对抗样本生成（仅 scam 扰动，须显式指定输入/输出路径）",
    )
    parser.add_argument("--input_json", required=True, help="输入测试集 JSON（list，含 input/output）")
    parser.add_argument("--output_json", required=True, help="输出对抗集路径（建议 results/ 下，勿提交仓库）")
    parser.add_argument("--limit", type=int, default=2100, help="仅处理前 N 条样本")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    generate_blackbox_dataset(
        args.input_json,
        args.output_json,
        limit=args.limit,
        seed=args.seed,
    )
