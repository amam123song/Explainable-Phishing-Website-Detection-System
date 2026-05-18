"""URLNet 白盒 URL 扰动。仅供学术防御研究，禁止用于非法用途。"""

import argparse
import json
import os
import re
import sys
import urllib.parse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

CLOUD_SSO_PREFIXES = [
    "login.microsoftonline.com.common.oauth2.authorize.client.application.identity.provider.active.directory.azure.cloud.services.tenant.management.portal.security.compliance.center.documentation.resources.support.api",
    "accounts.google.com.o.oauth2.v2.auth.identifier.service.developer.console.cloud.infrastructure.network.management.dashboard.enterprise.workspace.admin.policy.privacy.terms.of.service.user.authentication.gateway",
    "console.aws.amazon.com.cloud.compute.storage.database.analytics.machine.learning.artificial.intelligence.internet.of.things.mobile.development.management.tools.developer.services.corporate.business.solutions",
    "developer.apple.com.documentation.swift.programming.language.frameworks.libraries.api.reference.tutorials.sample.code.xcode.ide.continuous.integration.delivery.app.store.connect.guidelines.design.human.interface"
]

def urlnet_whitebox_attack(
    url: str,
    model: URLNetCharCNN,
    tokenizer: CharTokenizer,
    device: str,
    max_len: int = 200
):
    model.eval()
    def get_pred_and_obj(text: str):
        ids = tokenizer.encode(text).to(device)
        with torch.no_grad():
            logits = model(ids.unsqueeze(0))
            pred = int(torch.argmax(logits, dim=1)[0].item())
            obj = (logits[0, 1] - logits[0, 0]).item()
        return pred, obj
    pred, _ = get_pred_and_obj(url)
    if pred == 0:
        return url, True
    try:
        parsed = urllib.parse.urlparse(url)
        scheme = parsed.scheme if parsed.scheme else "http"
        original_netloc = parsed.netloc

        if "@" in original_netloc:
            original_netloc = original_netloc.split("@")[-1]
        for base_prefix in CLOUD_SSO_PREFIXES:
            prefix = base_prefix.replace('-', '.')
            payload = prefix[:max_len + 5].strip('.')
            spoof_netloc = f"{payload}.{original_netloc}"

            candidate_url = urllib.parse.urlunparse((
                scheme, spoof_netloc, parsed.path,
                parsed.params, parsed.query, parsed.fragment
            ))

            c_pred, _ = get_pred_and_obj(candidate_url)

            if c_pred == 0:
                return candidate_url, True
    except Exception:
        pass
    return url, False

def extract_url_from_input(sample_input: str) -> str:
    m = re.search(r"## URL:\n(.*?)\n", sample_input, flags=re.IGNORECASE | re.DOTALL)
    if not m: return ""
    return m.group(1).strip()

def extract_label_from_output(sample_output: str) -> str:
    m = re.search(r"Label:\s*(scam|legit)", sample_output, flags=re.IGNORECASE)
    if not m: return ""
    return m.group(1).lower()

def replace_url_in_input(sample_input: str, new_url: str) -> str:
    return re.sub(
        r"(## URL:\n)(.*?)(\n)",
        lambda m: f"{m.group(1)}{new_url}{m.group(3)}",
        sample_input,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="URLNet 白盒 URL 对抗生成（须显式指定输入/输出与 checkpoint）",
    )
    parser.add_argument("--input_json", required=True, help="输入测试集 JSON")
    parser.add_argument("--output_json", required=True, help="输出对抗集路径（建议 results/ 下）")
    parser.add_argument("--checkpoint_path", required=True, help="URLNet 权重 .pth 路径")
    parser.add_argument(
        "--urlnet_repo_dir",
        type=Path,
        default=REPO_ROOT / "09_baselines" / "urlnet",
        help="含 urlnet_charcnn.py 的目录（默认本仓库 09_baselines/urlnet）",
    )
    parser.add_argument("--max_len", type=int, default=200, help="URL 字符序列最大长度")
    args = parser.parse_args()

    urlnet_dir = args.urlnet_repo_dir.resolve()
    if not urlnet_dir.is_dir():
        raise FileNotFoundError(f"URLNet 代码目录不存在: {urlnet_dir}")
    if str(urlnet_dir) not in sys.path:
        sys.path.insert(0, str(urlnet_dir))

    from urlnet_charcnn import CharTokenizer, CharVocab, URLNetCharCNN  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = CharVocab.build_default()
    tokenizer = CharTokenizer(vocab=vocab, max_len=args.max_len)
    model = URLNetCharCNN(
        vocab_size=vocab.size, embed_dim=32, num_classes=2,
        kernel_sizes=(3, 4, 5, 6), out_channels=128,
        dropout_p=0.5, pad_id=vocab.pad_id,
    ).to(device)

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=True)
    model.eval()

    with open(args.input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("开始处理数据集", flush=True)

    out_records, scam_total, scam_success, legit_total = [], 0, 0, 0

    for idx, sample in enumerate(dataset):
        url = extract_url_from_input(sample.get("input", ""))
        label = extract_label_from_output(sample.get("output", ""))

        if not url or not label:
            out_records.append(sample)
            continue
        if label == "legit":
            legit_total += 1
            out_records.append(sample)
            continue
        if label != "scam":
            out_records.append(sample)
            continue

        scam_total += 1

        adv_url, is_success = urlnet_whitebox_attack(
            url, model, tokenizer, device=device, max_len=args.max_len
        )
        if is_success: scam_success += 1

        new_sample = dict(sample)
        new_sample["input"] = replace_url_in_input(sample.get("input", ""), adv_url if is_success else url)
        new_sample["original_url"] = url
        new_sample["adv_url"] = adv_url
        new_sample["is_success"] = bool(is_success)
        out_records.append(new_sample)

        if (idx + 1) % 10 == 0:
            print(f"进度: {idx+1}/{len(dataset)} | scam={scam_total} success={scam_success} legit={legit_total}", flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    asr = (scam_success/scam_total*100) if scam_total else 0
    print(f"\n完成。ASR: {scam_success}/{scam_total} ({asr:.2f}%)")
