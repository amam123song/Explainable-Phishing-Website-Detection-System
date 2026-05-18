"""DistilBERT 白盒稀疏词替换。仅供学术防御研究，禁止用于非法用途；路径均由命令行传入。"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def extract_label_from_output(sample_output: str) -> str:
    match = re.search(r"Label:\s*(scam|legit)", sample_output, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).lower()

def build_full_vocab_embeddings(model, tokenizer, device):
    """构建字母子词词表子集及其嵌入矩阵。"""
    valid_ids = []
    for token, idx in tokenizer.vocab.items():
        if token not in tokenizer.all_special_tokens:
            clean_token = token.replace("##", "")
            if clean_token.isalpha() and len(clean_token) >= 2:
                valid_ids.append(idx)

    valid_ids_tensor = torch.tensor(valid_ids, device=device)
    valid_embeddings = model.get_input_embeddings()(valid_ids_tensor).detach()
    return valid_ids, valid_embeddings

def is_mutable_token(token: str, tokenizer) -> bool:
    if token in tokenizer.all_special_tokens:
        return False
    return True

def exact_eval_attack(
    text: str,
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizer,
    valid_vocab_ids: List[int],
    valid_embeddings: torch.Tensor,
    *,
    max_steps_hard_limit: int = 30,
    top_k_words: int = 150,
    max_perturb_ratio: float = 0.05,
    device: torch.device,
) -> Tuple[str, bool, List[str]]:

    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    current_input_ids = inputs["input_ids"][0].clone()
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        orig_logits = model(input_ids=current_input_ids.unsqueeze(0), attention_mask=attention_mask).logits
        orig_margin = orig_logits[0][0] - orig_logits[0][1]
        if orig_margin.item() > 0:
            return text, True, ["Already Legit"]

    tokens = tokenizer.convert_ids_to_tokens(current_input_ids)
    valid_seq_len = sum(1 for t in tokens if t not in tokenizer.all_special_tokens)

    dynamic_max_steps = max(1, int(valid_seq_len * max_perturb_ratio))
    allowed_steps = min(dynamic_max_steps, max_steps_hard_limit)

    modifications = []
    modified_positions = set()
    V_valid = valid_embeddings.size(0)

    for step in range(allowed_steps):
        word_embeddings = model.get_input_embeddings()
        embeds = word_embeddings(current_input_ids.unsqueeze(0)).detach().clone()
        embeds.requires_grad = True

        outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
        logits = outputs.logits
        margin = logits[0][0] - logits[0][1]

        if margin.item() > 0:
            break

        model.zero_grad()
        margin.backward()

        grads = embeds.grad.squeeze(0)
        orig_embeds = embeds.squeeze(0).detach()

        term1 = torch.matmul(grads, valid_embeddings.t())
        term2 = torch.sum(grads * orig_embeds, dim=1, keepdim=True)
        gains = term1 - term2

        current_tokens = tokenizer.convert_ids_to_tokens(current_input_ids)
        seq_len = len(current_tokens)

        has_mutable = False
        for i in range(seq_len):
            if i in modified_positions or not is_mutable_token(current_tokens[i], tokenizer):
                gains[i, :] = -float('inf')
            else:
                has_mutable = True

        if not has_mutable:
            break

        flat_gains = gains.flatten()
        k = min(top_k_words, flat_gains.numel())
        topk = torch.topk(flat_gains, k)

        best_margin = margin.item()
        best_pos = -1
        best_cand_id = -1

        batch_size = 50
        for chunk_start in range(0, k, batch_size):
            chunk_indices = topk.indices[chunk_start:chunk_start+batch_size]
            actual_chunk_size = len(chunk_indices)

            temp_ids_batch = current_input_ids.unsqueeze(0).repeat(actual_chunk_size, 1)
            positions = []
            cand_ids = []

            for b_idx, flat_idx in enumerate(chunk_indices):
                if flat_gains[flat_idx] == -float('inf'):
                    continue
                pos = (flat_idx // V_valid).item()
                cand_idx = (flat_idx % V_valid).item()
                cand_id = valid_vocab_ids[cand_idx]

                temp_ids_batch[b_idx, pos] = cand_id
                positions.append(pos)
                cand_ids.append(cand_id)

            if not positions:
                continue

            with torch.no_grad():
                temp_logits = model(
                    input_ids=temp_ids_batch,
                    attention_mask=attention_mask.repeat(actual_chunk_size, 1)
                ).logits
                temp_margins = temp_logits[:, 0] - temp_logits[:, 1]

                max_val, max_idx = torch.max(temp_margins, dim=0)

                if max_val.item() > best_margin:
                    best_margin = max_val.item()
                    best_pos = positions[max_idx]
                    best_cand_id = cand_ids[max_idx]

        if best_pos != -1 and best_margin > margin.item():
            original_token = current_tokens[best_pos]
            new_token = tokenizer.convert_ids_to_tokens(best_cand_id)

            modifications.append(f"[{original_token}->{new_token}](M:{margin.item():.2f}→{best_margin:.2f})")
            current_input_ids[best_pos] = best_cand_id
            modified_positions.add(best_pos)
        else:
            break

    final_text = tokenizer.decode(current_input_ids, skip_special_tokens=True)
    with torch.no_grad():
        final_logits = model(input_ids=current_input_ids.unsqueeze(0), attention_mask=attention_mask).logits
        final_pred = torch.argmax(final_logits, dim=1).item()

    return final_text, bool(final_pred == 0), modifications


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--top_k_words", type=int, default=150)
    parser.add_argument("--max_perturb_ratio", type=float, default=0.05, help="相对序列长度的最大替换步数比例，默认 0.05")
    parser.add_argument("--start_exclusive", type=int, default=None)
    parser.add_argument("--end_inclusive", type=int, default=None)
    parser.add_argument("--first_n", type=int, default=None)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.start_exclusive is not None and args.end_inclusive is not None:
        subset = dataset[args.start_exclusive : args.end_inclusive]
    else:
        first_n = args.first_n if args.first_n is not None else 3000
        subset = dataset[:first_n]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载模型，max_perturb_ratio={args.max_perturb_ratio}", flush=True)
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir).to(device)

    valid_vocab_ids, valid_embeddings = build_full_vocab_embeddings(model, tokenizer, device)
    print(f"候选词表子集大小: {len(valid_vocab_ids)}\n", flush=True)

    out_records, scam_total, scam_success, legit_total = [], 0, 0, 0

    for idx, sample in enumerate(subset):
        sample_input = sample.get("input", "")
        label = extract_label_from_output(sample.get("output", ""))

        if label == "legit" or not label:
            if label == "legit": legit_total += 1
            out_records.append(sample)
            continue

        scam_total += 1
        adv_text, is_success, modifications = exact_eval_attack(
            sample_input, model, tokenizer, valid_vocab_ids, valid_embeddings,
            max_steps_hard_limit=args.max_steps, top_k_words=args.top_k_words,
            max_perturb_ratio=args.max_perturb_ratio, device=device
        )

        if is_success:
            scam_success += 1

        new_sample = dict(sample)
        new_sample["input"] = adv_text if is_success else sample_input
        new_sample["adv_text"] = adv_text
        new_sample["is_success"] = is_success
        new_sample["modifications"] = modifications
        out_records.append(new_sample)

        if modifications:
            print(f"样本 {idx+1}: {', '.join(modifications)} | 成功={is_success}", flush=True)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(subset):
            asr = (scam_success/scam_total*100) if scam_total > 0 else 0
            print(f"进度: {idx+1}/{len(subset)} | scam={scam_total} success={scam_success} ASR={asr:.2f}%", flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    print(f"\n已写入: {args.output_json}")
    asr_final = (scam_success/scam_total*100) if scam_total > 0 else 0
    print(f"ASR: {scam_success}/{scam_total} ({asr_final:.2f}%)")
