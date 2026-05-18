#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScamNet 第二阶段：Explain 学生 LoRA 的 rank 收缩（物理降 rank）脚本

设计目标：
1. 基于第一阶段已经剪枝+微调好的 Explain 学生 LoRA（如 outputs/decouple/explain_adapter_pruned_ft）；
2. 再次用 |w * grad| 的一阶泰勒近似评估每个 LoRA rank 的重要性；
3. 在保持 LoRA 结构正确的前提下，把每个 LoRA 适配器的 rank 从 r 降到 r_new（例如 r=8 -> r_new=4），
   也就是“物理收缩” A/B 的维度，而不是简单置零：
   - A: (r, in_features)  -> (r_new, in_features)
   - B: (out_features, r) -> (out_features, r_new)
4. 同步更新 peft 的 LoRA 配置中的 r 参数，并保存为新的 LoRA 目录，供后续小学习率微调与评估。

注意：
- 本脚本假设所有 LoRA 适配器共享同一个全局 rank r，且最终要收缩到统一的 r_new；
- 若你希望“分层不同 rank”，则需要进一步改写 peft 的配置结构，目前先实现统一 r_new 的版本；
- 与第一阶段的 prune_output_explain_lora_rank.py 不同，本脚本会真正改变 LoRA 权重矩阵的形状，
  推理时的矩阵乘法复杂度会线性下降，从而带来 Explain 阶段的真实加速。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


@dataclass
class ExplainExample:
    """
    用于剪枝/收缩阶段的简单数据结构：
    - inp: 原始输入（含 URL / HTML / External Links 等）；
    - out: 完整教师输出（Label + 解释），用于构造监督信号。
    """

    inp: str
    out: str


class ExplainPruneDataset(Dataset):
    """
    专用于统计梯度重要性的 Dataset：
    - 和 distillation_explain_train.ExplainDataset 类似，但更轻量；
    - 只负责把 (input, output) 转成 (input_ids, attention_mask, labels)，
      其中：
        - prompt 只包含截断后的 input 片段；
        - labels 中 prompt 部分被 mask 掉（= -100），只对 Label+解释部分计损失；
    """

    def __init__(
        self,
        data: List[ExplainExample],
        tokenizer,
        max_length: int = 1024,
        max_input_chars: int = 800,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_input_chars = max_input_chars

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        ex = self.data[idx]
        raw_input = ex.inp
        raw_output = ex.out

        website_snippet = raw_input[: self.max_input_chars]
        prompt = f"Website: {website_snippet}\n"
        full_text = prompt + raw_output

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_ids), self.max_length)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args():
    """
    命令行参数：
    - student_base: Mistral-7B 基座（推荐本地快照）；
    - student_lora: 已经做过第一阶段剪枝+微调的 Explain LoRA（如 output_explain_pruned_ft）；
    - data_path: 含 Label+解释的 JSON 数据（可直接用 dataset_explainable_200.json）；
    - output_dir: 收缩后 LoRA 保存目录；
    - keep_ratio: 目标 rank 占原始 rank 的比例，例如 0.5 表示 r_new ≈ r * 0.5；
    - target_r: 直接指定目标 rank（正整数，优先级高于 keep_ratio）；
    - max_prune_samples: 用于统计梯度重要性的样本数上限；
    - batch_size: 统计梯度时的 batch size；
    - max_length/max_input_chars: 序列长度与输入截断控制；
    - device_map: transformers 的 device_map。
    """
    p = argparse.ArgumentParser(
        description="Explain 学生 LoRA 的 rank 收缩（物理降 rank）"
    )

    p.add_argument(
        "--student_base",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="学生基座模型（HF repo id 或本地目录）",
    )
    p.add_argument(
        "--student_lora",
        type=str,
        default="outputs/decouple/explain_adapter_pruned_ft",
        help="待收缩的 Explain LoRA 目录（通常是第一阶段剪枝+微调后的结果）",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="data/dataset_explainable_200.json",
        help="包含 Label+解释 的 JSON 数据，用于统计梯度重要性",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/decouple/explain_adapter_pruned_rankshrink",
        help="收缩后 LoRA 保存目录",
    )
    p.add_argument(
        "--keep_ratio",
        type=float,
        default=0.5,
        help="目标 rank 占原始 rank 的比例（0~1 之间，若同时指定 target_r，则忽略）",
    )
    p.add_argument(
        "--target_r",
        type=int,
        default=0,
        help="直接指定目标 rank（>0 时优先使用；<=0 表示按 keep_ratio 自动计算）",
    )
    p.add_argument(
        "--max_prune_samples",
        type=int,
        default=200,
        help="用于梯度统计的最大样本数（越大越稳，但越慢）",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="统计梯度时的 batch size",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="token 级最大序列长度（prompt+输出）",
    )
    p.add_argument(
        "--max_input_chars",
        type=int,
        default=800,
        help="构造 prompt 时 input 截断的最大字符数",
    )
    p.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='transformers 的 device_map（默认 "auto"）',
    )

    return p.parse_args()


def load_explain_data(path: str, max_samples: int) -> List[ExplainExample]:
    """
    读取包含 Label+解释 的 JSON 数据，格式与 distillation_explain_train 一致：
    - 每条样本是一个 dict，至少包含：
      - "input": 原始输入（URL/HTML 等）
      - "output": 完整输出（从 Label: ... 到解释结束）
    - 本函数只取前 max_samples 条，用于收缩阶段的梯度统计。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    exs: List[ExplainExample] = []
    for item in data[:max_samples]:
        inp = item.get("input", "")
        out = item.get("output", "")
        if not inp or not out:
            continue
        exs.append(ExplainExample(inp=inp, out=out))

    if not exs:
        raise ValueError(f"数据集中有效样本数为 0，请检查 {path}")

    print(f"[INFO] 收缩阶段实际使用样本数: {len(exs)} (max_prune_samples={max_samples})")
    return exs


def collect_lora_layers(model: PeftModel):
    """
    从 PeftModel 中收集所有带 LoRA 的模块及其 A/B 子模块。

    返回：
    - layers: Dict[str, Dict]，形如：
        {
          "layer0.self_attn.v_proj.default": {
              "module": <LoraLayer>,
              "A": <nn.Linear>,  # in -> r
              "B": <nn.Linear>,  # r -> out
          },
          ...
        }
    """
    layers: Dict[str, Dict[str, torch.nn.Module]] = {}

    for module_name, module in model.named_modules():
        lora_A = getattr(module, "lora_A", None)
        lora_B = getattr(module, "lora_B", None)
        if lora_A is None or lora_B is None:
            continue

        for adapter_name, A_mod in lora_A.items():
            if adapter_name not in lora_B:
                continue
            B_mod = lora_B[adapter_name]

            if not hasattr(A_mod, "weight") or not hasattr(B_mod, "weight"):
                continue

            key = f"{module_name}.{adapter_name}"
            layers[key] = {
                "module": module,
                "A": A_mod,
                "B": B_mod,
            }

    if not layers:
        raise RuntimeError(
            "未在模型模块中找到任何带 lora_A/lora_B 的 LoRA 层，请确认 student_lora 是否为 LoRA 适配器，"
            "以及当前 peft 版本是否与训练时一致。"
        )

    print(f"[INFO] 检测到 LoRA 适配器数: {len(layers)}")
    for k, v in list(layers.items())[:5]:
        A_w = v["A"].weight
        B_w = v["B"].weight
        print(
            f"  - {k}: A.weight.shape={tuple(A_w.shape)}, B.weight.shape={tuple(B_w.shape)}"
        )
    if len(layers) > 5:
        print("  ... 其余适配器省略打印")

    return layers


def compute_rank_importance(
    model: PeftModel,
    dataset: ExplainPruneDataset,
    lora_layers: Dict[str, Dict[str, torch.nn.Module]],
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """
    基于一批样本，计算每个 LoRA 适配器内部每个 rank 的 |w * grad| 重要性得分。

    与第一阶段脚本类似：
    - 对每个 batch：
      1. 前向计算 loss（仅解释部分有 label）；
      2. loss.backward() 得到每个 LoRA A/B 的梯度；
      3. 对于每个适配器、每个 rank k：
         - A 部分：mean(|A_k * grad_A_k|)；
         - B 部分：mean(|B_k * grad_B_k|)；
         - 二者相加作为该 batch 对该 rank 的贡献；
      4. 在所有 batch 上累加，得到全局重要性。
    """
    device = model.device
    model.train()

    importance: Dict[str, torch.Tensor] = {}

    # 初始化重要性累加器
    for name, mods in lora_layers.items():
        A = mods["A"].weight
        r = A.shape[0]  # (r, in_features)
        importance[name] = torch.zeros(r, dtype=torch.float64, device="cpu")
        A.requires_grad_(True)
        mods["B"].weight.requires_grad_(True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"[INFO] 开始基于 {len(dataset)} 条样本统计 LoRA rank 重要性...")

    for batch in tqdm(loader, desc="Collecting gradients for importance"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss

        if not torch.isfinite(loss):
            print(f"[WARN] loss 非有限值（{loss.item()}），跳过该 batch")
            continue

        loss.backward()

        for name, mods in lora_layers.items():
            A_mod = mods["A"]
            B_mod = mods["B"]
            A = A_mod.weight
            B = B_mod.weight

            if A.grad is None or B.grad is None:
                continue

            grad_A = A.grad
            grad_B = B.grad

            # 约定：
            #   A.weight: (r, in_features)
            #   B.weight: (out_features, r)
            r_A = A.shape[0]
            r_B = B.shape[1]
            if r_A != r_B:
                raise RuntimeError(
                    f"LoRA 适配器 {name} 的 A/B rank 维度不一致: A.shape={A.shape}, B.shape={B.shape}"
                )

            with torch.no_grad():
                score_A = (A * grad_A).abs().mean(dim=1).detach().cpu()
                score_B = (B * grad_B).abs().mean(dim=0).detach().cpu()
                score = score_A + score_B

            importance[name] += score.to(importance[name].dtype)

        # 清梯度，避免显存累积
        for mods in lora_layers.values():
            A_mod = mods["A"]
            B_mod = mods["B"]
            if A_mod.weight.grad is not None:
                A_mod.weight.grad = None
            if B_mod.weight.grad is not None:
                B_mod.weight.grad = None

    print("[INFO] LoRA rank 重要性统计完成。")
    return importance


def determine_new_rank(
    lora_layers: Dict[str, Dict[str, torch.nn.Module]],
    keep_ratio: float,
    target_r: int,
) -> Tuple[int, int]:
    """
    依据 keep_ratio / target_r 决定统一的新 rank r_new。
    假设所有 LoRA 适配器当前 rank 相同（统一为 r）。
    返回 (r, r_new)。
    """
    if not (0.0 < keep_ratio <= 1.0) and target_r <= 0:
        raise ValueError(
            f"keep_ratio 必须在 (0,1] 之间，或指定正整数 target_r；"
            f"当前 keep_ratio={keep_ratio}, target_r={target_r}"
        )

    # 取第一个 LoRA 适配器的 rank 作为全局 r
    first_key = next(iter(lora_layers.keys()))
    A_mod = lora_layers[first_key]["A"]
    r = A_mod.weight.shape[0]

    # sanity check：所有适配器的 rank 是否一致
    for name, mods in lora_layers.items():
        cur_r = mods["A"].weight.shape[0]
        if cur_r != r:
            raise RuntimeError(
                f"检测到不同 LoRA 适配器的 rank 不一致：{first_key} 的 r={r}, 但 {name} 的 r={cur_r}，"
                f"当前脚本仅支持统一 rank 收缩。"
            )

    if target_r > 0:
        r_new = target_r
    else:
        r_new = int(round(r * keep_ratio))

    if r_new < 1:
        r_new = 1
    if r_new >= r:
        raise ValueError(
            f"目标 r_new={r_new} 不小于原始 r={r}，无收缩意义，请调整 keep_ratio/target_r。"
        )

    print(f"[INFO] 原始 LoRA rank = {r}, 目标 rank = {r_new}")
    return r, r_new


def shrink_lora_ranks(
    model: PeftModel,
    lora_layers: Dict[str, Dict[str, torch.nn.Module]],
    importance: Dict[str, torch.Tensor],
    r: int,
    r_new: int,
):
    """
    根据重要性得分，对每个 LoRA 适配器执行“物理降 rank”操作：
    - 对每个适配器 name：
      - importance[name] 是长度 r 的向量；
      - 取其中得分最高的 r_new 个 rank 索引 keep_idx；
      - 重新构造 A/B 子模块（新的 nn.Linear），维度缩小到 r_new；
      - 替换 module.lora_A[adapter] 与 module.lora_B[adapter]。
    同时会尝试更新 peft_config 中的 r 字段。
    """
    assert 0 < r_new < r

    # 统计一下每个适配器保留的 rank 索引，方便 debug
    print("[INFO] 开始对各 LoRA 适配器做 rank 收缩...")
    for name, mods in lora_layers.items():
        A_mod = mods["A"]
        B_mod = mods["B"]
        scores = importance[name]  # [r]
        if scores.numel() != r:
            raise RuntimeError(
                f"importance 中 {name} 的长度={scores.numel()} 与 r={r} 不一致"
            )

        # 取得分最高的 r_new 个 rank，降序排序
        topk = torch.topk(scores, k=r_new, largest=True, sorted=True)
        keep_idx = topk.indices.tolist()  # 从大到小
        keep_idx_sorted = sorted(keep_idx)  # 为了使用更稳定的列顺序
        print(f"  - {name}: keep ranks (sorted) = {keep_idx_sorted}")

        with torch.no_grad():
            old_A_w = A_mod.weight.data  # (r, in_features)
            old_B_w = B_mod.weight.data  # (out_features, r)

            # 新的 A/B 权重
            A_new_w = old_A_w[keep_idx_sorted, :].contiguous()
            B_new_w = old_B_w[:, keep_idx_sorted].contiguous()

            # 利用原模块的 in_features/out_features 构造新的 nn.Linear
            in_features = A_mod.in_features
            out_features = B_mod.out_features

            new_A = torch.nn.Linear(in_features, r_new, bias=False, device=old_A_w.device, dtype=old_A_w.dtype)
            new_B = torch.nn.Linear(r_new, out_features, bias=False, device=old_B_w.device, dtype=old_B_w.dtype)

            new_A.weight.data.copy_(A_new_w)
            new_B.weight.data.copy_(B_new_w)

            # 替换到原有的 lora_A / lora_B ModuleDict 中
            adapter_name = name.split(".")[-1]
            # module 本身在 lora_layers[name]["module"] 里
            module = lora_layers[name]["module"]
            module.lora_A[adapter_name] = new_A
            module.lora_B[adapter_name] = new_B

    # 更新 peft_config 中的 rank
    if isinstance(model.peft_config, dict):
        for cfg_name, cfg in model.peft_config.items():
            if hasattr(cfg, "r"):
                print(f"[INFO] 更新 peft_config[{cfg_name}].r: {cfg.r} -> {r_new}")
                cfg.r = r_new

    print("[INFO] LoRA rank 收缩完成。")


def main():
    args = parse_args()

    print("=" * 60)
    print("ScamNet Explain 学生 LoRA rank 收缩（第二阶段·物理降 rank）")
    print("=" * 60)

    # 基本检查
    if not os.path.exists(args.student_lora):
        raise FileNotFoundError(f"待收缩 Explain LoRA 不存在: {args.student_lora}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"收缩阶段数据集不存在: {args.data_path}")

    print(f"[CONFIG] student_base   = {args.student_base}")
    print(f"[CONFIG] student_lora   = {args.student_lora}")
    print(f"[CONFIG] data_path      = {args.data_path}")
    print(f"[CONFIG] output_dir     = {args.output_dir}")
    print(f"[CONFIG] keep_ratio     = {args.keep_ratio}")
    print(f"[CONFIG] target_r       = {args.target_r}")
    print(f"[CONFIG] max_samples    = {args.max_prune_samples}")
    print(f"[CONFIG] batch_size     = {args.batch_size}")
    print(f"[CONFIG] max_length     = {args.max_length}")
    print(f"[CONFIG] max_input_char = {args.max_input_chars}")

    # 1. 加载数据
    examples = load_explain_data(args.data_path, args.max_prune_samples)

    # 2. 加载学生模型（Mistral 基座 + Explain LoRA）
    print("\n[1/4] 加载 Mistral-7B 基座与 Explain LoRA...")
    bnb = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.student_base, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.student_base,
        quantization_config=bnb,
        device_map=args.device_map,
        local_files_only=True,
    )
    if hasattr(base, "config"):
        base.config.use_cache = False

    student = PeftModel.from_pretrained(base, args.student_lora)
    student.eval()
    device = student.device
    print(f"[INFO] 学生模型已加载，device = {device}")

    # 3. 构造 Dataset
    print("\n[2/4] 构造收缩用 ExplainPruneDataset...")
    prune_dataset = ExplainPruneDataset(
        examples,
        tokenizer,
        max_length=args.max_length,
        max_input_chars=args.max_input_chars,
    )
    print(f"[INFO] 收缩 Dataset 大小: {len(prune_dataset)}")

    # 4. 收集 LoRA 层并统计每个 rank 的 |w * grad| 重要性
    print("\n[3/4] 收集 LoRA A/B 模块，并统计各 rank 重要性...")
    lora_layers = collect_lora_layers(student)
    student.to(device)

    importance = compute_rank_importance(
        student, prune_dataset, lora_layers, batch_size=args.batch_size
    )

    # 5. 决定统一的新 rank，并对所有 LoRA 适配器执行物理收缩
    print("\n[4/4] 依据重要性执行 LoRA rank 收缩...")
    r, r_new = determine_new_rank(
        lora_layers, keep_ratio=args.keep_ratio, target_r=args.target_r
    )
    shrink_lora_ranks(student, lora_layers, importance, r=r, r_new=r_new)

    # 6. 保存收缩后的 LoRA 适配器
    print("\n保存收缩后的 Explain LoRA...")
    os.makedirs(args.output_dir, exist_ok=True)
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] 收缩后 LoRA 已保存到: {args.output_dir}")

    print("\n后续推荐步骤：")
    print("1) 使用 distillation_explain_train.py 在收缩后 LoRA 上做 2–3 epoch 小学习率微调，例如：")
    print(
        "   python distillation_explain_train.py \\"
        "\n       --student_base {base} \\"
        "\n       --student_lora_init {pruned} \\"
        "\n       --data_path data/dataset_explainable_200.json \\"
        "\n       --output_dir outputs/decouple/explain_adapter_pruned_rankshrink_ft \\"
        "\n       --epochs 3 --lr 1e-5".format(
            base=args.student_base, pruned=args.output_dir
        )
    )
    print("2) 用 06_inference/run_benchmark.py --student-explain-lora 指向新 LoRA，")
    print("   对比 rank 收缩前后解释结构/质量与解释阶段耗时。")

    print("\n" + "=" * 60)
    print("LoRA rank 收缩完成（第二阶段）。")
    print("=" * 60)


if __name__ == "__main__":
    main()

