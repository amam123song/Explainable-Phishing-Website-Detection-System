#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScamNet Explain 学生 LoRA 的层级剪枝脚本（按 LoRA 层重要性整体关闭）

设计目标（第二阶段·层级剪枝版）：
1. 在已经统一 rank=4 且效果良好的 Explain 学生 LoRA 上，进一步降低 LoRA 占用的显存；
2. 不再细抠每个 rank，而是以「LoRA 层」为单位：
   - 用一阶 |w * grad| 重要性对每个 LoRA 适配器（每一层的 LoRA）求总分；
   - 按总分排序，砍掉最不重要的若干个 LoRA 适配器（例如 20% 或 30%）；
3. 剪枝方式：
   - 默认实现为“彻底移除 adapter”：从 module.lora_A / lora_B 中删除对应键，
     这样该层不再有任何 LoRA 参数和前向计算，显存和计算开销都线性下降；
   - 若你希望保守一些，也可以把下面的 `remove_mode` 参数改成 "zero"，仅把权重置零。
4. 剪枝后保存新的 LoRA 目录，建议再用 distillation_explain_train.py 做 2–3 epoch 小学习率微调，
   并用 06_inference/run_benchmark.py --run dual_full --student-explain-lora 新路径评估解释质量与耗时/显存。
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
from peft import PeftModel


@dataclass
class ExplainExample:
    inp: str
    out: str


class ExplainPruneDataset(Dataset):
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
    关键参数：
    - student_base: Mistral-7B 基座（本地快照）；
    - student_lora: 已经是 rank 收缩 + 微调后的 Explain LoRA（如 output_explain_pruned_rankshrink_r4_ft）；
    - data_path: 含 Label+解释的 JSON 数据，用于统计梯度重要性；
    - output_dir: 层级剪枝后的 LoRA 保存目录；
    - prune_layer_ratio: 要砍掉的 LoRA 适配器比例（0.2 即 20%）；
    - max_prune_samples / batch_size: 梯度统计采样控制；
    - remove_mode:
        - "remove": 从 lora_A/lora_B 中删除整个 adapter（推荐，更省显存与算力）；
        - "zero": 仅将该 adapter 的 A/B 权重置零（结构不变，更保守）。
    """
    p = argparse.ArgumentParser(
        description="Explain 学生 LoRA 的层级剪枝（按 LoRA 层总重要性整体关闭）"
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
        default="outputs/decouple/explain_pruned_rankshrink_r4_ft",
        help="待做层级剪枝的 Explain LoRA 目录",
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
        default="outputs/decouple/explain_layerpruned",
        help="层级剪枝后 LoRA 保存目录",
    )
    p.add_argument(
        "--prune_layer_ratio",
        type=float,
        default=0.3,
        help="要砍掉的 LoRA 适配器占比（0~1 之间，例如 0.3 表示剪掉 30% 最不重要的层）",
    )
    p.add_argument(
        "--max_prune_samples",
        type=int,
        default=200,
        help="用于梯度统计的最大样本数",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="梯度统计时的 batch size",
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
    p.add_argument(
        "--remove_mode",
        type=str,
        default="remove",
        choices=["remove", "zero"],
        help='层级剪枝方式："remove" 彻底删除 adapter（推荐），"zero" 仅将其权重置零',
    )

    return p.parse_args()


def load_explain_data(path: str, max_samples: int) -> List[ExplainExample]:
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

    print(f"[INFO] 层级剪枝阶段实际使用样本数: {len(exs)} (max_prune_samples={max_samples})")
    return exs


def collect_lora_layers(model: PeftModel):
    """
    收集所有 LoRA 适配器，返回：
    - layers: Dict[str, Dict]，形如：
        {
          "layer0.self_attn.v_proj.default": {
              "module": <LoraLayer>,
              "adapter_name": "default",
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
                "adapter_name": adapter_name,
                "A": A_mod,
                "B": B_mod,
            }

    if not layers:
        raise RuntimeError(
            "未在模型模块中找到任何带 lora_A/lora_B 的 LoRA 适配器，请确认 student_lora 是否为 LoRA 适配器，"
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


def compute_layer_importance(
    model: PeftModel,
    dataset: ExplainPruneDataset,
    lora_layers: Dict[str, Dict[str, torch.nn.Module]],
    batch_size: int,
) -> Dict[str, float]:
    """
    统计每个 LoRA 适配器（层）的总重要性：
    - 仍基于 |w * grad| 一阶近似；
    - 但这次对该层所有 rank 和全部权重求和，得到一个标量 score_layer；
    - 后续按 score_layer 排序，剪掉最低的一部分。
    """
    device = model.device
    model.train()

    # 初始化每层重要性为 0
    importance: Dict[str, float] = {name: 0.0 for name in lora_layers.keys()}

    # 确保所有 LoRA 参数参与梯度计算
    for mods in lora_layers.values():
        A = mods["A"].weight
        B = mods["B"].weight
        A.requires_grad_(True)
        B.requires_grad_(True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"[INFO] 开始基于 {len(dataset)} 条样本统计 LoRA 层级重要性...")

    for batch in tqdm(loader, desc="Collecting gradients for layer importance"):
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

        # 对每一层的 A/B 做 |w * grad| 汇总
        for name, mods in lora_layers.items():
            A_w = mods["A"].weight
            B_w = mods["B"].weight
            if A_w.grad is None or B_w.grad is None:
                continue
            grad_A = A_w.grad
            grad_B = B_w.grad
            with torch.no_grad():
                # A 部分
                score_A = (A_w * grad_A).abs().sum().item()
                # B 部分
                score_B = (B_w * grad_B).abs().sum().item()
                importance[name] += score_A + score_B

        # 清梯度
        for mods in lora_layers.values():
            A_w = mods["A"].weight
            B_w = mods["B"].weight
            if A_w.grad is not None:
                A_w.grad = None
            if B_w.grad is not None:
                B_w.grad = None

    print("[INFO] LoRA 层级重要性统计完成。")
    return importance


def apply_layer_pruning(
    model: PeftModel,
    lora_layers: Dict[str, Dict[str, torch.nn.Module]],
    importance: Dict[str, float],
    prune_layer_ratio: float,
    remove_mode: str = "remove",
):
    """
    根据每层的重要性分数，对 LoRA 适配器做层级剪枝：
    - 按 score 从小到大排序；
    - 剪掉最不重要的 K = floor(prune_layer_ratio * num_layers) 个适配器；
    - remove_mode:
        - "remove": 从 module.lora_A / lora_B 中删除 adapter（推荐）；
        - "zero": 仅将该 adapter 的 A/B 权重置零。
    """
    if not (0.0 < prune_layer_ratio < 1.0):
        raise ValueError(
            f"prune_layer_ratio 必须在 (0,1) 之间，目前为 {prune_layer_ratio}"
        )

    num_layers = len(lora_layers)
    if num_layers == 0:
        raise RuntimeError("没有可剪枝的 LoRA 层（num_layers=0）")

    K = int(num_layers * prune_layer_ratio)
    if K < 1:
        K = 1
    if K >= num_layers:
        K = num_layers - 1

    # 排序：得分越小越不重要
    items = sorted(importance.items(), key=lambda x: x[1])
    to_prune = items[:K]

    print(
        f"[INFO] LoRA 总层数: {num_layers}, 计划剪掉 {K} 层 (~{K/num_layers:.2%})，"
        f"模式: {remove_mode}"
    )
    print("[INFO] 将被剪掉的若干层（name, score）：")
    for name, score in to_prune[:20]:
        print(f"  - {name}: importance={score:.4e}")
    if len(to_prune) > 20:
        print("  ... 其余被剪层省略打印")

    # 实际剪枝
    with torch.no_grad():
        for name, score in to_prune:
            mods = lora_layers[name]
            module = mods["module"]
            adapter_name = mods["adapter_name"]

            if remove_mode == "remove":
                # 从 ModuleDict 中删除 adapter
                if adapter_name in module.lora_A:
                    del module.lora_A[adapter_name]
                if adapter_name in module.lora_B:
                    del module.lora_B[adapter_name]
            else:  # "zero"
                A_w = mods["A"].weight
                B_w = mods["B"].weight
                A_w.zero_()
                B_w.zero_()

    print("[INFO] LoRA 层级剪枝操作完成。")


def main():
    args = parse_args()

    print("=" * 60)
    print("ScamNet Explain 学生 LoRA 层级剪枝（第二阶段·按层关闭 LoRA）")
    print("=" * 60)

    if not os.path.exists(args.student_lora):
        raise FileNotFoundError(f"待剪枝 Explain LoRA 不存在: {args.student_lora}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"层级剪枝数据集不存在: {args.data_path}")

    print(f"[CONFIG] student_base      = {args.student_base}")
    print(f"[CONFIG] student_lora      = {args.student_lora}")
    print(f"[CONFIG] data_path         = {args.data_path}")
    print(f"[CONFIG] output_dir        = {args.output_dir}")
    print(f"[CONFIG] prune_layer_ratio = {args.prune_layer_ratio}")
    print(f"[CONFIG] remove_mode       = {args.remove_mode}")
    print(f"[CONFIG] max_samples       = {args.max_prune_samples}")
    print(f"[CONFIG] batch_size        = {args.batch_size}")
    print(f"[CONFIG] max_length        = {args.max_length}")
    print(f"[CONFIG] max_input_chars   = {args.max_input_chars}")

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
    print("\n[2/4] 构造层级剪枝用 ExplainPruneDataset...")
    prune_dataset = ExplainPruneDataset(
        examples,
        tokenizer,
        max_length=args.max_length,
        max_input_chars=args.max_input_chars,
    )
    print(f"[INFO] 层级剪枝 Dataset 大小: {len(prune_dataset)}")

    # 4. 收集 LoRA 层并统计每层的总重要性
    print("\n[3/4] 收集 LoRA 模块，并统计每层的重要性...")
    lora_layers = collect_lora_layers(student)
    student.to(device)

    layer_importance = compute_layer_importance(
        student, prune_dataset, lora_layers, batch_size=args.batch_size
    )

    # 5. 按层重要性执行剪枝
    print("\n[4/4] 依据总重要性执行 LoRA 层级剪枝...")
    apply_layer_pruning(
        student,
        lora_layers,
        layer_importance,
        prune_layer_ratio=args.prune_layer_ratio,
        remove_mode=args.remove_mode,
    )

    # 6. 保存剪枝后的 LoRA
    print("\n保存层级剪枝后的 Explain LoRA...")
    os.makedirs(args.output_dir, exist_ok=True)
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] 层级剪枝后 LoRA 已保存到: {args.output_dir}")

    print("\n后续推荐步骤：")
    print("1) 使用 distillation_explain_train.py 在层级剪枝后 LoRA 上做 2–3 epoch 小学习率微调，例如：")
    print(
        "   python distillation_explain_train.py \\"
        "\n       --student_base {base} \\"
        "\n       --student_lora_init {pruned} \\"
        "\n       --data_path data/dataset_explainable_200.json \\"
        "\n       --output_dir outputs/decouple/explain_adapter \\"
        "\n       --epochs 3 --lr 1e-5".format(
            base=args.student_base, pruned=args.output_dir
        )
    )
    print("2) 用 06_inference/run_benchmark.py --run dual_full --student-explain-lora 指向新 LoRA，")
    print("   对比层级剪枝前后解释结构/质量与 Explain 阶段耗时 / GPU 显存占用。")

    print("\n" + "=" * 60)
    print("LoRA 层级剪枝完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()

