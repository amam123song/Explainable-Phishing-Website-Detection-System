#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScamNet 第一阶段：Explain 学生 LoRA 的 rank 级结构化剪枝脚本

目的（低风险版本）：
1. 只在解释学生 LoRA（outputs/decouple/explain_adapter）上做剪枝，不动 Mistral-7B 基座；
2. 粒度是「LoRA rank」级的结构化剪枝：
   - 对每个被注入 LoRA 的线性层，其 LoRA 形式为 A (d_out x r) 与 B (r x d_in)；
   - 每一个 rank k 对应一个 rank-1 更新 A[:,k] @ B[k,:]，可以看作一个“结构化神经元”；
   - 本脚本会基于 |w * grad| 的一阶泰勒近似，估计每个 rank 对损失的影响重要性；
   - 再全局排序，砍掉重要性最低的若干个 rank（默认 30%）。
3. 剪枝后保存一个新的 LoRA 目录（例如 outputs/decouple/explain_adapter_pruned_rank30）：
   - 后续第二阶段可以用 distillation_explain_train.py 再微调 2–3 epoch（lr≈1e-5）恢复性能；
   - 再用 06_inference/run_benchmark.py --run dual_full 评估解释质量与推理耗时。

注意：
- 本脚本不会更新权重，只做一次“统计梯度 → 计算重要性 → 置零部分 rank”；
- 置零是一种保守剪枝方式：保持 LoRA 结构与 config 不变，方便直接复用现有加载/推理脚本；
- 若后续需要“物理删掉列/行以进一步加速”，可以在此脚本的基础上做更激进改动。
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
    """
    用于剪枝阶段的简单数据结构：
    - input: 原始输入（含 URL / HTML / External Links 等）；
    - output: 完整教师输出（Label + 解释），用于构造监督信号。
    """

    inp: str
    out: str


class ExplainPruneDataset(Dataset):
    """
    剪枝专用 Dataset：
    - 和 distillation_explain_train.ExplainDataset 类似，但更简单；
    - 只负责把 (input, output) 转成 (input_ids, attention_mask, labels)，
      其中：
        - prompt 只包含截断后的 input 片段；
        - labels 中 prompt 部分被 mask 掉（= -100），只对 Label+解释部分计损失；
    - 这样模型的 loss 更直接反映“解释能力”，有利于基于梯度估计 LoRA rank 重要性。
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

        # 和 distillation_explain_train 中保持风格一致：
        # prompt 只包含截断后的输入摘要，模型需要在其后生成 Label + 解释。
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

        # 计算 prompt 的 token 长度，用于 mask 掉 prompt 的 loss
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_ids), self.max_length)

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # prompt 不计入 loss
        labels[attention_mask == 0] = -100  # padding 不计入 loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args():
    """
    命令行参数：
    - student_base: Mistral-7B 基座（推荐使用本地快照，避免网络下载）；
    - student_lora: 要被剪枝的 Explain LoRA（通常是 outputs/decouple/explain_adapter）；
    - data_path: 含 Label+解释的 JSON 数据（可直接用 dataset_explainable_200.json）；
    - output_dir: 剪枝后的 LoRA 保存目录；
    - prune_ratio: 全局要砍掉的 LoRA rank 占比（0.3 即 30%）；
    - max_prune_samples: 用于统计梯度的样本数上限（越多越稳，但越慢）；
    - batch_size: 统计梯度时的 batch size；
    - max_length/max_input_chars: 序列长度与输入截断控制；
    - device_map: transformers 的 device_map，用于多卡/单卡部署。
    """
    p = argparse.ArgumentParser(description="Explain 学生 LoRA 的 rank 级结构化剪枝")

    p.add_argument(
        "--student_base",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="学生基座模型（HF repo id 或本地目录）",
    )
    p.add_argument(
        "--student_lora",
        type=str,
        default="outputs/decouple/explain_adapter",
        help="待剪枝的 Explain LoRA 目录",
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
        default="outputs/decouple/explain_adapter_pruned_rank30",
        help="剪枝后 LoRA 保存目录",
    )
    p.add_argument(
        "--prune_ratio",
        type=float,
        default=0.3,
        help="全局要剪掉的 LoRA rank 占比（0~1 之间，默认 0.3）",
    )
    p.add_argument(
        "--max_prune_samples",
        type=int,
        default=200,
        help="用于梯度统计的最大样本数（越大越稳，但越慢）",
    )
    p.add_argument("--batch_size", type=int, default=2, help="统计梯度时的 batch size")
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
    读取包含 Label+解释 的 JSON 数据。

    期望格式与 distillation_explain_train 一致：
    - 每条样本是一个 dict，至少包含：
      - "input": 原始输入（URL/HTML 等）
      - "output": 完整输出（从 Label: ... 到解释结束）
    - 本函数只取前 max_samples 条，用于剪枝阶段的梯度统计。
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

    print(f"[INFO] 剪枝阶段实际使用样本数: {len(exs)} (max_prune_samples={max_samples})")
    return exs


def collect_lora_rank_handles(model: PeftModel):
    """
    从 PeftModel 中收集所有 LoRA A/B 参数，并按照“逻辑 LoRA 适配器”分组。

    说明：
    - 之前的实现是通过 named_parameters() 里匹配 "lora_A"/"lora_B" 的字符串来找权重，
      但不同 peft 版本/配置下参数名可能比较花哨，容易漏掉；
    - 更稳妥的做法是：直接遍历模块，查找具有 .lora_A / .lora_B 属性的 LoraLayer，
      再从其中取出每个 adapter 的 A/B 权重（module.lora_A[adapter].weight 等）。

    返回：
    - adapters: Dict[str, Dict[str, torch.nn.Parameter]]，形如：
        {
          "layer0.self_attn.v_proj.default": {"A": param_A, "B": param_B},
          ...
        }
    """
    adapters: Dict[str, Dict[str, torch.nn.Parameter]] = {}

    # 遍历所有模块，寻找带有 lora_A / lora_B 属性的 LoRA 层
    for module_name, module in model.named_modules():
        lora_A = getattr(module, "lora_A", None)
        lora_B = getattr(module, "lora_B", None)
        if lora_A is None or lora_B is None:
            continue

        # lora_A / lora_B 通常是一个 ModuleDict，里面按 adapter_name 存放不同适配器
        for adapter_name, A_mod in lora_A.items():
            if adapter_name not in lora_B:
                continue
            B_mod = lora_B[adapter_name]

            # 只处理带有 weight 的线性型 LoRA（最常见情况）
            if not hasattr(A_mod, "weight") or not hasattr(B_mod, "weight"):
                continue

            key = f"{module_name}.{adapter_name}"
            adapters[key] = {
                "A": A_mod.weight,
                "B": B_mod.weight,
            }

    if not adapters:
        # 这里保留清晰报错，方便你确认当前 LoRA 结构
        raise RuntimeError(
            "未在模型模块中找到任何带 lora_A/lora_B 的 LoRA 层，请确认 student_lora 是否为 LoRA 适配器，"
            "以及当前 peft 版本是否与训练时一致。"
        )

    print(f"[INFO] 检测到 LoRA 适配器数: {len(adapters)}")
    for k, v in list(adapters.items())[:5]:
        print(
            f"  - {k}: A.shape={tuple(v['A'].shape)}, B.shape={tuple(v['B'].shape)}"
        )
    if len(adapters) > 5:
        print("  ... 其余适配器省略打印")

    return adapters


def compute_rank_importance(
    model: PeftModel,
    dataset: ExplainPruneDataset,
    lora_adapters: Dict[str, Dict[str, torch.nn.Parameter]],
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """
    基于一批样本，计算每个 LoRA 适配器内部每个 rank 的 |w * grad| 重要性得分。

    具体做法（一次一阶泰勒近似）：
    - 对每个 batch：
      1. 前向计算 loss（仅解释部分有 label）；
      2. 对 loss.backward()，得到每个 LoRA 参数的梯度 grad_W；
      3. 对于每个适配器、每个 rank k：
         - 取 A[:,k] 与 grad_A[:,k]，计算 mean(|A[:,k] * grad_A[:,k]|)；
         - 取 B[k,:] 与 grad_B[k,:]，计算 mean(|B[k,:] * grad_B[k,:]|)；
         - 二者相加作为该 batch 对该 rank 的重要性贡献；
      4. 在所有 batch 上累加，最后得到每个 rank 的全局得分。
    - 理解为：|w * grad| 越大，对当前 loss 越敏感／贡献越大，不宜剪掉。

    返回：
    - importance: Dict[str, Tensor]，其中每个 key 对应 [r] 维度的张量，存储每个 rank 的总得分。
    """
    device = model.device
    model.train()  # 为了能计算梯度，这里设为 train 模式（不会真的更新权重）

    # 初始化重要性累加器：每个适配器一条 [r] 张量
    importance: Dict[str, torch.Tensor] = {}
    for name, params in lora_adapters.items():
        A = params["A"]
        # 注意：在当前 peft 版本中，LoRA 线性层的形状通常为：
        #   A.weight: (r, in_features)
        #   B.weight: (out_features, r)
        # 即 rank = A.shape[0] = B.shape[1]
        r = A.shape[0]  # LoRA rank
        importance[name] = torch.zeros(r, dtype=torch.float64, device="cpu")

        # 确保 LoRA 参数有梯度
        A.requires_grad_(True)
        params["B"].requires_grad_(True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"[INFO] 开始基于 {len(dataset)} 条样本统计 LoRA rank 重要性...")

    for batch in tqdm(loader, desc="Collecting gradients for importance"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 每个 batch 前先清空梯度
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss

        # 一般不会出现 NaN/Inf，如果出现则跳过该 batch
        if not torch.isfinite(loss):
            print(f"[WARN] loss 非有限值（{loss.item()}），跳过该 batch")
            continue

        loss.backward()

        # 对每个适配器、每个 rank 计算并累加 |w * grad|
        for name, params in lora_adapters.items():
            A = params["A"]
            B = params["B"]
            if A.grad is None or B.grad is None:
                # 理论上不会发生，仅做健壮性保护
                continue

            grad_A = A.grad
            grad_B = B.grad

            # 当前 peft 的 LoRA 形状约定：
            #   A: (r, in_features)
            #   B: (out_features, r)
            # 因此 rank = A.shape[0] = B.shape[1]
            r_A = A.shape[0]
            r_B = B.shape[1]
            if r_A != r_B:
                raise RuntimeError(
                    f"LoRA 适配器 {name} 的 A/B rank 维度不一致: A.shape={A.shape}, B.shape={B.shape}"
                )

            # 对每个 rank k，累加当前 batch 的 mean(|w * grad|)
            # 为了节省显存，分两步张量运算再在 CPU 上累加
            with torch.no_grad():
                # A 部分：shape (r, in_features)，在 dim=1 上取均值得到 [r]
                score_A = (A * grad_A).abs().mean(dim=1).detach().cpu()  # [r]
                # B 部分：shape (out_features, r)，在 dim=0 上取均值得到 [r]
                score_B = (B * grad_B).abs().mean(dim=0).detach().cpu()  # [r]
                score = score_A + score_B  # [r]

            importance[name] += score.to(importance[name].dtype)

        # 每个 batch 用完梯度后释放，避免显存累积
        for params in lora_adapters.values():
            if params["A"].grad is not None:
                params["A"].grad = None
            if params["B"].grad is not None:
                params["B"].grad = None

    # 最终 importance[name] 中每一维对应该适配器内某个 rank 的总重要性得分
    print("[INFO] LoRA rank 重要性统计完成。")
    return importance


def apply_pruning(
    lora_adapters: Dict[str, Dict[str, torch.nn.Parameter]],
    importance: Dict[str, torch.Tensor],
    prune_ratio: float,
):
    """
    根据重要性得分对 LoRA rank 做一次性全局剪枝：
    - 收集所有适配器内所有 rank 的 (adapter_name, rank_idx, score) 三元组；
    - 按 score 从小到大排序；
    - 选择前 K = floor(prune_ratio * total_ranks) 个 rank 作为剪枝目标；
    - 对这些 rank，将对应的 A[:,k] 与 B[k,:] 原地置零。

    注意：
    - 置零不会改变 LoRA 的 rank 配置，只是让这些 rank 对前向几乎不再贡献；
    - 这样可以最大限度兼容现有加载与推理脚本；
    - 若想要进一步物理删除这些列/行以获得潜在吞吐提升，可以在此基础上扩展。
    """
    if not (0.0 < prune_ratio < 1.0):
        raise ValueError(f"prune_ratio 必须在 (0,1) 之间，目前为 {prune_ratio}")

    triplets: List[Tuple[str, int, float]] = []
    for name, scores in importance.items():
        scores_cpu = scores.detach().cpu().numpy().tolist()
        for k, s in enumerate(scores_cpu):
            triplets.append((name, k, float(s)))

    total_ranks = len(triplets)
    if total_ranks == 0:
        raise RuntimeError("没有可剪枝的 LoRA rank（total_ranks=0）")

    # 计算要剪掉的 rank 数量，至少为 1
    K = int(total_ranks * prune_ratio)
    if K < 1:
        K = 1
    if K >= total_ranks:
        K = total_ranks - 1

    # 按 score 从小到大排序，得分越小越不重要，优先被剪
    triplets.sort(key=lambda x: x[2])
    to_prune = triplets[:K]

    print(
        f"[INFO] 总 LoRA rank 数量: {total_ranks}, "
        f"计划剪掉 {K} 个 (~{K/total_ranks:.2%})"
    )

    # 为了便于检查，统计每个适配器被剪掉的 rank 数
    per_adapter_count: Dict[str, int] = {}
    for name, k, score in to_prune:
        per_adapter_count[name] = per_adapter_count.get(name, 0) + 1

    print("[INFO] 各适配器被剪掉的 rank 数量（前若干条）：")
    for i, (name, cnt) in enumerate(per_adapter_count.items()):
        print(f"  - {name}: {cnt}")
        if i >= 10:
            print("  ... 其余适配器省略打印")
            break

    # 实际剪枝操作：原地将对应 rank 的 A[:,k] 与 B[k,:] 置零
    with torch.no_grad():
        for name, k, score in to_prune:
            params = lora_adapters[name]
            A = params["A"]
            B = params["B"]
            if k >= A.shape[1] or k >= B.shape[0]:
                # 极端情况下尺寸不对齐（理论上不该发生），跳过以免报错
                print(
                    f"[WARN] 适配器 {name} 的 rank 索引 {k} 超出范围，"
                    f"A.shape={A.shape}, B.shape={B.shape}，跳过该 rank"
                )
                continue

            A[:, k] = 0.0
            B[k, :] = 0.0

    print("[INFO] LoRA rank 级剪枝完成（指定 rank 已被置零）。")


def main():
    args = parse_args()

    print("=" * 60)
    print("ScamNet Explain 学生 LoRA rank 级剪枝（第一阶段·低风险版）")
    print("=" * 60)

    if not os.path.exists(args.student_lora):
        raise FileNotFoundError(f"待剪枝 Explain LoRA 不存在: {args.student_lora}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"剪枝数据集不存在: {args.data_path}")

    print(f"[CONFIG] student_base   = {args.student_base}")
    print(f"[CONFIG] student_lora   = {args.student_lora}")
    print(f"[CONFIG] data_path      = {args.data_path}")
    print(f"[CONFIG] output_dir     = {args.output_dir}")
    print(f"[CONFIG] prune_ratio    = {args.prune_ratio}")
    print(f"[CONFIG] max_samples    = {args.max_prune_samples}")
    print(f"[CONFIG] batch_size     = {args.batch_size}")
    print(f"[CONFIG] max_length     = {args.max_length}")
    print(f"[CONFIG] max_input_char = {args.max_input_chars}")

    examples = load_explain_data(args.data_path, args.max_prune_samples)

    print("\n[1/3] 加载 Mistral-7B 基座与 Explain LoRA...")
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
        # 为了在收集梯度时更稳定，关闭 KV cache
        base.config.use_cache = False

    student = PeftModel.from_pretrained(base, args.student_lora)
    student.eval()  # compute_rank_importance 内部会切到 train 模式
    device = student.device
    print(f"[INFO] 学生模型已加载，device = {device}")

    print("\n[2/3] 构造剪枝用 ExplainPruneDataset...")
    prune_dataset = ExplainPruneDataset(
        examples,
        tokenizer,
        max_length=args.max_length,
        max_input_chars=args.max_input_chars,
    )
    print(f"[INFO] 剪枝 Dataset 大小: {len(prune_dataset)}")

    print("\n[3/3] 收集 LoRA A/B 参数，并统计各 rank 重要性...")
    lora_adapters = collect_lora_rank_handles(student)

    # 将模型移动到正确设备（PeftModel 已经根据 device_map 放在合适 GPU 上）
    student.to(device)

    # 统计每个 LoRA rank 的 |w * grad| 重要性
    importance = compute_rank_importance(
        student, prune_dataset, lora_adapters, batch_size=args.batch_size
    )

    # 根据重要性做一次性全局剪枝（按 rank 置零）
    apply_pruning(lora_adapters, importance, prune_ratio=args.prune_ratio)

    print("\n保存剪枝后的 Explain LoRA...")
    os.makedirs(args.output_dir, exist_ok=True)
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] 剪枝后 LoRA 已保存到: {args.output_dir}")

    print("\n后续推荐步骤：")
    print("1) 使用 distillation_explain_train.py 在剪枝后 LoRA 上做 2–3 epoch 小学习率微调，例如：")
    print(
        "   python distillation_explain_train.py \\"
        "\n       --student_base {base} \\"
        "\n       --student_lora_init {pruned} \\"
        "\n       --data_path data/dataset_explainable_200.json \\"
        "\n       --output_dir outputs/decouple/explain_adapter_pruned_ft \\"
        "\n       --epochs 3 --lr 1e-5".format(
            base=args.student_base, pruned=args.output_dir
        )
    )
    print("2) 用 06_inference/run_benchmark.py --student-explain-lora 指向新 LoRA，")
    print("   对比剪枝前后解释结构/质量与解释阶段耗时。")

    print("\n" + "=" * 60)
    print("LoRA rank 级剪枝完成（第一阶段）。")
    print("=" * 60)


if __name__ == "__main__":
    main()

