#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
URLNet (Character-level CNN) 评估脚本

功能：
1) 从 ScamNet 风格 JSON 数据集中抽取 URL 与标签进行测试
2) 支持通过参数选择数据集路径
3) 支持通过参数截断到固定样本数量
4) 可选加载训练好的 checkpoint（若不加载则用随机初始化模型，仅用于流程联调）
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from urlnet_charcnn import (
    CharTokenizer,
    CharVocab,
    URLDataset,
    URLNetCharCNN,
    evaluate,
    load_urls_labels_from_scamnet_json,
    set_seed,
)


def maybe_truncate(
    urls: List[str],
    labels: List[int],
    max_samples: int,
    seed: int = 42,
    shuffle_before_truncate: bool = False,
) -> Tuple[List[str], List[int]]:
    """按需截断样本数量。可选先打乱再截断。"""
    if max_samples <= 0 or max_samples >= len(urls):
        return urls, labels

    idxs = list(range(len(urls)))
    if shuffle_before_truncate:
        rnd = random.Random(seed)
        rnd.shuffle(idxs)
    idxs = idxs[:max_samples]

    sub_urls = [urls[i] for i in idxs]
    sub_labels = [labels[i] for i in idxs]
    return sub_urls, sub_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate URLNet char-level CNN on ScamNet-style JSON dataset")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(REPO_ROOT / "data/dataset_test_strict_10000.json"),
        help="评估数据集 JSON 路径（可自由切换数据集）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="仅评估前 N 条样本（0 表示不截断）",
    )
    parser.add_argument(
        "--shuffle_before_truncate",
        action="store_true",
        help="截断前先随机打乱（默认不打乱，直接取前 N 条）",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="09_baselines/urlnet/urlnet_charcnn_best.pth",
        help="模型权重路径（.pt/.pth）；默认内置为训练脚本保存的最佳模型",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="评估 batch size")
    parser.add_argument("--max_len", type=int, default=200, help="URL 最大字符长度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")
    print(f"[Info] dataset_path = {args.dataset_path}")

    # 1) 读取并解析数据集（从 input 中抽 URL，从 output 中抽标签）
    urls, labels = load_urls_labels_from_scamnet_json(args.dataset_path)
    print(f"[Info] parsed samples = {len(urls)}")

    # 2) 可选截断
    urls, labels = maybe_truncate(
        urls,
        labels,
        max_samples=args.max_samples,
        seed=args.seed,
        shuffle_before_truncate=args.shuffle_before_truncate,
    )
    print(f"[Info] eval samples(after truncate) = {len(urls)}")

    # 3) 构建 tokenizer / dataset / dataloader
    vocab = CharVocab.build_default()
    tokenizer = CharTokenizer(vocab=vocab, max_len=args.max_len)
    eval_ds = URLDataset(urls, labels, tokenizer)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 4) 构建模型并按需加载权重
    model = URLNetCharCNN(
        vocab_size=vocab.size,
        embed_dim=32,
        num_classes=2,
        kernel_sizes=(3, 4, 5, 6),
        out_channels=128,
        dropout_p=0.5,
        pad_id=vocab.pad_id,
    ).to(device)

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        # 兼容两种保存格式：
        # 1) torch.save(model.state_dict(), path)
        # 2) torch.save({"model_state_dict": ...}, path)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        print(f"[Info] loaded checkpoint: {args.checkpoint_path}")
    else:
        print(f"[Warn] checkpoint 不存在或为空：{args.checkpoint_path}，当前使用随机初始化模型，仅用于流程联调。")

    # 5) 执行评估（含单样本平均推理耗时）
    metrics = evaluate(model, eval_loader, device, measure_inference_time=True, warmup_batches=1)

    print(
        "[Result] "
        f"loss={metrics['loss']:.6f} | "
        f"acc={metrics['accuracy']:.6f} | "
        f"precision={metrics['precision']:.6f} | "
        f"recall={metrics['recall']:.6f} | "
        f"f1={metrics['f1']:.6f} | "
        f"tp={metrics['tp']} | "
        f"fp={metrics['fp']} | "
        f"tn={metrics['tn']} | "
        f"fn={metrics['fn']} | "
        f"bad_cases={metrics['bad_cases']} | "
        f"infer_ms_per_sample={metrics['inference_ms_per_sample']:.6f}"
    )


if __name__ == "__main__":
    main()
