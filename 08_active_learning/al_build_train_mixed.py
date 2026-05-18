#!/usr/bin/env python3
"""合并基座训练集与主动学习对抗子集。仅供学术防御研究，禁止用于非法用途。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合并干净训练集与主动学习对抗子集")
    p.add_argument(
        "--base-train",
        type=Path,
        default=REPO_ROOT / "data" / "dataset_scamnet_5000.json",
        help="原始分类训练 JSON（list，含 input/output）",
    )
    p.add_argument("--adv-subset-json", type=Path, required=True, help="al_select_subset.py 产出的子集 JSON")
    p.add_argument("--out-json", type=Path, required=True, help="混合训练集输出路径（建议 data/ 下，勿提交仓库）")
    p.add_argument("--repeat-adv", type=int, default=3, help="对抗样本重复次数")
    p.add_argument("--base-max-samples", type=int, default=None, help="基座训练集最多使用条数（默认全量）")
    p.add_argument("--shuffle-seed", type=int, default=3407)
    return p.parse_args()


def _load_list(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 list JSON: {path}")
    return data


def _normalize_item(it: dict) -> dict:
    if "input" not in it or "output" not in it:
        raise KeyError("每条样本需含 input 与 output")
    return {"input": it["input"], "output": it["output"]}


def main() -> None:
    args = parse_args()
    if not args.base_train.is_file():
        print(f"[ERROR] 找不到基座训练集: {args.base_train}", file=sys.stderr)
        sys.exit(1)
    if not args.adv_subset_json.is_file():
        print(f"[ERROR] 找不到对抗子集: {args.adv_subset_json}", file=sys.stderr)
        sys.exit(1)

    base = [_normalize_item(x) for x in _load_list(args.base_train)]
    if args.base_max_samples is not None:
        base = base[: args.base_max_samples]

    adv = [_normalize_item(x) for x in _load_list(args.adv_subset_json)]
    adv_expanded: List[dict] = []
    for _ in range(max(args.repeat_adv, 1)):
        adv_expanded.extend([dict(x) for x in adv])

    merged = base + adv_expanded
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(merged)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] 基座 {len(base)} 条 + 对抗 {len(adv)}×{args.repeat_adv}={len(adv_expanded)} 条 "
        f"=> 合计 {len(merged)} 条 -> {args.out_json}"
    )


if __name__ == "__main__":
    main()
