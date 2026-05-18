#!/usr/bin/env python3
"""主动学习子集选择（kmeans / top_u / random）。仅供学术防御研究，禁止用于非法用途。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="主动学习：从对抗池选子集")
    p.add_argument("--scores-json", type=Path, required=True)
    p.add_argument("--pool-json", type=Path, required=True)
    p.add_argument("--out-indices", type=Path, required=True)
    p.add_argument("--out-subset-json", type=Path, required=True)
    p.add_argument("--mode", choices=["kmeans", "top_u", "random"], default="kmeans")
    p.add_argument("--k-clusters", type=int, default=50)
    p.add_argument("--budget", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--subset-fields",
        type=str,
        default="input,output",
        help="写入子集 JSON 时保留的字段（逗号分隔）；默认只保留 input/output 供训练",
    )
    return p.parse_args()


def _load_pool(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("pool-json 应为 list")
    return data


def _subset_item(raw: dict, fields: List[str]) -> dict:
    out: Dict[str, Any] = {}
    for k in fields:
        k = k.strip()
        if k in raw:
            out[k] = raw[k]
    if "input" not in out or "output" not in out:
        raise KeyError("子集至少需包含 pool 中的 input 与 output")
    return out


def select_kmeans(
    scores: List[dict],
    k: int,
    budget: int,
    seed: int,
) -> List[int]:
    import numpy as np
    from sklearn.cluster import KMeans

    indices = [int(s["index"]) for s in scores]
    if len(set(indices)) != len(indices):
        raise ValueError("scores 中存在重复 index")

    X_list = []
    for s in scores:
        if "feat" not in s:
            raise ValueError("kmeans 模式需要 scores 中含 feat（打分时不加 --no-feat）")
        X_list.append(s["feat"])
    X = np.asarray(X_list, dtype=np.float32)

    n_clusters = min(k, len(scores))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X)

    by_cluster: Dict[int, List[dict]] = {}
    for s, lab in zip(scores, labels):
        by_cluster.setdefault(int(lab), []).append(s)

    queues: List[List[dict]] = []
    for lab in sorted(by_cluster.keys()):
        q = sorted(by_cluster[lab], key=lambda x: float(x["U"]), reverse=True)
        queues.append(q)

    chosen: List[int] = []
    ptr = [0] * len(queues)

    def any_left() -> bool:
        return any(ptr[i] < len(queues[i]) for i in range(len(queues)))

    while len(chosen) < budget and any_left():
        for i in range(len(queues)):
            if len(chosen) >= budget:
                break
            if ptr[i] < len(queues[i]):
                idx = int(queues[i][ptr[i]]["index"])
                ptr[i] += 1
                chosen.append(idx)

    if len(chosen) < budget:
        seen = set(chosen)
        for s in sorted(scores, key=lambda x: float(x["U"]), reverse=True):
            if len(chosen) >= budget:
                break
            idx = int(s["index"])
            if idx not in seen:
                seen.add(idx)
                chosen.append(idx)

    return chosen[:budget]


def select_top_u(scores: List[dict], budget: int) -> List[int]:
    sorted_scores = sorted(scores, key=lambda x: float(x["U"]), reverse=True)
    return [int(s["index"]) for s in sorted_scores[:budget]]


def select_random(n_pool: int, budget: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    idxs = list(range(n_pool))
    rng.shuffle(idxs)
    return sorted(idxs[:budget])


def main() -> None:
    args = parse_args()
    pool = _load_pool(args.pool_json)
    with open(args.scores_json, "r", encoding="utf-8") as f:
        pack = json.load(f)
    scores: List[dict] = pack.get("scores") or []
    if not scores:
        print("[ERROR] scores-json 中无 scores 列表", file=sys.stderr)
        sys.exit(1)

    n_pool = len(pool)
    if args.mode == "random":
        selected = select_random(n_pool, min(args.budget, n_pool), args.seed)
    elif args.mode == "top_u":
        selected = select_top_u(scores, min(args.budget, len(scores)))
    else:
        if len(scores) != n_pool:
            print(
                f"[WARN] pool 条数 ({n_pool}) 与 scores 条数 ({len(scores)}) 不一致；"
                "请使用同一对抗池与未截断的打分结果。",
                file=sys.stderr,
            )
        selected = select_kmeans(scores, args.k_clusters, min(args.budget, len(scores)), args.seed)

    fields = [x.strip() for x in args.subset_fields.split(",") if x.strip()]
    subset = []
    for idx in selected:
        if idx < 0 or idx >= n_pool:
            print(f"[ERROR] 非法 index: {idx}（pool 长度 {n_pool}）", file=sys.stderr)
            sys.exit(1)
        subset.append(_subset_item(pool[idx], fields))

    args.out_indices.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "pool_json": str(args.pool_json.resolve()),
        "scores_json": str(args.scores_json.resolve()),
        "mode": args.mode,
        "k_clusters": args.k_clusters,
        "budget": args.budget,
        "selected_count": len(selected),
        "indices": selected,
        "seed": args.seed,
    }
    with open(args.out_indices, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(args.out_subset_json, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"[OK] 选中 {len(selected)} 条 -> {args.out_indices.name} / {args.out_subset_json.name}")


if __name__ == "__main__":
    main()
