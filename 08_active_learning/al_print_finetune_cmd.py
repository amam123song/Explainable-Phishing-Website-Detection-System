#!/usr/bin/env python3
"""打印分类 LoRA 增量复训命令（不执行）。仅供学术防御研究，禁止用于非法用途。"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.paths import MISTRAL_BASE, OUTPUT_DECOUPLE_CLS  # noqa: E402

DEFAULT_DISTILL = (
    REPO_ROOT / "04_decouple" / "classification" / "distill_step1_classification_mistral_student.py"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="打印 Step1 分类训练命令模板（主动学习增量复训）")
    p.add_argument("--mixed-json", type=Path, required=True, help="al_build_train_mixed.py 产出的混合训练 JSON")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "decouple" / "cls_adapter_al_r1",
        help="新 LoRA 输出目录",
    )
    p.add_argument(
        "--distill-script",
        type=Path,
        default=DEFAULT_DISTILL,
        help="分类蒸馏脚本路径",
    )
    p.add_argument(
        "--student-base",
        type=str,
        default=MISTRAL_BASE,
        help="Mistral 基座（Hub ID 或本地目录，可用环境变量 MISTRAL_BASE_MODEL 覆盖）",
    )
    p.add_argument(
        "--init-lora",
        type=Path,
        default=OUTPUT_DECOUPLE_CLS,
        help="当前分类 LoRA（仅作说明；默认蒸馏脚本需改代码才能热启动）",
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5, help="增量轮次常用略低于首轮 1e-4")
    p.add_argument("--lora-r", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mixed = args.mixed_json.resolve()
    out_dir = args.output_dir.resolve()
    distill = args.distill_script.resolve()

    if not mixed.is_file():
        print(f"[ERROR] 混合训练集不存在: {mixed}", file=sys.stderr)
        sys.exit(1)
    if not distill.is_file():
        print(f"[ERROR] 蒸馏脚本不存在: {distill}", file=sys.stderr)
        sys.exit(1)

    cmd = f"""\
cd {REPO_ROOT}
python {distill} \\
  --data_path {mixed} \\
  --output_dir {out_dir} \\
  --student_base {args.student_base} \\
  --epochs {args.epochs} \\
  --lr {args.lr} \\
  --lora_r {args.lora_r}
"""
    print(textwrap.dedent(cmd))
    print(
        f"\n[说明] 将在混合数据上训练得到新 LoRA：{out_dir}\n"
        f"       当前分类 LoRA（参考）: {args.init_lora.resolve()}\n"
        "       若需从旧 LoRA 热启动，请修改蒸馏脚本为 PeftModel.from_pretrained 后再运行上述命令。"
    )


if __name__ == "__main__":
    main()
