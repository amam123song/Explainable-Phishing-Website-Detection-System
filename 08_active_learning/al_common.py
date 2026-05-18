"""主动学习共用工具（4bit 加载、prompt 构造）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from config.paths import hf_local_files_only

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_causal_lm_4bit(
    model_id: str,
    *,
    local_files_only: bool | None = None,
    attn_implementation: str = "sdpa",
):
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

    if local_files_only is None:
        local_files_only = hf_local_files_only(model_id)
    config = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
    load_kwargs: dict = {
        "device_map": "auto",
        "local_files_only": local_files_only,
        "attn_implementation": attn_implementation,
    }
    if getattr(config, "quantization_config", None) is None:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as e:
        if attn_implementation == "sdpa":
            load_kwargs["attn_implementation"] = "eager"
            return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        raise e


def build_pred_prompt(inp: str, max_prompt_chars: int) -> str:
    """与 distill_step1 / run_benchmark dual_cls 对齐：input 截断后以 # Pred: 结尾。"""
    p = inp[:max_prompt_chars].replace("# Pred:", "").strip()
    if "# Information:" not in p:
        p = f"# Information:\n{p}"
    return f"{p}\n\n# Pred:\n"


def label_from_output(output_text: str) -> int:
    head = (output_text or "").split("\n", 1)[0].strip().lower()
    if "scam" in head:
        return 1
    if "legit" in head:
        return 0
    return 0
