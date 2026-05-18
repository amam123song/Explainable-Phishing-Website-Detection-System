# 解释 LoRA（θ_e）训练与压缩链路

目标产出：`outputs/decouple/explain_adapter`（推理时 `--student-explain-lora`）。

## 主线训练

1. 先完成分类 LoRA：`outputs/decouple/cls_adapter`（见 `CLASS_MODEL_LINEAGE.md`）。

2. **解释蒸馏**  
   ```bash
   python 04_decouple/explanation/distillation_explain_train.py \
     --student_lora_init outputs/decouple/cls_adapter \
     --data_path data/dataset_explainable_200.json \
     --output_dir outputs/decouple/explain_adapter
   ```
   默认：3 epoch，lr=5e-6，max_length=1024。

## 可选压缩（按顺序选用）

| 脚本 | 输入 LoRA（默认） | 输出目录（默认） |
|------|-------------------|------------------|
| `prune_output_explain_lora_rank.py` | `outputs/decouple/explain_adapter` | `outputs/decouple/explain_adapter_pruned_rank30` |
| 再次 `distillation_explain_train.py` | 剪枝后目录 | `..._pruned_ft` |
| `shrink_output_explain_lora_rank.py` | 剪枝+微调后 | `..._pruned_rankshrink` |
| `prune_output_explain_lora_layers.py` | 收缩后 | `outputs/decouple/explain_layerpruned` |

剪枝后需对 `outputs/decouple/explain_adapter` 或子目录重新微调时，将 `--student_lora_init` / `--student_lora` 指向上一步产出。

## 评测

```bash
python 06_inference/run_benchmark.py --run dual_full \
  --student-cls-lora outputs/decouple/cls_adapter \
  --student-explain-lora outputs/decouple/explain_adapter \
  --data-path data/dataset_test_strict.json
```

本仓库不附带预训练权重；上述路径均为自行训练后的默认约定。
