# 分类 LoRA（θ_c）训练链路

目标产出：`outputs/decouple/cls_adapter`（推理时 `--student-cls-lora` 指向此目录）。

## 推荐流程

1. **专家教师**（`02_expert/`）  
   - `train_step1_classification.py` → `outputs/expert/scamnet_step1_model`  
   - `train_step2_explainable.py` → `outputs/expert/scamnet_final_model`

2. **（可选）离线二类软标签**  
   ```bash
   python 04_decouple/classification/generate_teacher_soft2labels.py \
     --data_path data/dataset_scamnet_5000.json \
     --out_path data/dataset_scamnet_5000_soft2_T4.json \
     --teacher_lora_path outputs/expert/scamnet_final_model
   ```

3. **二类 logits 蒸馏（Mistral 学生）**  
   ```bash
   python 04_decouple/classification/distill_step1_classification_mistral_student.py \
     --data_path data/dataset_scamnet_5000.json \
     --teacher_lora_path outputs/expert/scamnet_final_model \
     --student_base mistralai/Mistral-7B-Instruct-v0.2 \
     --output_dir outputs/decouple/cls_adapter
   ```
   若使用软标签 JSON，增加 `--data_path data/dataset_scamnet_5000_soft2_T4.json --expect_soft2_field teacher_soft2`。

## 默认超参（脚本内）

| 参数 | 值 |
|------|-----|
| temperature T | 4.0 |
| alpha（硬/软） | 0.7 |
| epochs | 2 |
| lr | 1e-4 |
| LoRA r | 8 |
| batch × grad_accum | 1 × 8 |

## 旧版脚本

`distill_step1_classification_legacy.py`：全词表 KL 蒸馏（需 teacher/student 词表一致），默认输出 `outputs/decouple/cls_adapter`，仅作对照。

## 评测

```bash
python 06_inference/run_benchmark.py --run dual_cls \
  --student-cls-lora outputs/decouple/cls_adapter \
  --data-path data/dataset_test_strict.json
```
