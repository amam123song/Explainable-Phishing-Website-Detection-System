# 预训练权重说明

**本仓库不随代码分发任何 LoRA / 全量微调权重。**

出于安全研究与数据保护考虑，我们不直接提供可用于钓鱼检测与对抗样本生成的预训练权重。请使用仓库内完整的训练脚本与论文对齐的超参数，在自有合规数据上自行训练等价模型。

训练完成后，默认产出路径为：

| 组件 | 默认输出目录 |
|------|----------------|
| 专家 Stage1（分类） | `outputs/expert/scamnet_step1_model` |
| 专家 Stage2（解释） | `outputs/expert/scamnet_final_model` |
| 解耦分类 LoRA θ_c | `outputs/decouple/cls_adapter` |
| 解耦解释 LoRA θ_e | `outputs/decouple/explain_adapter` |
| 单体学生 Step1 / Step2 | `outputs/monolithic/step1_cls`、`outputs/monolithic/step2_explain` |

推理与评测时通过 `--student-cls-lora`、`--student-explain-lora`、`--expert-model` 等参数指向上述目录即可。

完整训练流程见仓库根目录 [README.md](../README.md)。
