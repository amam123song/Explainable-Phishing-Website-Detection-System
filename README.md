# 钓鱼网站检测系统 — 开源代码

基于大语言模型的可解释钓鱼网站检测：专家训练 → 知识蒸馏 → 双 LoRA 功能解耦 → 对抗鲁棒演化。

**本代码仅供学术研究与防御模型评估使用，不得用于构造、部署或传播真实钓鱼网站。**

## 预训练权重说明

**出于安全研究与数据保护考虑，本仓库不直接提供任何预训练 LoRA 或全量微调权重。**

我们提供完整的训练入口脚本与论文对齐的超参数配置，便于在合规数据上自行训练等价模型。训练完成后默认产出见 [`05_weights/README.md`](05_weights/README.md) 与 [`config/paths.py`](config/paths.py)。

## 目录结构

```
开源代码/
├── README.md
├── requirements.txt
├── LICENSE
├── config/paths.py           路径常量与环境变量
├── 01_data/                  数据集构建
├── 02_expert/                专家模型（Llama-3-8B 两阶段 LoRA）
├── 03_distill_monolithic/    单体学生蒸馏（多任务干涉基线）
├── 04_decouple/              功能解耦双 LoRA（分类 θ_c + 解释 θ_e）
├── 05_weights/               权重说明（不含 checkpoint）
├── 06_inference/             统一推理评测
├── 07_adversarial/           对抗样本生成
├── 08_active_learning/       主动学习采样
├── 09_baselines/             对比基线
└── 10_evaluation/            解释质量 LLM-as-a-Judge
```

## 环境安装

```bash
cd <REPO_ROOT>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 按需填写 API Key
```

需从 HuggingFace 获取 **Mistral-7B-Instruct-v0.2**（学生/推理）及 **Meta-Llama-3-8B-Instruct**（专家/教师，需 Meta 授权）。首次运行会自动下载（若未设置 `HF_HUB_OFFLINE=1`）。

## 数据准备

将 JSON 放入 `data/`（不随仓库分发），格式为 `list[{"input": "...", "output": "..."}]`：

| 文件 | 说明 |
|------|------|
| `dataset_scamnet_5000.json` | 分类训练集 |
| `dataset_explainable_200.json` | 解释微调子集 |
| `dataset_test_strict.json` | 独立测试集 |

可由 `01_data/process_scamnet_data.py` 从原始网页目录生成；解释子集可用 `01_data/generate_explainable_data_with_api.py`（需 `OPENAI_API_KEY`）。

## 完整训练流程（复现论文模型）

以下超参与 `02_expert/` 脚本内注释及论文 Table 4 对齐。

### 1. 专家模型（教师）

```bash
# Stage1：分类 SFT，2 epoch，lr=1e-4，LoRA r=16
python 02_expert/train_step1_classification.py
# 产出：outputs/expert/scamnet_step1_model

# Stage2：解释 SFT，2 epoch，lr=1e-5，LoRA r=32
python 02_expert/train_step2_explainable.py
# 产出：outputs/expert/scamnet_final_model
```

### 2. 解耦双 LoRA（推荐主线）

```bash
# （可选）离线教师二类软标签，省显存
python 04_decouple/classification/generate_teacher_soft2labels.py \
  --data_path data/dataset_scamnet_5000.json \
  --out_path data/dataset_scamnet_5000_soft2_T4.json \
  --teacher_lora_path outputs/expert/scamnet_final_model

# 分类学生 θ_c：T=4，alpha=0.7，2 epoch，lr=1e-4，LoRA r=8
python 04_decouple/classification/distill_step1_classification_mistral_student.py \
  --data_path data/dataset_scamnet_5000.json \
  --teacher_lora_path outputs/expert/scamnet_final_model \
  --output_dir outputs/decouple/cls_adapter

# 解释学生 θ_e：3 epoch，lr=5e-6
python 04_decouple/explanation/distillation_explain_train.py \
  --student_lora_init outputs/decouple/cls_adapter \
  --data_path data/dataset_explainable_200.json \
  --output_dir outputs/decouple/explain_adapter
```

可选剪枝：`04_decouple/explanation/prune_*.py`、`shrink_*.py`（见 `EXPLAIN_MODEL_LINEAGE.md`）。

### 3. 单体学生基线（第 3 章干涉实验）

```bash
python 03_distill_monolithic/distill_step1_classification.py \
  --teacher_lora outputs/expert/scamnet_final_model \
  --output_dir outputs/monolithic/step1_cls

python 03_distill_monolithic/distill_step2_explainable.py \
  --student_lora_init outputs/monolithic/step1_cls \
  --output_dir outputs/monolithic/step2_explain
```

### 4. 推理评测

```bash
python 06_inference/run_benchmark.py \
  --run dual_cls \
  --data-path data/dataset_test_strict.json \
  --student-cls-lora outputs/decouple/cls_adapter \
  --max-samples 100 \
  --no-save-explanations
```

`dual_full` 需额外指定 `--student-explain-lora outputs/decouple/explain_adapter`。

### 5. 主动学习 + 单侧增量演化（第 5 章）

```bash
# 对抗池不确定性打分 → 选子集 → 合并训练集 → 打印分类 LoRA 复训命令
python 08_active_learning/al_score_dual_models.py \
  --pool-json data/dataset_adversarial_pool.json \
  --out-json results/al_scores.json

python 08_active_learning/al_select_subset.py \
  --scores-json results/al_scores.json \
  --pool-json data/dataset_adversarial_pool.json \
  --out-indices results/al_selected_indices.json \
  --out-subset-json results/al_subset_500.json \
  --budget 500

python 08_active_learning/al_build_train_mixed.py \
  --adv-subset-json results/al_subset_500.json \
  --out-json data/dataset_scamnet_plus_adv.json

python 08_active_learning/al_print_finetune_cmd.py \
  --mixed-json data/dataset_scamnet_plus_adv.json
```

### 6. 动态路由阈值 τ 扫描（第 4 章）

```bash
python 06_inference/run_dynamic_threshold_experiment.py \
  --clean-data-path data/dataset_test_strict.json \
  --adv-data-path data/dataset_adversarial_pool.json \
  --student-cls-lora outputs/decouple/cls_adapter \
  --student-explain-lora outputs/decouple/explain_adapter
```

输出：`results/dynamic_threshold/dynamic_threshold_<timestamp>/threshold_metrics.csv`（解释覆盖率、延迟、漏解释风险等）。

## 各模块速查

| 目录 | 核心脚本 |
|------|----------|
| `01_data/` | `process_scamnet_data.py`, `generate_explainable_data_with_api.py` |
| `02_expert/` | `train_step1_classification.py`, `train_step2_explainable.py` |
| `04_decouple/` | `distill_step1_classification_mistral_student.py`, `distillation_explain_train.py` |
| `06_inference/` | `run_benchmark.py`, `run_dynamic_threshold_experiment.py`（τ 扫描） |
| `08_active_learning/` | `al_score_dual_models.py`, `al_select_subset.py`, `al_build_train_mixed.py`, `al_print_finetune_cmd.py` |
| `09_baselines/` | RF / URLNet / BERT / DistilBERT / GPT 子目录 |
| `10_evaluation/` | `score_explanations.py` |

权重溯源：`04_decouple/classification/CLASS_MODEL_LINEAGE.md`、`04_decouple/explanation/EXPLAIN_MODEL_LINEAGE.md`。

## 项目声明

- 项目名称：可解释钓鱼网站检测系统
- 项目作者：Song Xingyao
- 作者单位：暨南大学网络空间安全学院
- 开发语言：python
- 框架：PyTorch
- 核心技术：大语言模型微调、知识蒸馏、可解释钓鱼检测、对抗鲁棒与主动学习

## 安全与隐私

- API Key 仅通过环境变量配置，见 [`.env.example`](.env.example) 与 [`SECURITY.md`](SECURITY.md)。
- 对抗样本生成脚本**仅用于学术防御研究**，须通过命令行显式传入路径；约束与示例见 [`07_adversarial/README.md`](07_adversarial/README.md)。
- 勿将 `.env`、原始爬取数据、对抗样本 JSON、自行训练的 `outputs/` 提交到公开仓库。

## 许可证

[MIT License](LICENSE)
