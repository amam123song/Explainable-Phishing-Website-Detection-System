# 数据目录

将以下 JSON 数据集放入本目录（不随仓库分发，需自行准备或运行 `01_data/` 脚本生成）：

| 文件 | 说明 |
|------|------|
| `dataset_scamnet_5000.json` | 训练集（5869 条结构化样本） |
| `dataset_explainable_200.json` | 专家解释子集（200 条） |
| `dataset_test_strict.json` | 独立测试集（3000 条） |

路径常量见 `config/paths.py`。

**注意**：本仓库不提供预训练权重，完成训练后 LoRA 默认保存在 `outputs/decouple/` 与 `outputs/expert/`，见根目录 `README.md`。
