# 对抗样本生成（研究用途）

## 免责声明

**本目录下的脚本仅用于学术防御研究**：在受控、合规的实验环境中评估钓鱼检测模型的鲁棒性。

- 不得将本代码或生成结果用于**构造、部署或传播**真实钓鱼网站、钓鱼邮件或任何欺诈活动。
- 不得对未经授权的第三方网站或系统进行攻击测试。
- 生成物默认写入您指定的 `--output_json` 路径（通常为 `results/`），**不会**自动保存到本仓库内；请勿将对抗 JSON 提交到公开版本库。

使用本代码即表示您同意遵守当地法律法规及所在机构的研究伦理规范。

---

## 脚本说明

| 脚本 | 类型 | 说明 |
|------|------|------|
| `generate_blackbox_heuristic.py` | 黑盒启发式 | 对 **scam** 样本施加 URL/正文/视觉扰动；**legit** 原样保留 |
| `generate_whitebox_distilbert.py` | 白盒（DistilBERT） | 基于梯度的一阶词替换，有步数与扰动比例上限 |
| `generate_whitebox_urlnet.py` | 白盒（URLNet） | 在 URL 上拼接良性风格子域前缀，截断至 `max_len` |

所有脚本均需通过命令行显式传入输入/输出路径，**无默认无参执行**。

---

## 安全约束（实现层面）

### 黑盒（`generate_blackbox_heuristic.py`）

- 仅修改标签为 **scam** 的样本；合法样本不扰动。
- URL 变异前后均经 `is_parseable_url()` 校验（基于 `urlparse`）；无法解析或变异后非法的 URL **保持原样**。
- 正文扰动为有限规则替换与同义词表，不生成全新钓鱼文案。
- 可通过 `--limit` 限制处理条数，便于小规模消融。

### 白盒 DistilBERT（`generate_whitebox_distilbert.py`）

- `--max_perturb_ratio`（默认 `0.05`）：相对输入序列长度的最大替换步数比例。
- `--max_steps`（默认 `30`）：硬上限步数。
- 仅对 **scam** 样本尝试攻击；**legit** 原样输出。

### 白盒 URLNet（`generate_whitebox_urlnet.py`）

- `--max_len`（默认 `200`）：URL 字符序列上限，与 URLNet 训练一致。
- 攻击失败时保留原始 URL；仅修改 `## URL:` 字段，不改动正文结构。

---

## 命令示例

在仓库根目录执行（请先准备 `data/dataset_test_strict.json` 等测试集）：

### 1. 黑盒启发式

```bash
python 07_adversarial/generate_blackbox_heuristic.py \
  --input_json data/dataset_test_strict.json \
  --output_json results/adv_blackbox_heuristic.json \
  --limit 2100 \
  --seed 42
```

### 2. 白盒 DistilBERT

需先训练基线模型，例如产出 `09_baselines/distilbert/distilbert_finetuned_phishing`：

```bash
python 07_adversarial/generate_whitebox_distilbert.py \
  --input_json data/dataset_test_strict.json \
  --model_dir 09_baselines/distilbert/distilbert_finetuned_phishing \
  --output_json results/adv_whitebox_distilbert.json \
  --max_perturb_ratio 0.05 \
  --max_steps 30 \
  --first_n 3000
```

### 3. 白盒 URLNet

需先训练 URLNet  checkpoint，例如 `09_baselines/urlnet/urlnet_charcnn_best.pth`：

```bash
python 07_adversarial/generate_whitebox_urlnet.py \
  --input_json data/dataset_test_strict.json \
  --output_json results/adv_whitebox_urlnet.json \
  --checkpoint_path 09_baselines/urlnet/urlnet_charcnn_best.pth \
  --urlnet_repo_dir 09_baselines/urlnet \
  --max_len 200
```

生成结果可用于 `06_inference/run_benchmark.py` 复测，或供 `08_active_learning/` 构建对抗池（见 `config/paths.py` 中 `DATASET_ADV_POOL`）。

---

## 输出约定

- 输出 JSON 为 `list[dict]`，保留原 `input` / `output` 结构。
- 黑盒样本可能含 `attack_type` 字段；白盒样本可能含 `is_success`、`adv_text` / `adv_url` 等调试字段。
- 请将输出目录加入 `.gitignore`（仓库已忽略 `results/`），勿上传未脱敏的真实 URL 或页面内容。
