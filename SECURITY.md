# 安全与隐私说明

## 请勿提交到公开仓库的内容

- `.env` 及任何含 API Key、Token、密码的文件
- 个人机器绝对路径（如 `/root/...`、本机 HuggingFace cache 快照路径）
- 未脱敏的原始爬取网页、用户 URL、真实钓鱼站点数据
- 第三方代理服务的私有 endpoint（已改为环境变量配置）

## 环境变量配置

复制 `.env.example` 为 `.env` 后填写：

| 变量 | 用途 |
|------|------|
| `OPENAI_API_KEY` | 解释生成、GPT 基线、通用 OpenAI 兼容接口 |
| `OPENAI_BASE_URL` | API 基址（默认 `https://api.openai.com/v1`） |
| `JUDGE_API_KEY` | LLM-as-a-Judge 评测（可同 `OPENAI_API_KEY`） |
| `JUDGE_BASE_URL` | 评测 API 基址 |
| `JUDGE_MODEL` | 裁判模型名（默认 `gpt-4o-mini`） |
| `MISTRAL_BASE_MODEL` | Mistral 基座 Hub ID 或本地路径 |
| `LLAMA_EXPERT_*` | 专家模型基座与 LoRA（训练/蒸馏时） |
| `RAW_PHISHING_DIR` / `RAW_LEGIT_DIR` | 原始网页爬取目录 |

## 对抗样本生成

`07_adversarial/` 下脚本仅用于学术防御研究，须通过命令行传入输入/输出路径，详见 [`07_adversarial/README.md`](07_adversarial/README.md)。

## 开源前自检

```bash
cd <REPO_ROOT>
# 不应出现真实 sk- 密钥
rg -n 'sk-[A-Za-z0-9]{20,}' .
# 不应出现本机绝对路径
rg -n '/root/|/home/' .
```

若发现遗漏，请立即轮换已泄露的 API Key。
