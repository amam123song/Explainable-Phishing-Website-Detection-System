#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
URLNet (Character-level CNN) baseline in PyTorch.

特点：
- 字符级字典与Tokenizer（含 <PAD>/<UNK>）
- URL 字符序列：截断/填充到 max_len=200
- 模型：Embedding(32) + 并行 Conv1d(k=3/4/5/6, out=128) + GlobalMaxPool + Dropout + Linear(2)
- 训练：Adam + CrossEntropyLoss
- 评估：Accuracy / Precision / Recall / F1
- 推理：统计“单样本平均推理耗时”（ms / sample）

说明：
- 该脚本提供一个可直接运行的模拟入口（随机生成几十条 URL 与标签），用于证明整体流程无 bug。
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    """尽量保证可复现（CPU/GPU）。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class CharVocab:
    """字符级词表。

    - **PAD**：用于补齐到 max_len
    - **UNK**：未登录字符
    """

    stoi: Dict[str, int]
    itos: List[str]
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    @property
    def size(self) -> int:
        return len(self.itos)

    @staticmethod
    def build_default() -> "CharVocab":
        """构建默认字符表：大小写字母、数字、常见 URL 标点符号等。"""
        # 字母与数字
        letters_lower = "abcdefghijklmnopqrstuvwxyz"
        letters_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        digits = "0123456789"

        # 常见 URL 符号（覆盖你提到的：_, -, /, ., ?, =, & 等）
        # 额外补充一些论文/工程里经常出现的分隔符与转义符号，利于复现/鲁棒性。
        url_punct = "_-/.?=&:%+#@~!*'(),;[]{}|\\^$<>\""

        # 空格一般不出现在 URL 中，但有时数据里会混入；这里也纳入词表便于处理。
        extra = " \t"

        charset = list(dict.fromkeys(list(letters_lower + letters_upper + digits + url_punct + extra)))

        # 特殊符号必须放最前，确保 id 固定
        itos = ["<PAD>", "<UNK>"] + charset
        stoi = {ch: i for i, ch in enumerate(itos)}
        return CharVocab(stoi=stoi, itos=itos)


class CharTokenizer:
    """字符级Tokenizer：URL -> char id 序列（截断/填充）。"""

    def __init__(self, vocab: CharVocab, max_len: int = 200) -> None:
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, url: str) -> torch.LongTensor:
        # 将字符串逐字符映射到 id，未登录字符映射到 UNK
        ids = [self.vocab.stoi.get(ch, self.vocab.unk_id) for ch in url]
        # 截断/填充到固定长度
        if len(ids) >= self.max_len:
            ids = ids[: self.max_len]
        else:
            ids = ids + [self.vocab.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class URLDataset(Dataset):
    """标准 PyTorch Dataset：输入 URL 列表与二分类标签。"""

    def __init__(self, urls: Sequence[str], labels: Sequence[int], tokenizer: CharTokenizer) -> None:
        if len(urls) != len(labels):
            raise ValueError(f"urls 与 labels 长度不一致：{len(urls)} vs {len(labels)}")
        self.urls = list(urls)
        self.labels = list(labels)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        x = self.tokenizer.encode(self.urls[idx])  # (max_len,)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # scalar
        return x, y


class URLNetCharCNN(nn.Module):
    """URLNet（Character-level CNN）实现：并行卷积 + 全局最大池化 + 拼接分类。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        num_classes: int = 2,
        kernel_sizes: Sequence[int] = (3, 4, 5, 6),
        out_channels: int = 128,
        dropout_p: float = 0.5,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=out_channels, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(out_channels * len(kernel_sizes), num_classes)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        x: (B, L) 字符 id
        return: (B, 2) logits
        """
        # Embedding: (B, L, E)
        emb = self.embedding(x)
        # Conv1d 需要 (B, C_in, L)
        emb = emb.transpose(1, 2)  # (B, E, L)

        pooled_feats: List[torch.Tensor] = []
        for conv in self.convs:
            # (B, out, L-k+1)
            h = conv(emb)
            h = F.relu(h)
            # Global Max Pooling over length dimension -> (B, out)
            h = torch.max(h, dim=2).values
            pooled_feats.append(h)

        feat = torch.cat(pooled_feats, dim=1)  # (B, out*4)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits


@torch.no_grad()
def compute_metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """从二分类混淆矩阵计数计算指标（正类=1：钓鱼）。"""
    eps = 1e-12
    acc = (tp + tn) / (tp + fp + fn + tn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    measure_inference_time: bool = True,
    warmup_batches: int = 1,
) -> Dict[str, float]:
    """评估：返回 Accuracy / Precision / Recall / F1，并可统计单样本平均推理耗时（ms）。"""
    model.eval()

    tp = fp = fn = tn = 0
    total_loss = 0.0
    total_samples = 0
    loss_fn = nn.CrossEntropyLoss()

    # 推理耗时统计（按 batch 计时，折算到 per-sample）
    timed_batches = 0
    timed_samples = 0
    total_infer_seconds = 0.0

    # 若验证/测试集 batch 数过少，warmup 会导致完全没有计时样本，这里做自适应处理
    if measure_inference_time:
        try:
            num_batches = len(loader)
        except TypeError:
            num_batches = None
        if num_batches is not None and num_batches <= warmup_batches:
            warmup_batches = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # warmup：避免首个 batch 初始化/缓存影响（尤其是 GPU）
        do_time = measure_inference_time and (batch_idx >= warmup_batches)

        if do_time:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        logits = model(x)

        if do_time:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_infer_seconds += (t1 - t0)
            timed_batches += 1
            timed_samples += x.size(0)

        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        total_samples += x.size(0)

        pred = torch.argmax(logits, dim=1)
        # 统计混淆矩阵：正类=1（钓鱼）
        tp += int(((pred == 1) & (y == 1)).sum().item())
        fp += int(((pred == 1) & (y == 0)).sum().item())
        fn += int(((pred == 0) & (y == 1)).sum().item())
        tn += int(((pred == 0) & (y == 0)).sum().item())

    metrics = compute_metrics_from_counts(tp=tp, fp=fp, fn=fn, tn=tn)
    metrics["loss"] = float(total_loss / max(1, total_samples))
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tn"] = int(tn)
    metrics["bad_cases"] = int(fp + fn)

    if measure_inference_time and timed_samples > 0:
        metrics["inference_ms_per_sample"] = float((total_infer_seconds * 1000.0) / timed_samples)
    else:
        metrics["inference_ms_per_sample"] = float("nan")

    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一个 epoch，返回平均 loss。"""
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_samples += x.size(0)

    return float(total_loss / max(1, total_samples))


def split_train_val(
    urls: Sequence[str],
    labels: Sequence[int],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """简单随机划分训练/验证集。"""
    n = len(urls)
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    n_val = int(n * val_ratio)
    val_idxs = set(idxs[:n_val])

    train_urls, train_labels, val_urls, val_labels = [], [], [], []
    for i in range(n):
        if i in val_idxs:
            val_urls.append(urls[i])
            val_labels.append(labels[i])
        else:
            train_urls.append(urls[i])
            train_labels.append(labels[i])
    return train_urls, train_labels, val_urls, val_labels


def make_fake_urls(n: int = 80, seed: int = 42) -> Tuple[List[str], List[int]]:
    """生成一批可跑通流程的伪造 URL 与二分类标签（仅用于自测）。"""
    rnd = random.Random(seed)

    legit_domains = ["google.com", "wikipedia.org", "github.com", "pytorch.org", "mit.edu", "openai.com"]
    phish_domains = [
        "g00gle-login.com",
        "paypa1-secure.com",
        "micr0soft-verify.net",
        "apple-id-support.xyz",
        "bank-verify-account.top",
        "secure-login-update.cc",
    ]
    paths = ["/", "/login", "/account", "/verify", "/signin", "/security/check", "/oauth/callback"]
    params = ["", "?id=123", "?session=abc", "?redirect=/home", "?token=deadbeef", "?utm_source=email&ref=xx"]

    urls: List[str] = []
    labels: List[int] = []
    for _ in range(n):
        is_phish = rnd.random() < 0.5
        scheme = "https" if rnd.random() < 0.8 else "http"
        sub = ""
        if rnd.random() < 0.35:
            sub = rnd.choice(["www", "m", "login", "secure", "account"]) + "."
        domain = rnd.choice(phish_domains if is_phish else legit_domains)
        path = rnd.choice(paths)
        param = rnd.choice(params)
        url = f"{scheme}://{sub}{domain}{path}{param}"

        # 额外加入一些“噪声特征”：@、多级子域、奇怪端口等（更贴近真实 URL 的分布）
        if is_phish and rnd.random() < 0.25:
            url = url.replace("://", "://user@" if rnd.random() < 0.5 else "://")  # user@host
        if is_phish and rnd.random() < 0.25:
            url = url.replace("://", "://") + f":{rnd.choice([8080, 8443, 8888])}"
        if rnd.random() < 0.2:
            url = url + f"&rnd={rnd.randint(0, 9999)}" if "?" in url else url + f"?rnd={rnd.randint(0, 9999)}"

        urls.append(url)
        labels.append(1 if is_phish else 0)
    return urls, labels


def load_urls_labels_from_scamnet_json(path: str) -> Tuple[List[str], List[int]]:
    """从 ScamNet 风格的 JSON 数据集中抽取 URL 与标签。

    期望格式（从你给的样例推断）：
    - JSON 顶层为 list
    - 每条为 dict，包含：
      - input: 形如 "...\\n## URL:\\nhttp://example.com\\n## Content:..."
      - output: "Label: scam" 或 "Label: legit"

    返回：
    - urls: List[str]
    - labels: List[int]，0=legit，1=scam/phish
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    urls: List[str] = []
    labels: List[int] = []

    # URL 抽取（兼容对抗扰动导致的格式变化）
    #
    # 1) 严格模式：匹配 "## URL:" 后面的第一行（原始数据格式）
    strict_url_re = re.compile(r"##\s*URL:\s*\n([^\n\r]+)")
    # 2) 宽松模式：允许 "url ... : <something>"（url 与冒号之间可夹杂扰动词），
    #    并抓取到 "content" 标记之前（允许在同一行）
    #    例：
    #    - "# information : # # url : http : ... # # content : ..."
    #    - "#rov url doping doping : / / example . com ... content : ..."
    loose_block_re = re.compile(
        r"\b(?:url|icao)\b.{0,40}?[:：]\s*(.+?)\s*(?=(?:\b#*\s*content\s*[:：])|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    # 3) 兜底模式：在整段文本里找第一个 http(s):// 形态（允许被插入空格）
    http_anywhere_re = re.compile(r"https?\s*:\s*//\s*\S+", flags=re.IGNORECASE)
    # 4) 再兜底：抓取类似 ": / / www . example . com / ..." 的片段（无 http(s) 前缀）
    scheme_less_slashes_re = re.compile(
        r":\s*/\s*/\s*(.+?)\s*(?=(?:\b#*\s*content\s*[:：])|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    # 5) 最终兜底：抓取“域名.后缀/路径”形态（无任何 URL 标记）
    domain_path_re = re.compile(
        r"\b([a-z0-9][a-z0-9-]*(?:\s*\.\s*[a-z0-9-]+){1,5}\s*/\s*\S+)",
        flags=re.IGNORECASE,
    )

    def _normalize_url(raw: str) -> str:
        s = raw.strip()
        # 常见扰动：把符号前后插入空格，如 "http : //", "a . b", "x / y"
        s = re.sub(r"\s+", " ", s)
        s = s.replace(" : //", "://").replace(":/ /", "://")
        s = s.replace(" : ", ":")
        s = s.replace(" / ", "/")
        s = s.replace(" ? ", "?").replace(" = ", "=").replace(" & ", "&")
        s = s.replace(" . ", ".")
        s = s.replace(" - ", "-")
        # URL 本身不应含空格：最终移除残余空格，保证 CharTokenizer 行为更接近训练分布
        s = s.replace(" ", "")
        # 对抗扰动常见形态："http:example.com/..."（缺少 //）
        low = s.lower()
        if low.startswith("http:") and not low.startswith("http://"):
            s = "http://" + s[5:]
        elif low.startswith("https:") and not low.startswith("https://"):
            s = "https://" + s[6:]
        # 形态：以 // 开头（无 scheme）
        if s.startswith("//"):
            s = "http:" + s
        return s

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        inp = item.get("input", "")
        out = item.get("output", "")

        url: str | None = None
        m = strict_url_re.search(inp)
        if m:
            url = m.group(1).strip()
        else:
            m2 = loose_block_re.search(inp)
            if m2:
                url = m2.group(1).strip()
            else:
                m3 = http_anywhere_re.search(inp)
                if m3:
                    url = m3.group(0).strip()
                else:
                    m4 = scheme_less_slashes_re.search(inp)
                    if m4:
                        url = "//" + m4.group(1).strip()
                    else:
                        m5 = domain_path_re.search(inp)
                        if m5:
                            url = "http://" + m5.group(1).strip()

        if not url:
            # 找不到 URL 的样本直接跳过
            continue
        url = _normalize_url(url)
        if not url:
            continue

        out_s = str(out).strip().lower()
        if "label:" not in out_s:
            continue
        if "scam" in out_s:
            label = 1
        elif "legit" in out_s:
            label = 0
        else:
            continue

        urls.append(url)
        labels.append(label)

    if len(urls) == 0:
        raise ValueError(f"未能从数据集解析出任何 URL：{path}")
    return urls, labels


def main() -> None:
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    # 1) 准备数据：优先从指定 JSON 数据集读取；若文件不存在则回退到模拟数据
    dataset_path = str(REPO_ROOT / "data/dataset_scamnet_5000.json")
    if os.path.exists(dataset_path):
        urls, labels = load_urls_labels_from_scamnet_json(dataset_path)
        print(f"[Info] Loaded dataset: {dataset_path} | samples={len(urls)}")
    else:
        urls, labels = make_fake_urls(n=80, seed=42)
        print(f"[Warn] Dataset not found, fallback to fake data | samples={len(urls)}")

    train_urls, train_labels, val_urls, val_labels = split_train_val(urls, labels, val_ratio=0.25, seed=42)

    # 2) 构建 tokenizer / dataset / dataloader
    vocab = CharVocab.build_default()
    tokenizer = CharTokenizer(vocab=vocab, max_len=200)

    train_ds = URLDataset(train_urls, train_labels, tokenizer)
    val_ds = URLDataset(val_urls, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # 3) 构建模型
    model = URLNetCharCNN(
        vocab_size=vocab.size,
        embed_dim=32,
        num_classes=2,
        kernel_sizes=(3, 4, 5, 6),
        out_channels=128,
        dropout_p=0.5,
        pad_id=vocab.pad_id,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 4) 训练与评估（跑几个 epoch 验证流程）
    epochs = 3
    best_f1 = -1.0
    best_ckpt_path = str(REPO_ROOT / "09_baselines/urlnet/urlnet_charcnn_best.pth")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, measure_inference_time=True, warmup_batches=1)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"prec={val_metrics['precision']:.4f} "
            f"rec={val_metrics['recall']:.4f} "
            f"f1={val_metrics['f1']:.4f} | "
            f"infer={val_metrics['inference_ms_per_sample']:.3f} ms/sample"
        )

        # 保存验证集最佳模型（按 F1）
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_f1": best_f1,
                    "epoch": epoch,
                    "vocab_size": vocab.size,
                    "embed_dim": 32,
                    "max_len": 200,
                },
                best_ckpt_path,
            )
            print(f"[Info] Saved best checkpoint to: {best_ckpt_path} (f1={best_f1:.4f})")

    # 5) 简单演示：拿几条 URL 做预测
    model.eval()
    demo_urls = [
        "https://www.google.com/account",
        "http://secure-g00gle-login.com/verify?token=abcd",
        "https://github.com/login",
    ]
    x_demo = torch.stack([tokenizer.encode(u) for u in demo_urls], dim=0).to(device)
    with torch.no_grad():
        logits = model(x_demo)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()  # 预测为钓鱼(1)的概率
    for u, p in zip(demo_urls, probs):
        print(f"[Demo] url={u} | phish_prob={p:.4f}")


if __name__ == "__main__":
    main()
