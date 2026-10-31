"""P3 评估指标：二分类概率指标 / 可靠性曲线 / bootstrap AUC 置信区间。

口径（与 plan.md 一致，报告里必须照这个写）：
- **主指标是 ROC AUC**：它只看排序，不受阈值与正例率影响，也是竞赛提交的评分方式。
- logloss / brier 反映“概率是否可信”，用于校准分析。
- accuracy / precision / recall / F1 / 混淆矩阵都依赖阈值，只作描述；
  **禁止**把提交概率阈值化成 0/1（会直接毁掉 AUC）。

边界处理：
- y_true / y_prob 长度不一致 -> ValueError；
- 单类 y_true -> AUC 记为 NaN 并警告（其余指标仍可算）；
- 全 NaN / 空输入 -> 返回 n=0 的全 NaN 指标并警告，不抛异常；
- 概率越界 -> 裁剪到 [0,1] 并警告（正常模型不应出现）。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_EPS = 1e-15
DEFAULT_BINS = 10


def _as_1d(name: str, values) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype="float64")
    except (TypeError, ValueError) as e:
        raise ValueError(f"[metrics] {name} 无法转成数值数组：{e}") from e
    return arr.reshape(-1)


def _prepare(
    y_true,
    y_prob,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """清洗输入：剔除 NaN/inf、校验标签、裁剪越界概率。"""
    y = _as_1d("y_true", y_true)
    p = _as_1d("y_prob", y_prob)
    if y.size != p.size:
        raise ValueError(f"[{context}] y_true 长度 {y.size} 与 y_prob 长度 {p.size} 不一致")

    valid = np.isfinite(y) & np.isfinite(p)
    dropped = int((~valid).sum())
    if dropped:
        warnings.warn(
            f"[{context}] {dropped} 行 y_true/y_prob 为 NaN 或 inf，已剔除；"
            "若来自 OOF，请回查生成端而不是在这里静默处理。",
            stacklevel=2,
        )
    y, p = y[valid], p[valid]
    if y.size == 0:
        return y, p, dropped

    bad_y = ~np.isin(y, (0.0, 1.0))
    if bad_y.any():
        shown = sorted(set(y[bad_y].tolist()))[:5]
        raise ValueError(f"[{context}] y_true 必须是 0/1，发现 {shown}")

    n_out = int(((p < 0.0) | (p > 1.0)).sum())
    if n_out:
        warnings.warn(
            f"[{context}] {n_out} 个预测概率越界，已裁剪到 [0,1]；请检查模型输出。",
            stacklevel=2,
        )
        p = np.clip(p, 0.0, 1.0)
    return y, p, dropped


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _bin_table(y: np.ndarray, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    """等宽分箱的可靠性表（空箱也保留，方便画图与核对）。"""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    if p.size:
        bin_idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    else:
        bin_idx = np.array([], dtype="int64")

    rows: list[dict] = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        pred_mean = float(p[mask].mean()) if n_b else float("nan")
        y_mean = float(y[mask].mean()) if n_b else float("nan")
        rows.append(
            {
                "bin": b,
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "n": n_b,
                "pred_mean": pred_mean,
                "y_mean": y_mean,
                "abs_gap": abs(pred_mean - y_mean) if n_b else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def reliability_curve(y_true, y_prob, n_bins: int = DEFAULT_BINS) -> pd.DataFrame:
    """可靠性曲线：每个概率箱的预测均值、实际正例率与样本数。

    返回列：bin, bin_lower, bin_upper, n, pred_mean, y_mean, abs_gap。
    空箱的 mean 为 NaN、n=0，保留是为了让画图和表格不跳箱。
    """
    if int(n_bins) < 1:
        raise ValueError(f"[reliability] n_bins 必须 >= 1，收到 {n_bins}")
    y, p, _ = _prepare(y_true, y_prob, context="reliability")
    return _bin_table(y, p, int(n_bins))


def binary_metrics(y_true, y_prob, threshold: float = 0.5, n_bins: int = DEFAULT_BINS) -> dict:
    """二分类概率指标汇总（AUC 为主，阈值指标只作描述）。

    返回键：n, positives, auc, logloss, brier, ece, accuracy, precision,
    recall, f1, tn, fp, fn, tp, threshold。
    n_bins 只影响 ECE 的分箱数（默认 10，与 reliability_curve 保持一致）。
    """
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"[metrics] threshold 必须在 [0,1]，收到 {threshold}")
    if int(n_bins) < 1:
        raise ValueError(f"[metrics] n_bins 必须 >= 1，收到 {n_bins}")
    threshold = float(threshold)
    y, p, _ = _prepare(y_true, y_prob, context="metrics")
    n = int(y.size)

    if n == 0:
        warnings.warn(
            "[metrics] 有效样本为 0（全 NaN 或空输入），所有指标返回 NaN。",
            stacklevel=2,
        )
        return {
            "n": 0,
            "positives": 0,
            "auc": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "threshold": threshold,
        }

    positives = int(y.sum())
    pred = (p >= threshold).astype("int8")

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    if positives == 0 or positives == n:
        warnings.warn(
            f"[metrics] y_true 只有单一类别（正例 {positives}/{n}），AUC 无定义，记为 NaN；"
            "请检查数据切片是否只含一类。",
            stacklevel=2,
        )
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y, p))

    p_clip = np.clip(p, _EPS, 1.0 - _EPS)
    logloss = float(-np.mean(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)))
    logloss = max(logloss, 0.0)
    brier = float(np.mean((p - y) ** 2))

    rel = _bin_table(y, p, int(n_bins))
    weights = rel["n"].to_numpy(dtype="float64") / n
    gaps = rel["abs_gap"].to_numpy(dtype="float64")
    ece = float(np.nansum(weights * gaps))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    return {
        "n": n,
        "positives": positives,
        "auc": auc,
        "logloss": logloss,
        "brier": brier,
        "ece": ece,
        "accuracy": _safe_div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": threshold,
    }


def bootstrap_auc_ci(
    y_true,
    y_prob,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """AUC 的 bootstrap 置信区间，返回 (low, high, bootstrap_mean)。

    70 万行时每次重采样都要重算一次 AUC，n_boot=1000 约需 2–4 分钟；
    命令行评估默认只跑 200 次（可用 --n-boot 调整），报告里写清 n_boot。
    """
    if int(n_boot) <= 0:
        warnings.warn(f"[bootstrap] n_boot={n_boot} 非法，返回 NaN。", stacklevel=2)
        return float("nan"), float("nan"), float("nan")
    if not (0.0 < float(ci) < 1.0):
        raise ValueError(f"[bootstrap] ci 必须在 (0,1)，收到 {ci}")

    y, p, _ = _prepare(y_true, y_prob, context="bootstrap")
    n = int(y.size)
    if n == 0:
        warnings.warn("[bootstrap] 有效样本为 0，返回 NaN。", stacklevel=2)
        return float("nan"), float("nan"), float("nan")
    if np.unique(y).size < 2:
        warnings.warn("[bootstrap] 单类 y_true 无法计算 AUC，返回 NaN。", stacklevel=2)
        return float("nan"), float("nan"), float("nan")

    point = float(roc_auc_score(y, p))
    rng = np.random.default_rng(int(seed))
    aucs: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if np.unique(y_b).size < 2:
            continue
        aucs.append(float(roc_auc_score(y_b, p[idx])))

    if len(aucs) < 10:
        warnings.warn(
            f"[bootstrap] 仅 {len(aucs)}/{n_boot} 次重采样同时含两类样本，无法给出稳定区间。",
            stacklevel=2,
        )
        return float("nan"), float("nan"), float("nan")

    alpha = (1.0 - float(ci)) / 2.0
    low, high = np.percentile(np.asarray(aucs), [100.0 * alpha, 100.0 * (1.0 - alpha)])
    mean = float(np.mean(aucs))
    if point < low or point > high:  # pragma: no cover - 统计上偶发，记录但不改结果
        warnings.warn(
            f"[bootstrap] 点估计 {point:.6f} 落在区间 [{low:.6f}, {high:.6f}] 之外，"
            "样本量偏小或分布偏斜，报告里需注明。",
            stacklevel=2,
        )
    return float(low), float(high), mean
