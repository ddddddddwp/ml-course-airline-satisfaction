"""生成与 S6E10 同结构的合成数据，用于在没有 Kaggle 数据时跑通全链路。

刻意复刻了真实数据的三个坑：
  1. 14 个 0-5 序数评分，0 = "不适用"（少量）
  2. 两个延误列 90% 为 0，且高度相关；Arrival Delay 有极少量缺失
  3. Flight Distance 高基数（可做频次/目标编码）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate(logit: np.ndarray, target_rate: float) -> np.ndarray:
    """平移截距，使正例率贴近目标值（让合成数据更像真实数据）。"""
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = _sigmoid(logit + mid).mean()
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return logit + (lo + hi) / 2


def make(n: int, rng: np.random.Generator, start_id: int, with_target: bool) -> pd.DataFrame:
    gender = rng.choice(["Female", "Male"], n)
    loyal = rng.random(n) < 0.825
    business_travel = rng.random(n) < 0.711
    cls = rng.choice(["Business", "Eco", "Eco Plus"], n, p=[0.489, 0.468, 0.043])

    age = np.clip(rng.normal(40, 15, n), 7, 85).round().astype(int)
    dist = np.exp(rng.normal(6.7, 0.75, n)).clip(50, 5000).round().astype(int)

    is_biz = (cls == "Business").astype(float)
    svc = {}
    for c in SERVICE:
        mu = 3.1 + 0.8 * is_biz + 0.3 * business_travel
        v = np.clip(rng.normal(mu, 1.0, n).round(), 1, 5).astype(int)
        # 约 1.5% 为 0 = "不适用"
        v[rng.random(n) < 0.015] = 0
        svc[c] = v

    # 在线值机是最强特征
    ob = svc["Online boarding"]
    svc["Online boarding"] = np.where(
        rng.random(n) < 0.35, np.clip(ob + (is_biz * 1.2).round().astype(int), 1, 5), ob
    )

    dep = np.where(rng.random(n) < 0.90, 0, rng.gamma(2.0, 15.0, n).round())
    arr = np.where(rng.random(n) < 0.87, dep, np.clip(dep + rng.normal(0, 8, n), 0, None).round())

    raw = np.column_stack([svc[c] for c in SERVICE]).astype(float)
    n_zeros = (raw == 0).sum(axis=1)
    with np.errstate(invalid="ignore"):
        svc_mean = np.where(n_zeros == len(SERVICE), np.nan,
                            np.nanmean(np.where(raw == 0, np.nan, raw), axis=1))
    svc_mean = np.where(np.isnan(svc_mean), 3.0, svc_mean)

    logit = (
        -1.1
        + 2.2 * is_biz
        - 1.7 * (~business_travel)
        + 0.85 * (svc["Online boarding"] == 5)
        + 0.45 * (svc["Online boarding"] >= 4)
        + 0.30 * (svc["Inflight wifi service"] >= 4)
        + 1.10 * (svc_mean - 3.2)
        + 0.70 * loyal
        + 0.45 * (n_zeros >= 3)
        - 0.35 * (dep > 15)
        + 0.35 * (np.log1p(dist) - 6.6)
    )
    logit = _calibrate(logit, 0.444)
    y = rng.binomial(1, _sigmoid(logit))

    df = pd.DataFrame({
        "id": np.arange(start_id, start_id + n),
        "Gender": gender,
        "Customer Type": np.where(loyal, "Loyal Customer", "disloyal Customer"),
        "Age": age,
        "Type of Travel": np.where(business_travel, "Business travel", "Personal Travel"),
        "Class": cls,
        "Flight Distance": dist,
    })
    for c in SERVICE:
        df[c] = svc[c]
    df["Departure Delay in Minutes"] = dep.astype(int)
    df["Arrival Delay in Minutes"] = arr
    # 极少量缺失，模拟真实数据
    miss = rng.random(n) < 0.002
    df.loc[miss, "Arrival Delay in Minutes"] = np.nan

    if with_target:
        df["satisfaction"] = y
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy/raw")
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--test", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train = make(args.train, rng, 0, True)
    test = make(args.test, rng, args.train, False)

    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "satisfaction": 0.4436}).to_csv(
        out / "sample_submission.csv", index=False
    )

    print(f"[toy] {out}/train.csv  {train.shape[0]:,} × {train.shape[1]}")
    print(f"[toy] {out}/test.csv   {test.shape[0]:,} × {test.shape[1]}")
    print(f"[toy] 正例率 {train['satisfaction'].mean():.3f}  "
          f"Flight Distance 唯一值 {train['Flight Distance'].nunique():,}")


if __name__ == "__main__":
    main()
