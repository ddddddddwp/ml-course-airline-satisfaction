"""raw -> processed。幂等：删掉 data/processed 后重跑结果一致。

做四件事（只做不涉及模型选择的清洗）：
1. 去重（训练集内部完全重复行）
2. 类别列 strip
3. 目标列二值化
4. Arrival Delay 缺失标记 + 用 Departure Delay 行内填补（同一行，不跨样本，无泄漏）

注意：0 分 -> NaN 的语义修正**不在这里做**，它在特征阶段（特征组 A），
这样"修正 vs 不修正"才能作为一条消融轴。
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from src.common.config import load_config, path_of, sha256_file
from src.data.schema import detect_schema, save_schema, to_binary

ARR = "Arrival Delay in Minutes"
DEP = "Departure Delay in Minutes"


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 processed 数据集")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = path_of(cfg, "raw_dir")
    out = path_of(cfg, "processed_dir")
    target = cfg["data"]["target"]

    tr_path = raw / cfg["data"]["train_file"]
    te_path = raw / cfg["data"]["test_file"]
    if not tr_path.exists():
        raise SystemExit(
            f"[data] 找不到 {tr_path}\n"
            f"       把 train.csv / test.csv / sample_submission.csv 放进 {raw}/ 后重跑。\n"
            f"       想先跑通链路：make smoke（用合成数据，不需要 Kaggle）"
        )

    train = pd.read_csv(tr_path, low_memory=False)
    test = pd.read_csv(te_path, low_memory=False)
    print(f"[data] raw train {train.shape[0]:,} × {train.shape[1]}  test {test.shape[0]:,} × {test.shape[1]}")

    # ---- 校验和（只记录一次，用于发现中途换数据）
    ck = raw / "CHECKSUMS.txt"
    lines = [f"{sha256_file(p)}  {p.name}" for p in (tr_path, te_path)]
    ck.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[data] 校验和 -> {ck}")

    schema = detect_schema(train, cfg)
    schema_path = f"{cfg['paths']['reports']}/schema_{cfg['project']}.json"
    save_schema(schema, schema_path)

    # ---- 1. 去重
    n0 = len(train)
    train = train.drop_duplicates().reset_index(drop=True)
    if len(train) != n0:
        print(f"[data] 去重：删除 {n0 - len(train):,} 行重复样本")

    # ---- 2. 类别列规范化
    for c in schema["cat_cols"]:
        if c in train:
            train[c] = train[c].astype(str).str.strip()
        if c in test:
            test[c] = test[c].astype(str).str.strip()

    # ---- 3. 目标列二值化
    train[target] = to_binary(train[target])
    rate = train[target].mean()
    print(f"[data] 目标 {target}: 正例率 {rate:.2%}")
    if not 0.02 < rate < 0.98:
        print("[data][警告] 类别高度不平衡，主指标不要用 accuracy")

    # ---- 4. 到达延误：缺失标记 + 行内填补
    if ARR in train.columns and DEP in train.columns:
        for df, name in ((train, "train"), (test, "test")):
            miss = df[ARR].isna()
            df["arr_delay_missing"] = miss.astype("int8")
            df[ARR] = df[ARR].fillna(df[DEP])
            print(f"[data] {name}: {ARR} 缺失 {int(miss.sum()):,} 行，已用 {DEP} 行内填补")

    # ---- 校验：test 不应有目标列，列集合应可对齐
    if target in test.columns:
        print(f"[data][警告] test 里出现了目标列 {target}，已丢弃")
        test = test.drop(columns=[target])
    only_train = sorted(set(train.columns) - set(test.columns) - {target})
    if only_train:
        print(f"[data][提示] 仅 train 有的列（预测时不可用，注意泄漏）: {only_train}")

    # ---- 落盘
    tr_out = out / "train.parquet"
    te_out = out / "test.parquet"
    train.to_parquet(tr_out, index=False)
    test.to_parquet(te_out, index=False)
    print(f"[data] processed -> {tr_out} ({tr_out.stat().st_size / 1024**2:.1f} MB)")
    print(f"[data] processed -> {te_out} ({te_out.stat().st_size / 1024**2:.1f} MB)")
    print(f"[data] 完成 {date.today()}")


if __name__ == "__main__":
    main()
