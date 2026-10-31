"""raw -> processed。幂等：删掉 data/processed 后重跑结果一致。

做四件事（只做不涉及模型选择的清洗）：
1. 去重（训练集内部重复样本，按除 id 外的全部列判断）
2. 类别列 strip
3. 目标列二值化
4. Arrival Delay 缺失标记 + 用 Departure Delay 行内填补（同一行，不跨样本，无泄漏）

另外负责 raw CSV 的校验和闸门：数据被替换时直接退出，不静默重算。

注意：0 分 -> NaN 的语义修正**不在这里做**，它在特征阶段（特征组 A），
这样"修正 vs 不修正"才能作为一条消融轴。
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from src.common.config import (
    checksum_problems,
    load_config,
    path_of,
    verify_checksums,
    write_checksums,
)
from src.data.schema import detect_schema, save_schema, to_binary

ARR = "Arrival Delay in Minutes"
DEP = "Departure Delay in Minutes"


def dedupe(df: pd.DataFrame, id_col: str | None = None) -> tuple[pd.DataFrame, int, int]:
    """按除 id 外的全部列去重。

    旧实现 `df.drop_duplicates()` 把唯一 id 也算进比较键，重复样本永远删不掉
    （id 每行都不同）。这里把 id 从比较键里去掉，单独查 id 是否唯一。

    返回 (去重后的 df, 删除行数, 重复 id 个数)。
    """
    has_id = bool(id_col) and id_col in df.columns
    dup_ids = int(df[id_col].duplicated().sum()) if has_id else 0
    subset = [c for c in df.columns if c != id_col] if has_id else list(df.columns)

    n0 = len(df)
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")
    removed = n0 - len(df)

    # 兜底：整行完全重复（正常不会再有；防止 subset 构造意外为空）
    n1 = len(df)
    df = df.drop_duplicates(keep="first")
    removed += n1 - len(df)

    return df.reset_index(drop=True), removed, dup_ids


def _checksum_gate(
    raw, tr_path, te_path, accept: bool
) -> None:
    """校验和闸门：先确认原始数据没被换过，再清洗。

    - CHECKSUMS.txt 不存在：新建（首次记录）
    - 存在且一致：继续，不改文件
    - 存在但摘要对不上：SystemExit，除非显式 --accept-data-change
    """
    ck = raw / "CHECKSUMS.txt"
    report = verify_checksums(raw, [tr_path, te_path])

    if not report["exists"]:
        entries = write_checksums(ck, [tr_path, te_path])
        print(f"[data] 首次记录校验和 -> {ck}")
        for name, digest in entries.items():
            print(f"[data]   {digest}  {name}")
        return

    problems = checksum_problems(report)
    if problems and not accept:
        raise SystemExit(
            "[data] 原始数据与 CHECKSUMS.txt 不一致：\n"
            + "\n".join(f"         - {p}" for p in problems)
            + "\n       数据可能被替换（重新下载 / 换了版本 / 手工改过）。"
            "\n       确认是有意更换后，用 --accept-data-change 重新登记校验和；"
            "\n       否则请恢复原始文件（git checkout -- data/raw）。"
        )

    if problems:
        entries = write_checksums(ck, [tr_path, te_path])
        print("[data] 已按 --accept-data-change 更新校验和：")
        for name, digest in entries.items():
            print(f"[data]   {digest}  {name}")
        return

    print(f"[data] 校验和一致（{', '.join(report['verified'])}；CSV 按 CRLF->LF 归一化后哈希）")
    if report["extra"]:
        print(f"[data][提示] CHECKSUMS.txt 里还有未纳入本次校验的条目：{report['extra']}")
    if accept:
        print("[data] 数据未变化，CHECKSUMS.txt 无需更新")


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 processed 数据集")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument(
        "--accept-data-change",
        action="store_true",
        help="确认 raw CSV 已被有意更换，用当前文件更新 data/raw/CHECKSUMS.txt",
    )
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
    if not te_path.exists():
        raise SystemExit(f"[data] 找不到 {te_path}")

    # ---- 0. 校验和闸门（在读 92 MB CSV 之前先把"数据有没有被换"问清楚）
    _checksum_gate(raw, tr_path, te_path, args.accept_data_change)

    train = pd.read_csv(tr_path, low_memory=False)
    test = pd.read_csv(te_path, low_memory=False)
    print(
        f"[data] raw train {train.shape[0]:,} × {train.shape[1]}  "
        f"test {test.shape[0]:,} × {test.shape[1]}"
    )

    schema = detect_schema(train, cfg)
    schema_path = path_of(cfg, "reports") / f"schema_{cfg['project']}.json"
    save_schema(schema, schema_path)
    print(f"[data] schema -> {schema_path}")

    # ---- 1. 去重（按除 id 外的全部列；id 唯一性单独查）
    id_col = cfg["data"].get("id_col")
    train, removed, dup_ids = dedupe(train, id_col)
    print(f"[data] 去重：按除 {id_col or '(无 id 列)'} 外的全部列检测，删除 {removed:,} 行重复样本")
    if dup_ids:
        print(f"[data][警告] id 列 {id_col} 有 {dup_ids:,} 个重复值，请检查数据来源")

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
