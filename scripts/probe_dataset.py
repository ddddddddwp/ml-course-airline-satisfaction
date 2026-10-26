#!/usr/bin/env python3
"""数据集可行性探针（P1-01 / P1-02 / P1-03 用）

在选题阶段用同一个脚本横向比较多个候选数据集，10 秒出结论：
这个数据集能不能撑起 3 人 × 4 周 × 3 份 12~18 页报告。

用法：
    python scripts/probe_dataset.py data/raw/train.csv
    python scripts/probe_dataset.py data/raw/train.csv --test data/raw/test.csv
    python scripts/probe_dataset.py data/raw/train.csv --target satisfaction
    python scripts/probe_dataset.py data/raw/train.csv --time

输出：
    1. 规模与内存           -> 工作量是否饱满
    2. 目标列分布           -> 分类比例 / 回归偏度
    3. 缺失率排行           -> 清洗策略
    4. 完全重复行           -> 数据质量
    5. 常量列 / ID 嫌疑列   -> 泄漏排查（ID 类）
    6. 高基数类别列         -> 目标编码 / 频次编码的候选（特征组 C）
    7. train-test 行重叠    -> 泄漏排查（跨集重复，最高发）
    8. 单折耗时估算         -> 5 折是否 < 30 分钟

依赖：pandas / numpy 必装；--time 需要 lightgbm 或 scikit-learn。
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TARGET_HINTS = (
    "target", "label", "y", "satisfaction", "churn", "risk",
    "outcome", "readmitted", "default", "response",
    "price", "sale", "class",
)


def hr(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def detect_target(df: pd.DataFrame, explicit: str | None) -> str | None:
    if explicit:
        if explicit not in df.columns:
            sys.exit(f"[错误] 目标列 {explicit!r} 不在列名里：{list(df.columns)}")
        return explicit
    lowered = {c.lower(): c for c in df.columns}
    for hint in TARGET_HINTS:
        if hint in lowered:
            return lowered[hint]
    print("[提示] 没匹配到常见目标列名，已默认取最后一列。可用 --target 指定。")
    return df.columns[-1]


def sec_scale(df: pd.DataFrame) -> None:
    hr("1. 规模")
    mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"行数        : {len(df):,}")
    print(f"列数        : {df.shape[1]}")
    print(f"内存        : {mem:.1f} MB")
    n, p = df.shape
    verdict = "偏小，3 份报告可能撑不满『工作量饱满』" if n < 20_000 else "够用"
    if n > 2_000_000:
        verdict = "偏大，需采样或改 Parquet/分块读取"
    print(f"工作量判断  : {verdict}")
    print("\n类型分布：")
    for t, c in df.dtypes.astype(str).value_counts().items():
        print(f"  {t:<12} {c:>3} 列")


def sec_target(df: pd.DataFrame, target: str) -> None:
    hr(f"2. 目标列：{target}")
    s = df[target]
    n = len(s)
    print(f"缺失        : {s.isna().sum()} ({s.isna().mean():.2%})")
    print(f"唯一值      : {s.nunique()}")
    if s.nunique() <= 20:
        vc = s.value_counts(dropna=False)
        print(f"\n{'取值':<24}{'数量':>10}{'占比':>9}")
        for k, v in vc.items():
            print(f"{str(k):<24}{v:>10,}{v / n:>8.2%}")
        top = vc.iloc[0] / n
        if top > 0.85:
            print(f"\n[警告] 多数类占 {top:.1%}，必须看 balanced accuracy / PR-AUC，"
                  f"accuracy 会骗人。")
        if s.nunique() == 2:
            print("\n-> 二分类：建议 StratifiedKFold + 主指标 AUC，"
                  "阈值不要用 0.5（用 OOF 上调）。")
    else:
        print(f"最小/最大   : {s.min()} / {s.max()}")
        print(f"均值/中位数 : {s.mean():.4f} / {s.median():.4f}")
        print(f"偏度        : {s.skew():.3f}")
        print(f"零值占比    : {(s == 0).mean():.2%}")
        if abs(s.skew()) > 1:
            print("\n-> 偏态明显：报告里应给 log1p 变换前后的分布图，"
                  "并说明指标是否受极端值主导。")


def sec_missing(df: pd.DataFrame, target: str) -> None:
    hr("3. 缺失率排行")
    miss = df.drop(columns=[target]).isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        print("无缺失。")
        return
    print(f"{'列':<28}{'缺失率':>9}")
    for c, v in miss.head(15).items():
        flag = "  <- >40%，单独处理" if v > 0.4 else ""
        print(f"{c:<28}{v:>8.2%}{flag}")
    if len(miss) > 15:
        print(f"... 另有 {len(miss) - 15} 列有缺失")


def sec_duplicates(df: pd.DataFrame) -> None:
    hr("4. 完全重复行")
    d = int(df.duplicated().sum())
    print(f"重复行数    : {d:,} ({d / len(df):.2%})")
    if d:
        print("-> 训练集内部重复会虚高 CV，需在清洗阶段去重并记录。")


def sec_leak_columns(df: pd.DataFrame, target: str) -> None:
    hr("5. 常量列 / ID 嫌疑列（ID 类泄漏排查）")
    n = len(df)
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    print(f"常量列      : {const if const else '无'}")

    id_like = []
    for c in df.columns:
        if c == target:
            continue
        ratio = df[c].nunique(dropna=True) / n
        if ratio > 0.98 and df[c].dtype != "float64":
            id_like.append((c, ratio))
    if id_like:
        print("ID 嫌疑列   :")
        for c, r in sorted(id_like, key=lambda x: -x[1]):
            print(f"  - {c} (唯一率 {r:.1%})  -> 建模前删除，只保留做行对齐")
    else:
        print("ID 嫌疑列   : 无")

    # 单列与目标的关联强度速查（仅数值/低基数类别）
    if df[target].nunique() <= 20:
        print("\n与目标的粗略关联（用于发现可疑强相关列）：")
        y = df[target]
        if y.dtype == "object" or str(y.dtype) == "category":
            y = y.astype("category").cat.codes
        scores = []
        for c in df.columns:
            if c == target or df[c].nunique(dropna=False) <= 1:
                continue
            try:
                if df[c].dtype == "object" or df[c].nunique() <= 10:
                    ct = pd.crosstab(df[c], df[target], normalize="index")
                    scores.append((c, float(ct.max(axis=1).mean()), "类别"))
                else:
                    corr = pd.concat([df[c], y], axis=1).corr(numeric_only=True).iloc[0, 1]
                    if pd.notna(corr):
                        scores.append((c, abs(float(corr)), "数值"))
            except Exception:
                continue
        for c, v, kind in sorted(scores, key=lambda x: -x[1])[:8]:
            flag = "  <- 高得可疑，查是否泄漏" if v > 0.9 else ""
            print(f"  {c:<26}{kind}  {v:.3f}{flag}")


def sec_cardinality(df: pd.DataFrame, target: str) -> None:
    hr("6. 高基数类别列（特征组 C 的候选）")
    rows = []
    for c in df.columns:
        if c == target:
            continue
        if df[c].dtype == "object" or str(df[c].dtype) == "category":
            rows.append((c, df[c].nunique(dropna=True)))
    if not rows:
        print("无 object 类型列。")
        return
    print(f"{'列':<28}{'唯一值':>8}")
    for c, k in sorted(rows, key=lambda x: -x[1])[:15]:
        flag = "  <- 频次/目标编码候选" if k > 30 else ""
        print(f"{c:<28}{k:>8}{flag}")
    print("\n-> 唯一值 > 30 的类别列：目标编码必须 fold 内 fit，"
          "否则 CV 虚高 0.02~0.05 而 LB 不动。")


def sec_cross_overlap(train: pd.DataFrame, test: pd.DataFrame, target: str) -> None:
    hr("7. train / test 行重叠（跨集泄漏，最高发）")
    common = [c for c in train.columns if c in test.columns]
    if not common:
        print("两表无同名特征列，跳过。")
        return
    a = pd.util.hash_pandas_object(train[common].fillna("__NA__"), index=False)
    b = pd.util.hash_pandas_object(test[common].fillna("__NA__"), index=False)
    overlap = len(set(a) & set(b))
    print(f"比对列数    : {len(common)}")
    print(f"重叠行数    : {overlap:,} "
          f"(train {overlap / len(train):.2%} / test {overlap / len(test):.2%})")
    if overlap:
        print("-> 有重叠：查是否同一实体拆到了两集。若 test 的重复样本占比高，"
              "说明数据划分有问题，报告里必须写出来。")
    else:
        print("-> 无重叠。")


def sec_timing(df: pd.DataFrame, target: str) -> None:
    hr("8. 单折耗时估算")
    try:
        import lightgbm as lgb
    except ImportError:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier as HGB
            from sklearn.ensemble import HistGradientBoostingRegressor as HGReg
        except ImportError:
            print("未安装 lightgbm / scikit-learn，跳过。")
            return
        lgb = None
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target]).copy()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category").cat.codes
    X = X.select_dtypes(include=[np.number]).fillna(-999)
    y = df[target]

    is_clf = y.nunique() <= 20
    if is_clf and y.dtype == "object":
        y = y.astype("category").cat.codes

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if is_clf else None,
    )
    t0 = time.perf_counter()
    if lgb is not None:
        model = (lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, verbose=-1)
                 if is_clf else
                 lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, verbose=-1))
    else:
        model = HGB(max_iter=200) if is_clf else HGReg(max_iter=200)
    model.fit(Xtr, ytr)
    one = time.perf_counter() - t0
    print(f"样本/特征   : {len(Xtr):,} × {X.shape[1]}")
    print(f"单折(train+valid) : {one:.1f} s")
    print(f"5 折估算          : {one * 5 / 60:.1f} min")
    verdict = "可接受" if one * 5 < 30 * 60 else "偏慢，调参会很痛苦，考虑降采样/减特征"
    print(f"判断              : {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description="数据集可行性探针")
    ap.add_argument("train", help="训练集 csv 路径")
    ap.add_argument("--test", help="测试集 csv 路径（用于跨集重复检测）")
    ap.add_argument("--target", help="目标列名（不填则自动猜）")
    ap.add_argument("--time", action="store_true", help="跑一折估算耗时")
    args = ap.parse_args()

    print(f"读取 {args.train} ...")
    train = pd.read_csv(args.train, low_memory=False)
    target = detect_target(train, args.target)

    sec_scale(train)
    sec_target(train, target)
    sec_missing(train, target)
    sec_duplicates(train)
    sec_leak_columns(train, target)
    sec_cardinality(train, target)

    if args.test:
        print(f"\n读取 {args.test} ...")
        test = pd.read_csv(args.test, low_memory=False)
        print(f"test 规模   : {test.shape[0]:,} 行 × {test.shape[1]} 列")
        sec_cross_overlap(train, test, target)

    if args.time:
        sec_timing(train, target)

    hr("结论模板（复制进 data/README.md）")
    print(f"""- 数据集：{args.train}
- 规模：{train.shape[0]:,} 行 × {train.shape[1]} 列
- 目标列：{target}（{train[target].nunique()} 个取值）
- 单轮 5 折耗时：见第 8 节
- 是否支撑 3 组可解释特征：待 P1 判定
- 泄漏风险：见第 5、7 节
""")


if __name__ == "__main__":
    main()
