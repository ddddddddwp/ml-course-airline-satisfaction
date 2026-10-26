"""P1 交付物验收脚本。

设计原则：**不信任任何口头结论**，能算的都算一遍。
    python scripts/verify.py              # 结构与一致性（约 15 秒）
    python scripts/verify.py --reproduce  # 额外重跑 base，对比 exp_log（约 50 秒）

退出码 0 = 全通过；1 = 有 FAIL。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config, path_of
from src.common.split import load_folds
from src.data.schema import detect_schema
from src.features.build import prepare_X

# 数据契约：换数据集时改这里
CONTRACT = {
    "train_shape": (699_635, 23),
    "test_shape": (299_844, 22),
    "target_rate": 0.4436,
    "n_splits": 5,
    "base_auc": 0.958829,
}

RESULTS: list[tuple[str, bool, str]] = []


def ck(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  |  {detail}" if detail else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--reproduce", action="store_true", help="重跑 base 并对比 exp_log")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = path_of(cfg, "raw_dir")
    proc = path_of(cfg, "processed_dir")
    target = cfg["data"]["target"]

    # ---------------- 1. 原始数据
    print("\n=== 1. 原始数据 ===")
    tr_raw = pd.read_csv(raw / cfg["data"]["train_file"], low_memory=False)
    te_raw = pd.read_csv(raw / cfg["data"]["test_file"], low_memory=False)
    ck("train 形状与契约一致", tr_raw.shape == CONTRACT["train_shape"],
       f"{tr_raw.shape[0]:,} × {tr_raw.shape[1]}（契约 {CONTRACT['train_shape']}）")
    ck("test 形状与契约一致", te_raw.shape == CONTRACT["test_shape"],
       f"{te_raw.shape[0]:,} × {te_raw.shape[1]}")
    rate = float(tr_raw[target].mean())
    ck("正例率与契约一致", abs(rate - CONTRACT["target_rate"]) < 5e-4, f"{rate:.4%}")
    ck("train 内无完全重复行", int(tr_raw.duplicated().sum()) == 0)
    ck("test 不含目标列", target not in te_raw.columns)
    ck("CHECKSUMS.txt 存在", (raw / "CHECKSUMS.txt").exists())

    # ---------------- 2. 清洗后的 processed
    print("\n=== 2. processed ===")
    tr = pd.read_parquet(proc / "train.parquet")
    te = pd.read_parquet(proc / "test.parquet")
    ck("processed 行数 = raw 去重后行数",
       len(tr) == len(tr_raw) - int(tr_raw.duplicated().sum()),
       f"{len(tr):,}")
    ck("processed 列数 = raw + 1（arr_delay_missing）",
       tr.shape[1] == tr_raw.shape[1] + 1, f"{tr.shape[1]} 列")
    ck("目标已二值化", set(tr[target].unique()) <= {0, 1})
    ck("到达延误无残留缺失", tr["Arrival Delay in Minutes"].isna().sum() == 0)

    # ---------------- 3. 冻结划分
    print("\n=== 3. splits/fold_v1.json ===")
    folds, meta = load_folds(cfg)
    fold_arr = np.asarray(meta["fold_of_row"])
    ck("行数与 meta 一致", len(fold_arr) == meta["n_rows"] == len(tr),
       f"{len(fold_arr):,} 行")
    ck("折数 = 5", meta["n_splits"] == CONTRACT["n_splits"])
    ck("每行恰好属于 1 折（全覆盖）",
       bool((np.bincount(fold_arr, minlength=meta["n_splits"]) > 0).all()
            and len(fold_arr) == sum(len(va) for _, va in folds)))

    overlap = 0
    for tr_idx, va_idx in folds:
        overlap += len(np.intersect1d(tr_idx, va_idx))
    ck("train/valid 无交集", overlap == 0, f"交集样本数 {overlap}")

    y = tr[target].to_numpy()
    rates = [float(y[va].mean()) for _, va in folds]
    ck("各折正例率一致（极差 < 1e-4）",
       max(rates) - min(rates) < 1e-4, f"极差 {max(rates) - min(rates):.2e}")
    ck("该划分来自 stratified 策略", meta.get("strategy") == "stratified",
       f"strategy={meta.get('strategy')} seed={meta.get('seed')}")

    # ---------------- 4. 特征管线
    print("\n=== 4. 特征管线 ===")
    schema = detect_schema(tr, cfg, verbose=False)
    ck("服务评分列数 = 13", len(schema["service_cols"]) == 13,
       f"{len(schema['service_cols'])}")
    ck("类别列数 = 4", len(schema["cat_cols"]) == 4, f"{schema['cat_cols']}")

    cols = [c for c in ("Flight Distance", "Age") if c in te.columns]
    X = prepare_X(tr, schema, ["base", "A", "B", "C"],
                  count_ref=pd.concat([tr[cols], te[cols]], ignore_index=True))
    ck("特征矩阵不含目标列", target not in X.columns)
    ck("特征矩阵不含 id", (schema["id_col"] or "__none__") not in X.columns)

    dict_path = ROOT / "docs/feature-dictionary.md"
    text = dict_path.read_text(encoding="utf-8") if dict_path.exists() else ""
    missing = [c for c in X.columns if f"`{c}`" not in text]
    ck("特征字典覆盖全部列（含 T 的 TE 列）",
       dict_path.exists() and not missing, f"{len(X.columns)} 列，缺 {len(missing)} 列")

    # ---------------- 5. 实验登记
    print("\n=== 5. reports/exp_log.csv ===")
    log_path = ROOT / cfg["paths"]["exp_log"]
    log = pd.read_csv(log_path)
    ck("表头与约定一致",
       list(log.columns) == ["run_id", "date", "owner", "stage", "model", "feat_ver",
                            "split_ver", "params_hash", "cv_mean", "cv_std",
                            "public_lb", "git_commit", "notes"], f"{len(log)} 行")
    base_rows = log[(log["model"] == "lgbm") & (log["feat_ver"] == "base")
                    & (log["split_ver"] == "v1")]
    ck("存在 base 基线记录", len(base_rows) >= 1,
       f"{len(base_rows)} 条，cv_mean={base_rows['cv_mean'].iloc[0]:.6f}" if len(base_rows) else "")
    if len(base_rows):
        ck("base 分数与契约一致", abs(base_rows["cv_mean"].iloc[0] - CONTRACT["base_auc"]) < 1e-6,
           f"exp_log={base_rows['cv_mean'].iloc[0]:.6f} 契约={CONTRACT['base_auc']:.6f}")
    ck("所有 run 的 cv_mean 在合理区间",
       bool(((log["cv_mean"] > 0.5) & (log["cv_mean"] < 1.0)).all()))
    ck("消融行的 split_ver 一致（同 fold 可比）",
       log["split_ver"].nunique() == 1, f"取值 {sorted(log['split_ver'].unique())}")

    # ---------------- 6. 报告物料
    print("\n=== 6. 报告物料 ===")
    figs = sorted((ROOT / "reports/figures").glob("*.png"))
    small = [f.name for f in figs if f.stat().st_size < 10_000]
    ck("5 张图存在且非空", len(figs) >= 5 and not small,
       f"{len(figs)} 张，最小 {min(f.stat().st_size for f in figs) // 1024} KB")
    sc = ROOT / "reports" / f"split_compare_{cfg['project']}.csv"
    ck("三方案对比表存在且 3 行",
       sc.exists() and len(pd.read_csv(sc)) == 3,
       f"{sc.name}" if sc.exists() else "缺失")
    fi = ROOT / "reports/feature_importance.csv"
    ck("重要性 CSV 存在", fi.exists(), f"{len(pd.read_csv(fi))} 行" if fi.exists() else "")

    # ---------------- 7. 可复现（可选）
    if args.reproduce:
        print("\n=== 7. 重跑 base 对比（约 30 秒）===")
        from src.models.train import run_cv

        res = run_cv(cfg, model="lgbm", groups=["base"], log=False, verbose=False)
        ref = CONTRACT["base_auc"]
        diff = abs(res["cv_mean"] - ref)
        ck("重跑 base 与记录一致（< 1e-6）", diff < 1e-6,
           f"重跑 {res['cv_mean']:.6f} vs 记录 {ref:.6f}，差 {diff:.2e}")

    # ---------------- 汇总
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n = len(RESULTS)
    print("\n" + "=" * 66)
    print(f"结论：{n_pass}/{n} 通过")
    if n_pass < n:
        print("未通过项：")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}  {detail}")
        sys.exit(1)
    print("全部通过。注意：本脚本只验证机器可验的部分，")
    print("        结论是否成立、图是否好看、文字是否自洽，仍需人工判断（见 docs/verify.md）。")


if __name__ == "__main__":
    main()
