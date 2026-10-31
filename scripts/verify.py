"""交付物验收脚本。

设计原则：**不信任任何口头结论**，能算的都算一遍。
    python scripts/verify.py              # 结构与一致性（约 15 秒）
    python scripts/verify.py --reproduce  # 额外重跑 base，对比 exp_log（约 50 秒）

数据集契约（形状 / 正例率 / 折数 / 基线 AUC / 列分组）从 config 的 `contract:`
段读；没有 contract（如 configs/toy.yaml）时跳过 dataset-specific 检查，
其余"数据没被换、划分没被改、schema 一致、产物齐全"的检查照跑。

退出码 0 = 全通过；1 = 有 FAIL。
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.common.config import (
    ROOT,
    checksum_problems,
    git_commit,
    git_dirty,
    load_config,
    path_of,
    verify_checksums,
)
from src.common.split import load_folds, verify_frozen
from src.data.schema import detect_schema, to_binary
from src.features.build import prepare_X

# P3 评估产物：约定键（存在才检查；口径与 src/eval/evaluate.py 的 payload 对齐）
# 指标键 auc/roc_auc 任一；样本量键 n_rows/n/n_oof 任一——P3 实际输出的是 auc + n/n_oof。
P3_METRICS_PATH = ROOT / "reports" / "eval" / "metrics.json"
P3_METRIC_KEYS = ("auc", "roc_auc")
P3_COUNT_KEYS = ("n_rows", "n", "n_oof")

RESULTS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []


def ck(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  |  {detail}" if detail else ""))


def skip(name: str, why: str = "") -> None:
    """跳过 dataset-specific 检查：不算 PASS 也不算 FAIL，单独列出来。"""
    SKIPPED.append((name, why))
    print(f"[SKIP] {name}" + (f"  |  {why}" if why else ""))


def contract_of(cfg: dict) -> dict:
    c = cfg.get("contract")
    return c if isinstance(c, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--reproduce", action="store_true", help="重跑 base 并对比 exp_log")
    args = ap.parse_args()

    cfg = load_config(args.config)
    contract = contract_of(cfg)
    raw = path_of(cfg, "raw_dir")
    proc = path_of(cfg, "processed_dir")
    target = cfg["data"]["target"]
    id_col = cfg["data"].get("id_col")

    # ---------------- 0. 环境
    print("\n=== 0. 环境 ===")
    print(f"config      : {cfg['_path']}")
    print(f"git_commit  : {git_commit()}" + ("（工作区有未提交改动）" if git_dirty() else ""))
    if contract:
        print(f"contract    : {len(contract)} 项（{', '.join(sorted(contract))}）")
    else:
        print("contract    : 未定义 -> 跳过 dataset-specific 检查（形状/正例率/折数/列分组/基线 AUC）")

    # ---------------- 1. 原始数据
    print("\n=== 1. 原始数据 ===")
    tr_raw_path = raw / cfg["data"]["train_file"]
    te_raw_path = raw / cfg["data"]["test_file"]
    if not tr_raw_path.exists() or not te_raw_path.exists():
        missing = tr_raw_path if not tr_raw_path.exists() else te_raw_path
        raise SystemExit(
            f"[verify] 找不到 {missing}\n"
            f"         先跑 make data（或 make smoke CONFIG=configs/toy.yaml 用合成数据）"
        )
    for name in ("train.parquet", "test.parquet"):
        if not (proc / name).exists():
            raise SystemExit(f"[verify] 找不到 {proc / name}，先跑 make data")

    tr_raw = pd.read_csv(tr_raw_path, low_memory=False)
    te_raw = pd.read_csv(te_raw_path, low_memory=False)

    if "train_shape" in contract:
        want = [int(v) for v in contract["train_shape"]]
        ck("train 形状与契约一致", list(tr_raw.shape) == want,
           f"{tr_raw.shape[0]:,} × {tr_raw.shape[1]}（契约 {tuple(want)}）")
    else:
        skip("train 形状与契约一致", "contract.train_shape 未定义")

    if "test_shape" in contract:
        want = [int(v) for v in contract["test_shape"]]
        ck("test 形状与契约一致", list(te_raw.shape) == want,
           f"{te_raw.shape[0]:,} × {te_raw.shape[1]}（契约 {tuple(want)}）")
    else:
        skip("test 形状与契约一致", "contract.test_shape 未定义")

    rate = float(to_binary(tr_raw[target]).mean())
    if "target_rate" in contract:
        ref = float(contract["target_rate"])
        ck("正例率与契约一致", abs(rate - ref) < 5e-4, f"{rate:.4%}（契约 {ref:.4%}）")
    else:
        skip("正例率与契约一致", "contract.target_rate 未定义")

    # 去重口径：唯一 id 不能算进比较键，否则"无重复行"永远成立（旧实现的坑）
    dedup_cols = [c for c in tr_raw.columns if c != id_col]
    n_dup = int(tr_raw.duplicated(subset=dedup_cols).sum())
    ck("train 内无重复行（按除 id 外的全部列）", n_dup == 0, f"重复 {n_dup:,} 行")
    if id_col and id_col in tr_raw.columns:
        n_dup_id = int(tr_raw[id_col].duplicated().sum())
        ck(f"id 列唯一（{id_col}）", n_dup_id == 0, f"重复 id {n_dup_id:,} 个")
    else:
        skip(f"id 列唯一（{id_col}）", "配置没有 data.id_col")

    ck("test 不含目标列", target not in te_raw.columns)

    # 校验和：真的重算哈希（CSV 归一化 CRLF->LF），不是只看文件在不在
    ck_rep = verify_checksums(raw, [tr_raw_path, te_raw_path])
    problems = checksum_problems(ck_rep)
    ck("CHECKSUMS.txt 校验通过（CSV 按 CRLF->LF 归一化）",
       ck_rep["ok"] and not problems,
       f"已校验 {len(ck_rep['verified'])} 个文件" + (f"；问题：{'；'.join(problems)}" if problems else ""))

    # ---------------- 2. 清洗后的 processed
    print("\n=== 2. processed ===")
    tr = pd.read_parquet(proc / "train.parquet")
    te = pd.read_parquet(proc / "test.parquet")
    ck("processed 行数 = raw 去重后行数", len(tr) == len(tr_raw) - n_dup, f"{len(tr):,}")
    ck("processed 列数 = raw + 1（arr_delay_missing）",
       tr.shape[1] == tr_raw.shape[1] + 1, f"{tr.shape[1]} 列")
    ck("目标已二值化", set(tr[target].unique()) <= {0, 1})
    if "Arrival Delay in Minutes" in tr.columns:
        n_miss = int(tr["Arrival Delay in Minutes"].isna().sum())
        ck("到达延误无残留缺失", n_miss == 0, f"缺失 {n_miss:,}")
    else:
        skip("到达延误无残留缺失", "processed 没有 Arrival Delay in Minutes 列")

    # ---------------- 3. 冻结划分
    print("\n=== 3. splits/fold_v1.json ===")
    folds, meta = load_folds(cfg)
    fold_arr = np.asarray(meta["fold_of_row"])
    ck("行数与 meta 一致", len(fold_arr) == int(meta["n_rows"]) == len(tr),
       f"{len(fold_arr):,} 行")
    if "n_splits" in contract:
        ck("折数 = 契约", int(meta["n_splits"]) == int(contract["n_splits"]),
           f"{meta['n_splits']} 折（契约 {contract['n_splits']}）")
    else:
        skip("折数 = 契约", "contract.n_splits 未定义")
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

    frozen = verify_frozen(cfg)
    if frozen["checked"]:
        ck("冻结划分与 processed 数据指纹一致", frozen["ok"], frozen["detail"])
    else:
        skip("冻结划分与 processed 数据指纹一致", frozen["detail"])

    # ---------------- 4. 特征管线
    print("\n=== 4. 特征管线 ===")
    schema = detect_schema(tr, cfg, verbose=False)

    want_service = [str(c) for c in (contract.get("service_cols") or [])]
    if want_service:
        got = list(schema["service_cols"])
        extra = sorted(set(got) - set(want_service))
        absent = sorted(set(want_service) - set(got))
        ck("服务评分列与契约一致", not extra and not absent,
           f"{len(got)}/{len(want_service)} 列"
           + (f"；多 {extra}；缺 {absent}" if extra or absent else ""))
    else:
        skip("服务评分列与契约一致", "contract.service_cols 未定义")

    want_cat = [str(c) for c in (contract.get("cat_cols") or [])]
    if want_cat:
        got = list(schema["cat_cols"])
        extra = sorted(set(got) - set(want_cat))
        absent = sorted(set(want_cat) - set(got))
        ck("类别列与契约一致", not extra and not absent,
           f"{len(got)}/{len(want_cat)} 列"
           + (f"；多 {extra}；缺 {absent}" if extra or absent else ""))
    else:
        skip("类别列与契约一致", "contract.cat_cols 未定义")

    # 落盘的 schema 必须和重新探测的结果一致：防止报告引用过期 schema
    schema_path = path_of(cfg, "reports") / f"schema_{cfg['project']}.json"
    if not schema_path.exists():
        ck("保存的 schema 与 detect_schema 一致", False, f"缺少 {schema_path.name}")
    else:
        saved = json.loads(schema_path.read_text(encoding="utf-8"))
        fresh = detect_schema(tr_raw, cfg, verbose=False)
        diff_keys = [k for k in fresh if saved.get(k) != fresh[k]]
        ck("保存的 schema 与 detect_schema 一致", not diff_keys,
           f"{schema_path.name}"
           + (f"；不一致字段 {diff_keys}" if diff_keys else "；4 组列全一致"))

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
    base_rows = pd.DataFrame()
    if not log_path.exists():
        if contract:
            ck("exp_log 存在", False, f"缺少 {log_path.relative_to(ROOT)}")
        else:
            skip("reports/exp_log.csv", f"{log_path.name} 尚未生成（先跑 train）")
    else:
        log = pd.read_csv(log_path)
        ck("表头与约定一致",
           list(log.columns) == ["run_id", "date", "owner", "stage", "model", "feat_ver",
                                "split_ver", "params_hash", "cv_mean", "cv_std",
                                "public_lb", "git_commit", "notes"], f"{len(log)} 行")
        base_rows = log[(log["model"] == "lgbm") & (log["feat_ver"] == "base")
                        & (log["split_ver"] == "v1")]
        base_detail = (
            f"{len(base_rows)} 条，cv_mean={base_rows['cv_mean'].iloc[0]:.6f}"
            if len(base_rows) else ""
        )
        ck("存在 base 基线记录", len(base_rows) >= 1, base_detail)
        if len(base_rows):
            if "base_auc" in contract:
                ref = float(contract["base_auc"])
                ck("base 分数与契约一致", abs(base_rows["cv_mean"].iloc[0] - ref) < 1e-6,
                   f"exp_log={base_rows['cv_mean'].iloc[0]:.6f} 契约={ref:.6f}")
            else:
                skip("base 分数与契约一致", "contract.base_auc 未定义")
        ck("所有 run 的 cv_mean 在合理区间",
           bool(((log["cv_mean"] > 0.5) & (log["cv_mean"] < 1.0)).all()))
        ck("消融行的 split_ver 一致（同 fold 可比）",
           log["split_ver"].nunique() == 1, f"取值 {sorted(log['split_ver'].unique())}")

    # ---------------- 6. 报告物料
    print("\n=== 6. 报告物料 ===")
    figs = sorted((ROOT / "reports/figures").glob("*.png"))
    small = [f.name for f in figs if f.stat().st_size < 10_000]
    ck("5 张图存在且非空", len(figs) >= 5 and not small,
       f"{len(figs)} 张，最小 {min(f.stat().st_size for f in figs) // 1024} KB" if figs else "无 PNG")
    sc = ROOT / "reports" / f"split_compare_{cfg['project']}.csv"
    if sc.exists():
        ck("三方案对比表存在且 3 行", len(pd.read_csv(sc)) == 3, sc.name)
    elif contract:
        ck("三方案对比表存在且 3 行", False, f"缺少 {sc.relative_to(ROOT)}")
    else:
        skip("三方案对比表存在且 3 行", f"{sc.name} 尚未生成（先跑 split-compare）")
    fi = ROOT / "reports/feature_importance.csv"
    if fi.exists():
        ck("重要性 CSV 存在", True, f"{len(pd.read_csv(fi))} 行")
    elif contract:
        ck("重要性 CSV 存在", False, "缺少 reports/feature_importance.csv")
    else:
        skip("重要性 CSV 存在", "reports/feature_importance.csv 尚未生成（先跑 importance）")

    # P3 评估产物：目前由 P3 生成，存在则校验键；不存在只记录 SKIP
    if P3_METRICS_PATH.exists():
        try:
            metrics = json.loads(P3_METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            ck("P3 metrics.json 含约定键", False, f"JSON 解析失败：{e}")
        else:
            keys = set(metrics) if isinstance(metrics, dict) else set()
            has_metric = any(k in keys for k in P3_METRIC_KEYS)
            has_count = any(k in keys for k in P3_COUNT_KEYS)
            detail = f"keys={sorted(keys)}"
            if not has_metric:
                detail += f"；缺指标键 {list(P3_METRIC_KEYS)} 之一"
            if not has_count:
                detail += f"；缺样本量键 {list(P3_COUNT_KEYS)} 之一"
            ck("P3 metrics.json 含约定键",
               isinstance(metrics, dict) and has_metric and has_count, detail)
    else:
        skip("P3 metrics.json 含约定键",
             f"{P3_METRICS_PATH.relative_to(ROOT)} 尚未生成（P3 交付物）")

    # ---------------- 7. 可复现（可选）
    if args.reproduce:
        print("\n=== 7. 重跑 base 对比（约 30 秒）===")
        from src.models.train import run_cv

        res = run_cv(cfg, model="lgbm", groups=["base"], log=False, verbose=False)
        if "base_auc" in contract:
            ref = float(contract["base_auc"])
            origin = "契约"
        elif len(base_rows):
            ref = float(base_rows["cv_mean"].iloc[0])
            origin = "exp_log"
        else:
            ref = None
            origin = ""
        if ref is None:
            skip("重跑 base 与记录一致", "没有 contract.base_auc，也没有 base 记录可比")
        else:
            diff = abs(res["cv_mean"] - ref)
            ck("重跑 base 与记录一致（< 1e-6）", diff < 1e-6,
               f"重跑 {res['cv_mean']:.6f} vs {origin} {ref:.6f}，差 {diff:.2e}")

    # ---------------- 汇总
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n = len(RESULTS)
    print("\n" + "=" * 66)
    print(f"结论：{n_pass}/{n} 通过" + (f"，{len(SKIPPED)} 项跳过" if SKIPPED else ""))
    if n_pass < n:
        print("未通过项：")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}  {detail}")
    if SKIPPED:
        print("跳过项（原因）：")
        for name, why in SKIPPED:
            print(f"  - {name}  {why}")
    if n_pass < n:
        sys.exit(1)
    print("全部通过。注意：本脚本只验证机器可验的部分，")
    print("        结论是否成立、图是否好看、文字是否自洽，仍需人工判断（见 docs/verify.md）。")


if __name__ == "__main__":
    main()
