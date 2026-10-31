"""P2 超参敏感性实验：LightGBM 的 num_leaves 三档（31/63/127）。

严格复用 run_cv 契约（同折、同编码、同特征 base+C+T），只换 num_leaves，
每档自动登记到 reports/exp_log.csv（owner=P2, tag=sens_nl{val}）。

目的：验证"默认超参已近最优、天花板已卷平"，为报告「调参」章节提供量化证据，
而非追求分数提升（plan.md 第 8 节风险预案：模型涨不上去 → 转复现+分析型）。

路径约定：config 与输出都按仓库根解析（--config / --out），不再依赖当前工作目录。

用法：
    python -m src.models.sensitivity --config configs/base.yaml --groups C,T
    python -m src.models.sensitivity --leaves 31,63 --out reports/figures/_sens
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from src.common.config import ROOT, load_config, path_of
from src.models.train import run_cv


def _resolve_out(out: str | Path | None, cfg: dict) -> Path:
    """--out 未给时用 config 的 reports 目录；给了就按仓库根解析相对路径。"""
    if out in (None, ""):
        return path_of(cfg, "reports")
    p = Path(out)
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="LightGBM 超参敏感性（num_leaves）")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--groups", default="C,T", help="逗号分隔：base,A,B,C,T（base 隐含）")
    ap.add_argument("--leaves", default="31,63,127", help="逗号分隔的 num_leaves 取值")
    ap.add_argument("--out", default="", help="汇总表输出目录（默认 config 的 reports）")
    ap.add_argument("--owner", default="P2")
    ap.add_argument("--tag", default="sens", help="run_id 标签前缀（默认 sens）")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg0 = load_config(args.config)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    leaves = [int(x) for x in args.leaves.split(",") if x.strip()]
    out_dir = _resolve_out(args.out, cfg0)

    rows = []
    for nl in leaves:
        cfg = copy.deepcopy(cfg0)
        cfg["model"]["lgbm"]["num_leaves"] = nl
        print(f"\n===== num_leaves = {nl} =====")
        res = run_cv(
            cfg,
            model="lgbm",
            groups=groups,
            owner=args.owner,
            tag=f"{args.tag}_nl{nl}" if args.tag else f"nl{nl}",
            seed=args.seed,
            notes=f"超参敏感性 num_leaves={nl}（其余默认）",
            stage="feat",
        )
        rows.append({
            "num_leaves": nl,
            "run_id": res["run_id"],
            "cv_mean": round(res["cv_mean"], 6),
            "cv_std": round(res["cv_std"], 6),
            "n_features": res["n_features"],
        })

    table = pd.DataFrame(rows)
    out_path = out_dir / "p2_sensitivity_summary.csv"
    table.to_csv(out_path, index=False, encoding="utf-8")
    print("\n[sensitivity] 汇总")
    print(table.to_string(index=False))
    print(f"[sensitivity] -> {out_path}")
    print("[sensitivity] 选型口径：只看 CV 差异是否大于折间波动；不要为 0.0001 换超参。")


if __name__ == "__main__":
    main()
