"""P2 超参敏感性实验：LightGBM 的 num_leaves 三档（31/63/127）。

严格复用 run_cv 契约（同折、同编码、同特征 base+C+T），只换 num_leaves，
每档自动登记到 reports/exp_log.csv（owner=P2, tag=sens_nl{val}）。

目的：验证"默认超参已近最优、天花板已卷平"，为报告「调参」章节提供量化证据，
而非追求分数提升（plan.md 第 8 节风险预案：模型涨不上去 → 转复现+分析型）。
"""
from __future__ import annotations

import copy

from src.common.config import load_config
from src.models.train import run_cv


def main() -> None:
    base = load_config("configs/base.yaml")
    for nl in (31, 63, 127):
        cfg = copy.deepcopy(base)
        cfg["model"]["lgbm"]["num_leaves"] = nl
        print(f"\n===== num_leaves = {nl} =====")
        run_cv(
            cfg,
            model="lgbm",
            groups=("C", "T"),
            owner="P2",
            tag=f"sens_nl{nl}",
            notes=f"超参敏感性 num_leaves={nl}（其余默认）",
        )


if __name__ == "__main__":
    main()
