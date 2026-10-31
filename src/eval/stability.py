"""P3 稳定性实验：多 seed × 5 折，量化“换随机种子”带来的波动。

为什么必须做（plan.md P3-09）：
折间 std ≈ 0.0006，而全部特征工程增量只有约 0.0014。单 seed 的“提升”很可能只是
seed 噪声，因此任何“变好/变差”的结论都必须先看这里的 seed 间波动。

口径：
- 复用 `src.models.train.run_cv`，冻结划分 `splits/fold_v1.json` 只读；
- 每个 seed 跑完整 5 折，记录 cv_mean / cv_std / fold_aucs / run_id；
- 相对主模型 Δ = 本 seed 的 OOF AUC − 主模型 OOF AUC（默认从 exp_log 找主模型）。

用法：
    $env:PYTHONPATH="."
    python -m src.eval.stability --config configs/base.yaml --seeds 42,2026,7
    python -m src.eval.stability --config configs/base.yaml --seeds 42      # 单 seed 冒烟
    python -m src.eval.stability --config configs/base.yaml --seeds 42 --dry-run
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.eval._io import write_json
from src.models.train import run_cv

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

# 主模型（P2_feat_lgbm_CT_1029_main）的 OOF AUC；仅当 exp_log 里找不到时兜底。
MAIN_OOF_AUC_FALLBACK = 0.960191


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def parse_seeds(text: str) -> list[int]:
    """解析 --seeds：逗号分隔、去重、保持输入顺序。"""
    parts = [x.strip() for x in str(text).split(",") if x.strip()]
    if not parts:
        raise ValueError("[stability] --seeds 为空；示例：--seeds 42,2026,7")
    seeds: list[int] = []
    for x in parts:
        try:
            s = int(x)
        except ValueError as e:
            raise ValueError(f"[stability] --seeds 里的 {x!r} 不是整数") from e
        if s < 0:
            raise ValueError(f"[stability] seed 不能为负数：{s}")
        if s not in seeds:
            seeds.append(s)
    if len(seeds) > 5:
        print(f"[stability][提示] {len(seeds)} 个 seed × 5 折会跑很久，建议先 --dry-run 看计划。")
    return seeds


def _baseline_from_exp_log(cfg: dict) -> tuple[float, str] | None:
    """从 exp_log 找主模型 OOF AUC：优先 run_id 带 main 的 C+T LightGBM 行。"""
    p = _resolve(cfg.get("paths", {}).get("exp_log", "reports/exp_log.csv"))
    if not p.exists():
        return None
    try:
        log = pd.read_csv(p)
    except Exception:
        return None
    need = {"run_id", "model", "feat_ver", "cv_mean"}
    if not need <= set(log.columns):
        return None

    sub = log[
        (log["model"].astype(str) == "lgbm")
        & (log["feat_ver"].astype(str).str.replace(" ", "", regex=False) == "C+T")
        & log["cv_mean"].notna()
    ]
    if sub.empty:
        return None

    main = sub[sub["run_id"].astype(str).str.contains("main", case=False, na=False)]
    if not main.empty:
        row = main.iloc[-1]
        return float(row["cv_mean"]), f"exp_log:{row['run_id']}"
    row = sub.sort_values("cv_mean", kind="mergesort").iloc[-1]
    return float(row["cv_mean"]), f"exp_log:{row['run_id']}（C+T 最高分行）"


def resolve_baseline(cfg: dict, override: float | None = None) -> tuple[float, str]:
    """返回 (主模型 OOF AUC, 来源说明)。"""
    if override is not None:
        return float(override), "CLI --baseline"
    found = _baseline_from_exp_log(cfg)
    if found is not None:
        return found
    print(
        f"[stability][警告] exp_log 中找不到 C+T 主模型，Δ 将使用内置参考值 "
        f"{MAIN_OOF_AUC_FALLBACK:.6f}（如需覆盖请加 --baseline）。"
    )
    return MAIN_OOF_AUC_FALLBACK, "内置参考值（exp_log 缺失）"


def summarize_runs(runs: list[dict], baseline: float) -> dict:
    """跨 seed 汇总：均值、seed 间 std（ddof=1）、极差、平均折内 std。"""
    if not runs:
        nan = float("nan")
        return {
            "n_seeds": 0,
            "seed_mean_auc": nan,
            "seed_std_auc": nan,
            "seed_min_auc": nan,
            "seed_max_auc": nan,
            "seed_range": nan,
            "mean_fold_std": nan,
            "mean_delta_vs_baseline": nan,
        }
    aucs = np.asarray([float(r["cv_mean"]) for r in runs], dtype="float64")
    fold_stds = np.asarray([float(r["cv_std"]) for r in runs], dtype="float64")
    mean = float(aucs.mean())
    std = float(aucs.std(ddof=1)) if aucs.size > 1 else float("nan")
    return {
        "n_seeds": int(aucs.size),
        "seed_mean_auc": mean,
        "seed_std_auc": std,
        "seed_min_auc": float(aucs.min()),
        "seed_max_auc": float(aucs.max()),
        "seed_range": float(aucs.max() - aucs.min()),
        "mean_fold_std": float(fold_stds.mean()),
        "mean_delta_vs_baseline": mean - float(baseline),
    }


def _plot_seed_variance(runs: list[dict], summary: dict, baseline: float, path: Path) -> None:
    """左：每 seed 的 OOF AUC 点 + 均值线（y 轴放大）；右：每 seed 的 5 个 fold AUC。

    为什么不把 fold std 画成左图误差棒：fold std≈6e-4，而 seed 间极差只有 5e-5，
    叠在一起会把 seed 差异压成一条线，读者看不出"换 seed 几乎不动"。
    分开画才能同时回答：seed 间差多少、折间差多少。
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    xs = np.arange(len(runs))
    aucs = np.asarray([float(r["cv_mean"]) for r in runs], dtype="float64")
    mean = summary.get("seed_mean_auc", float("nan"))

    ax.plot(xs, aucs, "o-", ms=6, lw=1.2, color="#2b6cb0", label="OOF AUC (5-fold)")
    for x, a in zip(xs, aucs):
        ax.annotate(f"{a:.6f}", (x, a), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7)
    if np.isfinite(mean):
        ax.axhline(mean, color="#c53030", ls="--", lw=1.1,
                   label=f"seed mean = {mean:.6f}")
    if np.isfinite(baseline):
        ax.axhline(baseline, color="#718096", ls=":", lw=1.1,
                   label=f"main model = {baseline:.6f}")

    span = float(np.ptp(aucs)) if len(aucs) else 0.0
    pad = max(span * 0.8, 2.0e-5)
    lows = [float(aucs.min())] if len(aucs) else [mean]
    highs = [float(aucs.max())] if len(aucs) else [mean]
    if np.isfinite(baseline):
        lows.append(baseline)
        highs.append(baseline)
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xticks(xs, [str(r["seed"]) for r in runs])
    ax.set_xlabel("random seed")
    ax.set_ylabel("OOF ROC AUC (y-axis zoomed)")
    ax.set_title(
        f"Seed-level: OOF AUC (n={len(runs)}, "
        f"seed std={summary.get('seed_std_auc', float('nan')):.6f})"
    )
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.margins(x=0.12)

    # 右图：同一批 fold、不同 seed 的 fold AUC，显示折间波动远大于 seed 间波动
    rng = np.random.default_rng(0)
    for i, r in enumerate(runs):
        folds = np.asarray(r.get("fold_aucs", []), dtype="float64")
        if folds.size == 0:
            continue
        ax2.scatter(i + rng.uniform(-0.12, 0.12, folds.size), folds,
                    s=22, color="#2b6cb0", alpha=0.75, linewidths=0)
        ax2.hlines(float(r["cv_mean"]), i - 0.28, i + 0.28, color="#c53030", lw=1.2)
        ax2.annotate(f"std={float(r['cv_std']):.6f}", (i, folds.min()),
                     textcoords="offset points", xytext=(0, -12),
                     ha="center", fontsize=6, color="#4a5568")
    ax2.set_xticks(xs, [str(r["seed"]) for r in runs])
    ax2.set_xlim(-0.5, max(len(runs) - 0.5, 0.5))
    ax2.set_xlabel("random seed")
    ax2.set_ylabel("Fold ROC AUC")
    ax2.set_title("Fold-level: 5 fold AUCs per seed (red = OOF mean)")
    ax2.margins(y=0.18)

    fig.suptitle(
        "P3 stability: seed-level OOF AUC (left, zoomed) vs fold-level AUC (right)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(
    out: Path,
    runs: list[dict],
    summary: dict,
    baseline: float,
    baseline_source: str,
    seeds: list[int],
    groups: list[str],
) -> None:
    lines = [
        "# P3 稳定性：多 seed × 5 折",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        f"> 特征组：`{'base' if set(groups) <= {'base'} else '+'.join(g for g in groups)}`  ｜  划分：`splits/fold_v1.json`（只读）  ",
        f"> 主模型参考 OOF AUC：**{baseline:.6f}**（来源：{baseline_source}）",
        "",
        "| seed | OOF AUC | fold std | Δ vs 主模型 | run_id | 耗时(s) |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for r in runs:
        delta = float(r["cv_mean"]) - baseline
        lines.append(
            f"| {r['seed']} | {r['cv_mean']:.6f} | {r['cv_std']:.6f} | "
            f"{delta:+.6f} | `{r['run_id']}` | {r.get('seconds', float('nan')):.1f} |"
        )
    if runs:
        lines.append(
            f"| **seed 均值** | **{summary['seed_mean_auc']:.6f}** | "
            f"**{summary['mean_fold_std']:.6f}**（平均折内） | "
            f"**{summary['mean_delta_vs_baseline']:+.6f}** | - | - |"
        )

    read_lines = [
        "",
        "## 读法",
        "",
        f"- 共 {summary['n_seeds']} 个 seed；seed 间 std = "
        f"**{summary['seed_std_auc']:.6f}**，极差 = **{summary['seed_range']:.6f}**。",
        "- `seed std` 是“换 random_state”的模型随机性；`fold std` 是同一 seed 内 5 折之间的数据波动。",
        "  两者含义不同，**不能**把 `fold std` 当成 OOF AUC 的误差棒——OOF 的抽样不确定性看 `metrics.json` 的 bootstrap CI。",
    ]
    if (
        np.isfinite(summary.get("seed_std_auc", float("nan")))
        and summary["seed_std_auc"] > 0
        and np.isfinite(summary.get("mean_fold_std", float("nan")))
    ):
        ratio = summary["mean_fold_std"] / summary["seed_std_auc"]
        read_lines.append(
            f"- 本次平均折内 std 约为 seed 间 std 的 **{ratio:.0f} 倍**："
            "换 seed 几乎不动，折与折之间的差异大得多；这正说明“冻结同一把尺子”比多跑 seed 更关键。"
        )
    read_lines += [
        "- 若 |Δ vs 主模型| 小于 seed 间 std，就不能宣称“提升/下降”，只能写“在噪声范围内”；",
        "  更实用的判据是 |Δ| 是否小于 AUC 的 bootstrap 95% CI 半宽（通常约 0.0004）。",
        "- 多 seed 只是估计方差，不是正式显著性检验；正式结论还需配对检验或更多 seed。",
    ]
    lines += read_lines
    if len(seeds) < 3:
        lines += ["", f"> 本次只跑了 {len(seeds)} 个 seed（冒烟）。**不足以**支撑“显著/不显著”的结论。"]
    lines += [
        "",
        "## 产物",
        "",
        "- `stability.json`：每个 seed 的 cv_mean / cv_std / fold_aucs / run_id；",
        "- `p3_seed_variance.png`：左图每 seed 一个 OOF 点 + seed 均值线（y 轴放大），右图每个 seed 的 5 折 AUC；",
        "- OOF 文件与 `reports/exp_log.csv` 中 `P3_*` 行（除非用了 `--no-log`）。",
        "",
    ]
    (out / "stability.md").write_text("\n".join(lines), encoding="utf-8")


def _print_plan(
    cfg: dict,
    seeds: list[int],
    groups: list[str],
    out: Path,
    baseline: float,
    baseline_source: str,
    log: bool,
) -> None:
    k = int(cfg.get("split", {}).get("n_splits", 5))
    splits = _resolve(cfg.get("paths", {}).get("splits", "splits/fold_v1.json"))
    print("[stability] DRY-RUN（不训练，也不写任何文件）")
    print(f"  配置      : {cfg.get('_path', 'N/A')}")
    print(f"  特征组    : {'+'.join(groups) if groups else 'base'}")
    print(f"  划分      : {splits}（{'存在' if splits.exists() else '缺失！'}，{k} 折）")
    print(f"  seeds     : {', '.join(map(str, seeds))}  ->  {len(seeds)} × {k} = {len(seeds) * k} 次拟合")
    print(f"  主模型参考: {baseline:.6f}（{baseline_source}）")
    print("  预计耗时  : 每 seed 约 30–60 秒（70 万行 × 5 折），合计约 "
          f"{len(seeds) * 0.75:.1f} 分钟")
    print(f"  输出      : {out / 'stability.json'} / {out / 'p3_seed_variance.png'} / {out / 'stability.md'}")
    print(f"  是否登记  : {'是（写 exp_log 与 OOF）' if log else '否（--no-log）'}")
    print("  真正执行  : 去掉 --dry-run")


def run_stability(
    cfg: dict,
    *,
    seeds: list[int],
    groups: list[str],
    out_dir: str | Path = "reports/eval",
    dry_run: bool = False,
    baseline: float | None = None,
    log: bool = True,
) -> dict:
    """跑多 seed × 5 折并写 stability.json / p3_seed_variance.png / stability.md。"""
    if not groups:
        raise ValueError("[stability] --groups 不能为空")
    baseline_auc, baseline_source = resolve_baseline(cfg, baseline)
    out = _resolve(out_dir)

    if dry_run:
        _print_plan(cfg, seeds, groups, out, baseline_auc, baseline_source, log)
        return {
            "dry_run": True,
            "seeds": seeds,
            "groups": groups,
            "baseline_auc": baseline_auc,
            "baseline_source": baseline_source,
        }

    out.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    print(
        f"[stability] {len(seeds)} seeds × {int(cfg.get('split', {}).get('n_splits', 5))} 折"
        f"  |  groups={'+'.join(groups)}  |  baseline={baseline_auc:.6f}"
    )

    for i, seed in enumerate(seeds, start=1):
        print(f"\n[stability] ({i}/{len(seeds)}) seed={seed}")
        t0 = time.time()
        try:
            # 直接复用 P2 的训练入口：FeatureBuilder 已负责 str -> category 与全量统计，
            # T 组目标编码仍由 run_cv 逐折 fit，稳定性实验与主模型口径完全一致。
            res = run_cv(
                cfg,
                model="lgbm",
                groups=groups,
                seed=int(seed),
                log=bool(log),
                verbose=True,
                owner="P3",
                stage="eval",
                tag="stab",
                notes=f"P3 稳定性 seed={seed}",
            )
        except SystemExit:
            raise
        except Exception as e:
            raise RuntimeError(
                f"seed={seed} 训练失败：{type(e).__name__}: {e}\n"
                "  排查顺序：1) data/processed/train.parquet 是否存在；"
                "2) splits/fold_v1.json 是否完整；3) FeatureBuilder 输出的类别列是否正常。"
            ) from e

        elapsed = time.time() - t0
        fold_aucs = [float(x) for x in res.get("fold_aucs", [])]
        record = {
            "seed": int(seed),
            "run_id": str(res.get("run_id", "")),
            "cv_mean": float(res["cv_mean"]),
            "cv_std": float(res["cv_std"]),
            "fold_aucs": fold_aucs,
            "n_features": int(res.get("n_features", 0)),
            "delta_vs_baseline": float(res["cv_mean"]) - baseline_auc,
            "seconds": round(elapsed, 1),
        }
        runs.append(record)
        print(
            f"[stability] seed={seed} -> OOF AUC={record['cv_mean']:.6f} "
            f"± {record['cv_std']:.6f}  Δ={record['delta_vs_baseline']:+.6f}  "
            f"({elapsed:.1f}s)"
        )

    summary = summarize_runs(runs, baseline_auc)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": cfg.get("_path", ""),
        "groups": groups,
        "seeds": seeds,
        "n_splits": int(cfg.get("split", {}).get("n_splits", 5)),
        "baseline_auc": baseline_auc,
        "baseline_source": baseline_source,
        "runs": runs,
        "summary": summary,
    }
    write_json(out / "stability.json", payload)
    _plot_seed_variance(runs, summary, baseline_auc, out / "p3_seed_variance.png")
    _write_markdown(out, runs, summary, baseline_auc, baseline_source, seeds, groups)

    print(
        f"\n[stability] seed 均值 {summary['seed_mean_auc']:.6f}  "
        f"seed std {summary['seed_std_auc']:.6f}  "
        f"极差 {summary['seed_range']:.6f}"
    )
    print(f"[stability] 输出 -> {out}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="P3 稳定性：多 seed × 5 折")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seeds", default="42,2026,7", help="逗号分隔，例如 42,2026,7")
    ap.add_argument("--groups", default="base,C,T", help="逗号分隔特征组")
    ap.add_argument("--out", default="reports/eval")
    ap.add_argument("--baseline", type=float, default=None,
                    help="主模型 OOF AUC；默认从 exp_log 的 C+T main 行读取")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不训练、不写文件")
    ap.add_argument("--no-log", action="store_true",
                    help="不写 exp_log / OOF（默认写，便于按 run_id 溯源）")
    args = ap.parse_args()

    try:
        seeds = parse_seeds(args.seeds)
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
        cfg = load_config(args.config)
        run_stability(
            cfg,
            seeds=seeds,
            groups=groups,
            out_dir=args.out,
            dry_run=args.dry_run,
            baseline=args.baseline,
            log=not args.no_log,
        )
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"[stability] {e}") from None
    except RuntimeError as e:
        raise SystemExit(f"[stability] {e}") from None


if __name__ == "__main__":
    main()
