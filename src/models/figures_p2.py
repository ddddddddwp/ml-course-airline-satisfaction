"""P2 必备图表生成（plan.md 7.1）：结构图 / 学习曲线 / 模型对比表 / 超参敏感性。

读取：
  - reports/exp_log.csv        （P2 自有实验：主力 LightGBM、LR 对照、num_leaves 敏感性）
  - reports/figures/p2_learning_curve.json （diag.py 产出；缺失则跳过学习曲线）
输出（reports/figures/，前缀 p2_ 避免与 P3 图表冲突）：
  - p2_structure.png          结构图
  - p2_learning_curve.png     学习曲线（train/valid AUC vs boosting round）
  - p2_compare_table.png      模型对比表
  - p2_sensitivity.png        超参敏感性（num_leaves vs OOF AUC）
全部为 P2 写域内的产物；不修改任何 P1/P3 代码。

可移植性与健壮性：
  - 中文字体跨平台探测（Windows 微软雅黑/黑体 → Linux Noto Sans CJK/文泉驿 →
    macOS PingFang/Heiti）；都找不到时退回默认字体，图内文字改用英文，避免方块。
  - 敏感性基准线取自 exp_log.csv（owner=P2 且 run_id 含 "_main"），不写死分数。
  - 学习曲线 JSON 缺失 / 损坏时跳过该图，不抛异常。
  - 全部图 300 dpi（plan.md D17 图表规范化要求）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DPI = 300

# 已知中文字体文件（存在才注册）与常见字体族名，按优先级排列
_FONT_FILES = (
    ("Microsoft YaHei", "C:/Windows/Fonts/msyh.ttc"),
    ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
    ("Noto Sans CJK SC", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ("Noto Sans CJK SC", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ("Noto Sans CJK SC", "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"),
    ("Source Han Sans SC", "/usr/share/fonts/adobe-source-han-sans/SourceHanSansSC-Regular.otf"),
    ("WenQuanYi Zen Hei", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ("PingFang SC", "/System/Library/Fonts/PingFang.ttc"),
    ("Heiti SC", "/System/Library/Fonts/STHeiti Light.ttc"),
    ("Heiti SC", "/System/Library/Fonts/STHeiti Medium.ttc"),
    ("Hiragino Sans GB", "/System/Library/Fonts/Hiragino Sans GB.ttc"),
)
_FONT_FAMILIES = (
    "Microsoft YaHei", "SimHei",
    "Noto Sans CJK SC", "Noto Sans SC", "Noto Sans CJK JP",
    "Source Han Sans SC", "Source Han Sans CN",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
    "PingFang SC", "Heiti SC", "STHeiti", "Hiragino Sans GB",
    "Arial Unicode MS",
)


def _setup_cjk_font() -> str | None:
    """注册/探测中文字体；返回首选字体族名，找不到返回 None（图内改用英文）。"""
    extra: list[str] = []
    for _name, path in _FONT_FILES:
        p = Path(path)
        if not p.exists():
            continue
        try:
            fm.fontManager.addfont(str(p))
        except Exception:
            continue
        # 不同发行版/系统的字体族名可能和文件名不同，以 matplotlib 解析结果为准
        try:
            real = fm.FontProperties(fname=str(p)).get_name()
        except Exception:
            real = _name
        if real and real not in extra:
            extra.append(real)

    installed = {f.name for f in fm.fontManager.ttflist}
    chain: list[str] = []
    for cand in extra + list(_FONT_FAMILIES):
        if cand in installed and cand not in chain:
            chain.append(cand)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    if chain:
        plt.rcParams["font.sans-serif"] = chain + ["DejaVu Sans"]
        print(f"[fig] 中文字体: {chain[0]}")
        return chain[0]
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    print("[fig][警告] 未找到中文字体（msyh/simhei/Noto Sans CJK/PingFang/Heiti），"
          "图内文字改用英文，避免显示为方块。")
    return None


_CJK_FONT = _setup_cjk_font()
HAS_CJK = _CJK_FONT is not None


def _t(zh: str, en: str) -> str:
    """有中文字体用中文，否则用英文（避免图中出现方块）。"""
    return zh if HAS_CJK else en


def _load_p2() -> pd.DataFrame | None:
    p = ROOT / "reports" / "exp_log.csv"
    if not p.exists():
        print(f"[fig][提示] 找不到 {p}，跳过依赖实验记录的图")
        return None
    df = pd.read_csv(p)
    if "owner" not in df.columns:
        print(f"[fig][提示] {p.name} 缺 owner 列，跳过依赖实验记录的图")
        return None
    missing = [c for c in ("run_id", "cv_mean") if c not in df.columns]
    if missing:
        print(f"[fig][提示] {p.name} 缺列 {missing}，跳过依赖实验记录的图")
        return None
    p2 = df[df["owner"] == "P2"].copy()
    if p2.empty:
        print("[fig][提示] exp_log.csv 里没有 owner=P2 的记录，跳过依赖实验记录的图")
        return None
    return p2


def _main_baseline(df: pd.DataFrame) -> tuple[float, str] | None:
    """主力模型基准线：owner=P2 且 run_id 含 "_main" 的最后一条 cv_mean。"""
    if "run_id" not in df.columns or "cv_mean" not in df.columns:
        print("[fig][提示] exp_log 缺 run_id/cv_mean 列，敏感性图不画基准线")
        return None
    hit = df[df["run_id"].astype(str).str.contains("_main", na=False)]
    if hit.empty:
        print("[fig][提示] exp_log 找不到 owner=P2 且 run_id 含 '_main' 的记录，"
              "敏感性图不画基准线")
        return None
    row = hit.iloc[-1]
    try:
        val = float(row["cv_mean"])
    except (TypeError, ValueError):
        print(f"[fig][提示] {row['run_id']} 的 cv_mean 不是数值，敏感性图不画基准线")
        return None
    print(f"[fig] 敏感性基准线 -> {row['run_id']}  cv_mean={val:.5f}")
    return val, str(row["run_id"])


def fig_structure():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333"):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.04",
                     fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#1a1a1a")

    box(0.02, 0.40, 0.20, 0.20,
        _t("输入特征 x\n(25维: base+C+T)", "Input features x\n(25 dims: base+C+T)"),
        "#dbeafe", "#2563eb")
    box(0.30, 0.62, 0.16, 0.16, _t("树 f1", "tree f1"), "#dcfce7", "#16a34a")
    box(0.30, 0.40, 0.16, 0.16, _t("树 f2", "tree f2"), "#dcfce7", "#16a34a")
    box(0.30, 0.18, 0.16, 0.16, _t("树 fK", "tree fK"), "#dcfce7", "#16a34a")
    box(0.54, 0.40, 0.18, 0.20, _t("Σ fk(x)\n(加权和)", "Σ fk(x)\n(weighted sum)"),
        "#fef9c3", "#ca8a04")
    box(0.78, 0.40, 0.18, 0.20, "σ(·)\n→ P(satisfied)", "#fce7f3", "#db2777")

    ax.annotate("", xy=(0.30, 0.70), xytext=(0.22, 0.50),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.4))
    ax.annotate("", xy=(0.30, 0.48), xytext=(0.22, 0.50),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.4))
    ax.annotate("", xy=(0.30, 0.26), xytext=(0.22, 0.50),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.4))
    ax.annotate("", xy=(0.54, 0.50), xytext=(0.46, 0.50),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.4))
    ax.annotate("", xy=(0.78, 0.50), xytext=(0.72, 0.50),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.4))

    ax.text(0.39, 0.86,
            _t("K 棵决策树（梯度提升集成）", "K decision trees (gradient boosting ensemble)"),
            ha="center", fontsize=10, color="#16a34a", fontweight="bold")
    ax.text(0.5, 0.05,
            _t("P = σ( Σk fk(x) ) ，fk 为第 k 棵树对样本的打分（叶值加和）",
               "P = σ( Σk fk(x) ), fk = score of the k-th tree (sum of leaf values)"),
            ha="center", fontsize=9.5, color="#444")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "p2_structure.png", dpi=DPI)
    plt.close(fig)
    print("[fig] 结构图 ->", FIG / "p2_structure.png")


def fig_learning_curve():
    src = FIG / "p2_learning_curve.json"
    if not src.exists():
        print(f"[fig][跳过] 找不到 {src}（先跑 python -m src.models.diag），跳过学习曲线")
        return
    try:
        j = json.loads(src.read_text(encoding="utf-8"))
        r = j["round"]
        train_auc = [float(v) for v in j["train_auc"]]
        valid_auc = [float(v) for v in j["valid_auc"]]
        bi = int(j["best_iter"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"[fig][跳过] {src.name} 读取失败（{e}），跳过学习曲线")
        return
    if not r or not train_auc or not valid_auc:
        print(f"[fig][跳过] {src.name} 没有曲线数据，跳过学习曲线")
        return

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(r, train_auc, label="train AUC", color="#2563eb", lw=1.6)
    ax.plot(r, valid_auc, label="valid AUC", color="#db2777", lw=1.6)
    ax.axvline(bi, ls="--", color="#ca8a04", lw=1.2)
    ax.text(bi, min(train_auc), f" early_stop≈{bi}", color="#ca8a04", fontsize=9)
    ax.set_xlabel("boosting round")
    ax.set_ylabel("AUC")
    ax.set_title(_t("LightGBM 学习曲线（fold 0, base+C+T）",
                    "LightGBM learning curve (fold 0, base+C+T)"))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    # 保持原纵轴下限 0.90；数据更低时自动扩展，避免曲线被裁掉
    ax.set_ylim(min(0.90, min(train_auc + valid_auc) - 0.005), 1.001)
    fig.tight_layout()
    fig.savefig(FIG / "p2_learning_curve.png", dpi=DPI)
    plt.close(fig)
    print("[fig] 学习曲线 ->", FIG / "p2_learning_curve.png")


def fig_sensitivity():
    df = _load_p2()
    if df is None:
        return
    sens = df[df["run_id"].astype(str).str.contains("sens_nl", na=False)].copy()
    if sens.empty:
        print("[fig][提示] exp_log 里没有 sens_nl* 记录，跳过超参敏感性图")
        return
    sens["nl"] = sens["run_id"].astype(str).str.extract(r"sens_nl(\d+)").astype(int)
    sens = sens.sort_values("nl")

    base = _main_baseline(df)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sens["nl"], sens["cv_mean"], "o-", color="#16a34a", lw=1.8, ms=8)
    for _, row in sens.iterrows():
        ax.annotate(f"{row['cv_mean']:.5f}", (row["nl"], row["cv_mean"]),
                    textcoords="offset points", xytext=(0, 9), fontsize=9, ha="center")
    if base is not None:
        ax.axhline(base[0], ls="--", color="#db2777", lw=1.2,
                   label=_t(f"主力 {base[0]:.5f}", f"main model {base[0]:.5f}"))
        ax.legend()
    ax.set_xlabel("num_leaves")
    ax.set_ylabel("OOF AUC")
    ax.set_title(_t("超参敏感性：num_leaves（其余默认超参）",
                    "Hyperparameter sensitivity: num_leaves (other params default)"))
    ax.grid(alpha=0.3)
    # 保持原纵轴范围；有数据落在范围外时自动扩展
    yvals = [float(v) for v in sens["cv_mean"]]
    if base is not None:
        yvals.append(base[0])
    ax.set_ylim(min(0.959, min(yvals) - 5e-4), max(0.961, max(yvals) + 5e-4))
    fig.tight_layout()
    fig.savefig(FIG / "p2_sensitivity.png", dpi=DPI)
    plt.close(fig)
    print("[fig] 超参敏感性 ->", FIG / "p2_sensitivity.png")


def fig_compare_table():
    df = _load_p2()
    if df is None:
        return
    rows = []
    for _, r in df.iterrows():
        run_id = str(r["run_id"])
        if run_id.endswith("_main") or run_id.endswith("_contrast"):
            model = "LightGBM" if r["model"] == "lgbm" else "LogisticRegression"
            typ = (_t("梯度提升树集成", "Gradient boosting ensemble")
                   if r["model"] == "lgbm" else _t("线性模型", "Linear model"))
            rows.append([model, typ, r["feat_ver"], f"{r['cv_mean']:.5f}",
                         _t("P2 自有", "P2 own")])
    if not rows:
        print("[fig][提示] exp_log 里没有 *_main / *_contrast 记录，跳过对比表")
        return

    header = [_t("模型", "Model"), _t("类型", "Type"), _t("特征组", "Features"),
              "OOF AUC", _t("来源", "Source")]
    tbl = pd.DataFrame(rows, columns=header)
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.axis("off")
    t = ax.table(cellText=tbl.values, colLabels=tbl.columns,
                 cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 1.8)
    for (ri, ci), cell in t.get_celld().items():
        if ri == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", fontweight="bold")
        elif tbl.values[ri - 1][0] == "LightGBM":
            cell.set_facecolor("#dcfce7")
    ax.set_title(_t("模型对比总表（P2 自有实验，同折 fold_v1 / base+C+T）",
                    "Model comparison (P2 own runs, same folds fold_v1 / base+C+T)"),
                 fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIG / "p2_compare_table.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("[fig] 模型对比表 ->", FIG / "p2_compare_table.png")


if __name__ == "__main__":
    fig_structure()
    fig_learning_curve()
    fig_sensitivity()
    fig_compare_table()
    print("全部 P2 图表生成完成。")
