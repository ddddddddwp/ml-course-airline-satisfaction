"""P2 必备图表生成（plan.md 7.1）：结构图 / 学习曲线 / 模型对比表 / 超参敏感性。

读取：
  - reports/exp_log.csv        （P2 自有实验：主力 LightGBM、LR 对照、num_leaves 敏感性）
  - reports/figures/p2_learning_curve.json （diag.py 产出）
输出（reports/figures/，前缀 p2_ 避免与 P3 图表冲突）：
  - p2_structure.png          结构图
  - p2_learning_curve.png     学习曲线（train/valid AUC vs boosting round）
  - p2_compare_table.png      模型对比表
  - p2_sensitivity.png        超参敏感性（num_leaves vs OOF AUC）
全部为 P2 写域内的产物；不修改任何 P1/P3 代码。
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

# 中文字体（Windows 自带 Microsoft YaHei），避免图中汉字显示为方块
for _fp in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    try:
        fm.fontManager.addfont(_fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _load_p2():
    df = pd.read_csv(ROOT / "reports" / "exp_log.csv")
    return df[df["owner"] == "P2"].copy()


def fig_structure():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333"):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.04",
                     fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#1a1a1a")

    box(0.02, 0.40, 0.20, 0.20, "输入特征 x\n(25维: base+C+T)", "#dbeafe", "#2563eb")
    box(0.30, 0.62, 0.16, 0.16, "树 f1", "#dcfce7", "#16a34a")
    box(0.30, 0.40, 0.16, 0.16, "树 f2", "#dcfce7", "#16a34a")
    box(0.30, 0.18, 0.16, 0.16, "树 fK", "#dcfce7", "#16a34a")
    box(0.54, 0.40, 0.18, 0.20, "Σ fk(x)\n(加权和)", "#fef9c3", "#ca8a04")
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

    ax.text(0.39, 0.86, "K 棵决策树（梯度提升集成）", ha="center",
            fontsize=10, color="#16a34a", fontweight="bold")
    ax.text(0.5, 0.05, "P = σ( Σk fk(x) ) ，fk 为第 k 棵树对样本的打分（叶值加和）",
            ha="center", fontsize=9.5, color="#444")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "p2_structure.png", dpi=150)
    plt.close(fig)
    print("[fig] 结构图 ->", FIG / "p2_structure.png")


def fig_learning_curve():
    j = json.loads((FIG / "p2_learning_curve.json").read_text())
    r = j["round"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(r, j["train_auc"], label="train AUC", color="#2563eb", lw=1.6)
    ax.plot(r, j["valid_auc"], label="valid AUC", color="#db2777", lw=1.6)
    bi = j["best_iter"]
    ax.axvline(bi, ls="--", color="#ca8a04", lw=1.2)
    ax.text(bi, min(j["train_auc"]), f" early_stop≈{bi}", color="#ca8a04", fontsize=9)
    ax.set_xlabel("boosting round")
    ax.set_ylabel("AUC")
    ax.set_title("LightGBM 学习曲线（fold 0, base+C+T）")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.90, 1.001)
    fig.tight_layout()
    fig.savefig(FIG / "p2_learning_curve.png", dpi=150)
    plt.close(fig)
    print("[fig] 学习曲线 ->", FIG / "p2_learning_curve.png")


def fig_sensitivity():
    df = _load_p2()
    sens = df[df["run_id"].str.contains("sens_nl")].copy()
    sens["nl"] = sens["run_id"].str.extract(r"sens_nl(\d+)").astype(int)
    sens = sens.sort_values("nl")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sens["nl"], sens["cv_mean"], "o-", color="#16a34a", lw=1.8, ms=8)
    for _, row in sens.iterrows():
        ax.annotate(f"{row['cv_mean']:.5f}", (row["nl"], row["cv_mean"]),
                    textcoords="offset points", xytext=(0, 9), fontsize=9, ha="center")
    ax.axhline(0.96019, ls="--", color="#db2777", lw=1.2, label="主力 0.96019")
    ax.set_xlabel("num_leaves")
    ax.set_ylabel("OOF AUC")
    ax.set_title("超参敏感性：num_leaves（其余默认超参）")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.959, 0.961)
    fig.tight_layout()
    fig.savefig(FIG / "p2_sensitivity.png", dpi=150)
    plt.close(fig)
    print("[fig] 超参敏感性 ->", FIG / "p2_sensitivity.png")


def fig_compare_table():
    df = _load_p2()
    rows = []
    for _, r in df.iterrows():
        if r["run_id"].endswith("_main") or r["run_id"].endswith("_contrast"):
            model = "LightGBM" if r["model"] == "lgbm" else "LogisticRegression"
            typ = "梯度提升树集成" if r["model"] == "lgbm" else "线性模型"
            rows.append([model, typ, r["feat_ver"], f"{r['cv_mean']:.5f}", "P2 自有"])
    tbl = pd.DataFrame(rows, columns=["模型", "类型", "特征组", "OOF AUC", "来源"])
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
    ax.set_title("模型对比总表（P2 自有实验，同折 fold_v1 / base+C+T）",
                 fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIG / "p2_compare_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[fig] 模型对比表 ->", FIG / "p2_compare_table.png")


if __name__ == "__main__":
    fig_structure()
    fig_learning_curve()
    fig_sensitivity()
    fig_compare_table()
    print("全部 P2 图表生成完成。")
