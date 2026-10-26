# 航空乘客满意度预测

Kaggle Playground Series **S6E10 — Predicting Airline Satisfaction** 的课程项目仓库。
三人组，四周，一份答辩 PPT + 三份独立课程报告。

---

## 结果速览

5 折 Stratified CV，OOF ROC AUC（`splits/fold_v1.json`，seed 42）：

| 特征组 | 特征数 | LightGBM | Δ | 逻辑回归 | Δ |
|---|---:|---:|---:|---:|---:|
| base（原始 21 特征） | 22 | 0.95883 | — | 0.92155 | — |
| +A 语义修正（0=不适用） | 35 | 0.95880 | −0.00003 | **0.94081** | **+0.01926** |
| +B 聚合 / 交互 | 46 | 0.95864 | −0.00019 | **0.94386** | **+0.02231** |
| +C 频次编码 | 24 | 0.95941 | +0.00058 | 0.92164 | +0.00009 |
| +C+T 目标编码（折内拟合） | 25 | **0.96019** | **+0.00136** | 0.92330 | +0.00175 |
| 全部相加 | 49 | 0.96009 | +0.00126 | **0.94516** | **+0.02361** |

**核心结论**：同一组特征，在两个模型族上的价值相差两个数量级。

- **对树模型**：聚合类特征 A/B 是**负增益**，因为非单调关系它自己就能学到；有效的只有频次 + 目标编码（+0.0014）。
- **对线性模型**：A/B 值 **+0.019 ~ +0.022**，因为"0 分 = 不适用"这个非单调语义它表达不出来。
- **全加不一定最好**：LightGBM 上 `A+B+C+T`（0.96009）反而低于 `C+T`（0.96019）。

> 一句话：特征工程的价值不在特征本身，而在**模型能不能表达它**。

---

## 快速开始

```bash
git clone git@github.com:ddddddddwp/ml-course-airline-satisfaction.git
cd ml-course-airline-satisfaction
pip install -r requirements.txt

# 没有数据也能先验证整条链路（用同结构的合成数据，约 40 秒）
make smoke

# 有数据之后的正式流程
make data      # raw -> processed（清洗 + 校验和）
make split     # 冻结划分 -> splits/fold_v1.json
make train     # 5 折基线（约 32 秒）
```

单轮 5 折在 CPU 上 32 秒，不需要 GPU。

---

## 数据（不在仓库里）

数据体积 92 MB，已加入 `.gitignore`。两条获取路径，任选其一：

**A. 官方（推荐，需 Kaggle 登录）**

```bash
# 先把 kaggle.json 放到 ~/.kaggle/
kaggle competitions download -c playground-series-s6e10 -p data/raw
cd data/raw && unzip -o '*.zip' && cd -
```

**B. 公开镜像（CC BY 4.0，无需登录）**

本仓库早期复现用的是公开数据集 `vijaikm/s6e10-downcasted-parquet-5fold`，
与官方数据逐位等价——用 `base + LightGBM` 跑出的 OOF AUC **0.95883**，
与独立公开分析报告的 0.95883 **一致到小数点后 5 位**。

```bash
curl -L -o /tmp/s6e10.zip \
  "https://www.kaggle.com/api/v1/datasets/download/vijaikm/s6e10-downcasted-parquet-5fold"
unzip -o /tmp/s6e10.zip -d /tmp/s6e10
python - <<'EOF'
import pandas as pd
tr = pd.read_parquet("/tmp/s6e10/train.parquet").drop(columns=["fold"])
te = pd.read_parquet("/tmp/s6e10/test.parquet")
tr.to_csv("data/raw/train.csv", index=False)
te.to_csv("data/raw/test.csv", index=False)
pd.DataFrame({"id": te["id"], "satisfaction": 0.4436}).to_csv(
    "data/raw/sample_submission.csv", index=False)
EOF
```

若课程要求"必须用官方数据"，请用 A 方案重下后重跑 `make data`，预计数字一致。

---

## 目录结构

```
├── Makefile                 # 所有入口：data / split / train / abl / smoke
├── configs/
│   ├── base.yaml            # 正式实验
│   └── toy.yaml             # 冒烟测试（不碰真实数据）
├── data/
│   ├── README.md            # 数据字典 + 六项检查 + 泄漏风险（P1 维护）
│   ├── raw/                 # 只读，不进 git
│   └── processed/           # 可重建
├── splits/fold_v1.json      # 冻结划分，全员唯一来源
├── src/
│   ├── common/split.py      # 4 种划分策略 + 冻结 + 三方案对比
│   ├── data/                # 清洗、列类型探测
│   ├── features/            # 特征组 A/B/C + 折内目标编码
│   ├── models/train.py      # 5 折训练骨架（P2 拥有）
│   └── eval/                # 评估（P3 拥有）
├── reports/
│   ├── exp_log.csv          # 实验登记表：每个数字可追溯
│   └── oof/                 # OOF 预测，供误差分析
└── docs/
    ├── team-plan.md         # 项目计划与 WBS
    ├── p1-plan.md           # P1 的闭环说明与任务清单
    └── topic-s6e10.md       # 选题冻结稿
```

---

## 分工

| 代号 | 责任域 | 唯一写权限 |
|---|---|---|
| **P1** | 数据与特征 | `data/`、`src/data/`、`src/features/`、`splits/` |
| **P2** | 模型与训练 | `src/models/`、`configs/exp_p2_*` |
| **P3** | 评估与交付 | `src/eval/`、`reports/figures/` |

`src/common/` 改动需两人 review。

---

## 五条铁律

1. `splits/fold_v1.json` 冻结后**只读**。谁改，谁的历史实验作废。
2. 所有实验必须 `load_folds()` 取折，**禁止自己 `train_test_split`**。
3. 目标编码只在**训练折内** `fit`，验证折只 `transform`。
4. 提交的是**概率**，不是 0/1 标签。
5. 报告里的每个数字都要能按 `run_id` 查到、按 `git_commit` 复现。

```python
from src.common.split import load_folds

folds, meta = load_folds()          # [(train_idx, valid_idx), ...]
for tr, va in folds:
    ...
```

---

## 任务与进度

完整 WBS（55 项，含依赖与验收标准）：[`docs/team-plan.md`](docs/team-plan.md) · [`wbs.csv`](wbs.csv)

当前进度：W1 基线打通，`fold_v1.json` 已冻结，base = **0.95883 ± 0.00065**。
