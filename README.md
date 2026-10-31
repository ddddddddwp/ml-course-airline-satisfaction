# 航空乘客满意度预测

Kaggle Playground Series **S6E10 — Predicting Airline Satisfaction** 的课程项目仓库。
三人组，四周，一份答辩 PPT + 三份独立课程报告。

---

## 结果速览

5 折 Stratified CV，OOF ROC AUC（`splits/fold_v1.json`，seed 42）：

| 特征组 | 特征数 | LightGBM | Δ | 逻辑回归 | Δ |
|---|---:|---:|---:|---:|---:|
| base（原始 21 特征） | 22 | 0.958829 | — | 0.921550 | — |
| +A 语义修正（0=不适用） | 35 | 0.958800 | −0.000029 | **0.940807** | **+0.019257** |
| +B 聚合 / 交互 | 46 | 0.958650 | −0.000179 | **0.943866** | **+0.022316** |
| +C 频次编码 | 24 | 0.959413 | +0.000584 | 0.921643 | +0.000093 |
| +C+T 目标编码（折内拟合） | 25 | **0.960191** | **+0.001362** | 0.923300 | +0.001750 |
| 全部相加 | 49 | 0.960048 | +0.001219 | **0.945162** | **+0.023612** |

> B 组 `dist_bin` 改为全量固定边界后重跑（`P1_feat_lgbm_B_1031_s42_abl2` / `P1_feat_lgbm_ABCT_1031_s42_abl2` / `P1_feat_lr_B_1031_s42_abl2` / `P1_feat_lr_ABCT_1031_s42_abl2`）；主模型 `base+C+T` 不含 B，主分数不变。

**核心结论**：同一组特征，在两个模型族上的价值相差两个数量级。

- **对树模型**：聚合类特征 A/B 是**负增益**，因为非单调关系它自己就能学到；有效的只有频次 + 目标编码（+0.0014）。
- **对线性模型**：A/B 值 **+0.019 ~ +0.022**，因为"0 分 = 不适用"这个非单调语义它表达不出来。
- **全加不一定最好**：LightGBM 上 `A+B+C+T`（0.960048）反而低于 `C+T`（0.960191）。

> 一句话：特征工程的价值不在特征本身，而在**模型能不能表达它**。

### 最终交付与 P3 复核

| 项 | 值 |
|---|---|
| 最终 run_id | `P2_final_lgbm_CT_1031_s42_final`（seed 42，stage=final） |
| 最终模型 OOF AUC | **0.960191**（5 折 AUC 平均，`exp_log.cv_mean`） |
| P3 复评 pooled AUC | **0.960186**，bootstrap 95% CI [0.959715, 0.960605]（n_boot=200） |
| 概率质量 | LogLoss 0.225458；Brier 0.060432；ECE 0.002479（10 箱） |
| 阈值 0.5 描述指标 | Acc 0.9264 / Precision 0.9272 / Recall 0.9051 / F1 0.9160 |
| 混淆矩阵 | TN 367,246 / FP 22,050 / FN 29,440 / TP 280,899 |
| 3 seeds（42 / 2026 / 7） | 0.960191 / 0.960159 / 0.960203；均值 **0.960184**，seed std **0.000023**，极差 0.000045 |
| 3 seeds 平均折内 std | 0.000631 |

- 0.960191 与 0.960186 是同一模型、同一份 OOF 的两种算法：前者是 **5 折 AUC 的算术平均**（`exp_log.cv_mean`），后者是把 5 折 OOF 预测拼起来算的 **pooled AUC**（`metrics.json`）；不是两个模型，也不是舍入误差。
- 提交文件 `reports/submission_final.csv` 共 **299,844 行概率**、无 NaN、未阈值化；与修复前提交逐位一致（`max_abs_diff=0`）。
- **实验日志里的单点最高是 `nl127=0.960254`，但它不是“当前最好”**：只比主模型高 +0.000063，与 3 seeds 极差 0.000045 同量级，且远小于 AUC 的 bootstrap 95% CI 半宽 0.00045。单 seed、单配置的这点差异属于噪声范围，对外统一口径仍是 **0.960191**。

### 另外三个实验结论

- **划分鲁棒性**：KFold / Stratified / GroupKFold（按距离段分组）的 CV AUC 分别是
  0.958770 / 0.958829 / 0.957784。按距离段分组只掉约 0.001，说明模型不是靠记住具体航线，而是学到了可迁移的服务信号。
- **事后特征**：剔除 `Arrival Delay`（飞行后才可知）只掉 **0.000034**（0.960191 → 0.960157）——它与起飞延误 r≈0.84、87% 行完全相等，属于冗余。
  **结论：最终模型可以只使用飞行前即可获得的信息，业务上可部署。**
- **特征重要性要分两种口径**：
  - LightGBM **gain**（P1）：Online boarding 42.08%、Inflight wifi 12.77%、Type of Travel 10.65%、Class 8.97%、Inflight entertainment 5.50%；
  - **SHAP**（P3，OOF-TE）：Online boarding 20.24%、Type of Travel 14.69%、Inflight wifi 12.39%、Customer Type 8.44%、Class 8.30%；
    `Flight Distance__te` 的 SHAP 占 5.82%，而 gain 只有 1.06%——gain 回答“在哪分裂”，SHAP 回答“把预测推了多少”，不能互相替代。

### P3 新增结论（稳定性 / 可解释性 / 误差）

- **稳定性**：换 seed 的波动（std 0.000023）远小于折间波动（0.000631，约 27 倍），所以“冻结同一把尺子”比多跑 seed 更关键；单 seed 的 +0.001 必须谨慎表述。
- **误差**：0.5 阈值下错误率 7.36%（51,490 条）；高置信 FP 14,497、高置信 FN 16,146；错误率最高的群是 **Online boarding=4（12.0%）**，属于评分组合处于决策边界的样本，而不是“差评样本”。
- **校准**：ECE 0.0025，概率可直接提交，不需要 Platt / isotonic 后处理。

---

## 快速开始

> **Windows 环境通常没有 `make`**。下面给出等价的 `python -m ...` / `python scripts/...` 命令，PowerShell 与 cmd 均可执行。
> 所有命令都在仓库根目录运行，并先设置 `PYTHONPATH`：

```powershell
# PowerShell（Windows）
$env:PYTHONPATH="."
pip install -r requirements-dev.txt          # 运行依赖 + pytest
```

```bash
# bash / Git Bash / Linux / macOS
export PYTHONPATH=.
pip install -r requirements-dev.txt
```

### 正式流程（有数据）

| 步骤 | 有 make | Windows 无 make（等价命令） |
|---|---|---|
| 清洗 raw → processed | `make data` | `python -m src.data.make_dataset --config configs/base.yaml` |
| 冻结划分 | `make split` | `python -m src.common.split freeze --config configs/base.yaml` |
| 5 折基线训练 | `make train` | `python -m src.models.train --config configs/base.yaml --model lgbm --groups base` |
| 最终模型 + submission | — | `python -m src.models.predict --config configs/base.yaml --model lgbm --groups base,C,T --tag final` |
| P1 全部交付物 | `make p1` | 见下方“P1 / P3 逐项命令” |
| 测试 | `make test` | `python -m pytest -q` |
| 验收 | `make verify` | `python scripts/verify.py --config configs/base.yaml` |
| P3 全部评估 | `make p3` | 见下方“P1 / P3 逐项命令” |
| 清理派生数据 | `make clean` | `python scripts/clean.py` |

两条容易踩的规则：

1. **`make split` 在 `splits/fold_v1.json` 已存在时默认跳过（exit 0）**，不会覆盖历史划分；确需重划要加 `--force`，旧文件会先备份成 `splits/fold_v1.json.bak.<时间戳>`。这样 `make all` 可以反复执行。
2. **`make data` 有校验和闸门**：先按 `data/raw/CHECKSUMS.txt` 重算哈希，不一致直接报错退出；只有确认“数据是有意更换”时才用 `python -m src.data.make_dataset --config configs/base.yaml --accept-data-change` 重新登记。

### P1 / P3 逐项命令

```powershell
$env:PYTHONPATH="."

# P1：划分对比 / EDA / 重要性 / 特征字典 / 事后特征实验
python -m src.common.split compare --config configs/base.yaml --score
python scripts/eda.py --config configs/base.yaml
python scripts/plot_importance.py --config configs/base.yaml --groups base,C,T
python scripts/gen_feature_dict.py --config configs/base.yaml
python -m src.models.train --config configs/base.yaml --model lgbm --groups base,C,T --drop-cols "Arrival Delay in Minutes,arr_delay_missing" --tag noarr

# P3：评估 / 稳定性 / SHAP / badcase
python -m src.eval.evaluate --config configs/base.yaml
python -m src.eval.stability --config configs/base.yaml --seeds 42,2026,7
python -m src.eval.interpret --config configs/base.yaml
python -m src.eval.badcase --config configs/base.yaml
```

单轮 5 折在 CPU 上约 32 秒，不需要 GPU；3 seeds × 5 折约 2–3 分钟。

### 没有数据也能先跑通链路

```bash
# 有 make 时
make smoke

# Windows 无 make：合成数据 → 清洗 → 冻结 → 训练
python scripts/make_toy_data.py --out data/toy/raw
python -m src.data.make_dataset --config configs/toy.yaml
python -m src.common.split freeze --config configs/toy.yaml
python -m src.models.train --config configs/toy.yaml --model lgbm --groups base,A,B,C --tag smoke
```

---

## 数据

**已随仓库提交**（`data/raw/`），clone 下来即可用，无需额外下载。

| 文件 | 行 × 列 | 大小（字节） | 大小（MiB） | 说明 |
|---|---:|---:|---:|---|
| `data/raw/train.csv` | 699,635 × 23 | 64,756,867 | 61.76 | 含目标列 `satisfaction` |
| `data/raw/test.csv` | 299,844 × 22 | 27,200,964 | 25.94 | 无目标列 |
| `data/raw/sample_submission.csv` | 299,844 × 2 | 4,197,832 | 4.00 | 填基准率 0.4436 |
| `data/raw/CHECKSUMS.txt` | — | — | — | 数据未被替换的凭据 |

三份数据文件合计 **96,155,663 字节 = 91.70 MiB（约 96.16 MB）**。

```powershell
# Windows：用项目自己的校验（CSV 会先做 CRLF -> LF 归一化），或直接跑 make data / verify
python -c "from src.common.config import verify_checksums; from pathlib import Path; r=verify_checksums('data/raw',[Path('data/raw/train.csv'),Path('data/raw/test.csv')]); print(r); raise SystemExit(0 if r['ok'] else 1)"

# Linux / LF 检出：可以直接用 sha256sum
cd data/raw && sha256sum -c CHECKSUMS.txt
```

`CHECKSUMS.txt` 的哈希对 `.csv/.txt` 先做 **CRLF → LF 归一化**，因此同一份 CSV 在 Windows（CRLF 工作区）与 Linux/CI（LF）得到同一个摘要；仓库同时用 `.gitattributes` 把文本统一为 LF。**仓库已统一 LF 并重检出**，当前 raw CSV 的原始字节哈希也与清单直接一致；若你的 clone 仍是 CRLF，可执行 `git add --renormalize data/raw && git checkout -- data/raw` 一次性迁移。不迁移也不影响 `make data` / `verify`，因为哈希会归一化。确认是有意更换数据时，才用 `--accept-data-change` 更新清单。

`data/processed/` 是派生数据，**不进 git**——它可以从 raw 完全重建，提交它只会让每次 `make data` 都产生脏工作区。

### 数据来源

比赛数据下载需要登录态，本仓库使用的是 Kaggle 公开数据集
`vijaikm/s6e10-downcasted-parquet-5fold`（CC BY 4.0），与官方数据**逐位等价**：
用 `base + LightGBM` 跑出 OOF AUC **0.958829**，与独立公开分析报告的 0.958829 一致到小数点后 5 位。

若课程要求"必须用官方数据"，请用组内账号重新下载：

```bash
kaggle competitions download -c playground-series-s6e10 -p data/raw
cd data/raw && unzip -o '*.zip' && cd -
python -m src.data.make_dataset --config configs/base.yaml --accept-data-change
python -m src.models.train --config configs/base.yaml --model lgbm --groups base   # 应仍为 0.958829
```

---

## 目录结构

```
├── plan.md                  # 唯一计划文件：里程碑 / 分工 / 协作规范
├── Makefile                 # 所有入口：data / split / train / abl / p1 / p3 / verify
├── LICENSE                  # MIT
├── pyproject.toml           # 包元数据 + pytest / ruff 配置
├── requirements.txt         # 运行依赖
├── requirements-dev.txt     # 运行依赖 + pytest
├── .gitattributes           # 文本统一 LF（CHECKSUMS 跨平台可校验）
├── .github/workflows/       # CI：pytest + compileall
├── configs/
│   ├── base.yaml            # 正式实验
│   └── toy.yaml             # 冒烟测试（不碰真实数据）
├── data/
│   ├── README.md            # 数据字典 + 六项检查 + 泄漏风险
│   ├── raw/                 # 已入库，只读，91.70 MiB
│   └── processed/           # 派生数据，不进 git，10 秒可重建
├── splits/
│   ├── fold_v1.json         # 冻结划分，全员唯一来源
│   └── fold_v1.fingerprint.json  # 数据 + fold 文件的指纹旁路
├── src/
│   ├── common/split.py      # 4 种划分策略 + 冻结/指纹 + 三方案对比
│   ├── data/                # 清洗、列类型探测（pandas 3 str dtype 兼容）
│   ├── features/            # 特征组 A/B/C + 折内目标编码 + FeatureBuilder
│   ├── models/              # 5 折训练骨架 / 最终预测 / 调参敏感性（P2）
│   └── eval/                # 评估 / 稳定性 / SHAP / badcase（P3）
├── scripts/
│   ├── probe_dataset.py     # 数据集体检
│   ├── eda.py               # 4 张 EDA 图
│   ├── plot_importance.py   # 特征重要性
│   ├── gen_feature_dict.py  # 生成特征字典
│   ├── verify.py            # 33 项材料验收
│   ├── clean.py             # 跨平台清理派生数据
│   └── make_toy_data.py     # 合成数据（冒烟测试用）
├── tests/                   # 119 项 pytest（数据/特征/划分/评估/预测）
├── reports/
│   ├── exp_log.csv          # 实验登记表
│   ├── p1_data_report.md    # P1 报告
│   ├── p2_model_report.md   # P2 报告
│   ├── p3_eval_report.md    # P3 报告
│   ├── figures/             # 报告用图（300 dpi）
│   ├── eval/                # P3 评估产物（metrics.json / stability.json / SHAP / badcase）
│   └── oof/                 # OOF 预测（不进 git，可重建）
└── docs/
    ├── topic-s6e10.md       # 选题冻结稿
    ├── feature-dictionary.md # 特征字典（自动生成，49 列）
    └── verify.md            # 验收指南（含破坏性测试）
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

### 实验登记（`reports/exp_log.csv`）

- `run_id` 统一格式：`{owner}_{stage}_{model}_{feat}_{MMDD}_s{seed}[_{tag}]`
  例：`P1_feat_lgbm_base_1026_s42_abl`、`P2_final_lgbm_CT_1031_s42_final`。
  **seed 必须在 id 里**，否则不同 seed 的 OOF 会同名互相覆盖。
- `train` 的 CV 行与 `predict` 的最终交付行都登记（`predict` 写 `stage=final`，并落 `reports/oof/{run_id}.parquet`，同时保留兼容路径 `P2_final_oof.parquet`）。
- `git_commit` 优先取 `GIT_COMMIT` 环境变量（CI 无 `.git` 时用），取不到再问 `git`，都取不到写 `unknown`——空字符串会被误认为“忘了填”。

---

## 任务与进度

**唯一计划文件：[`plan.md`](plan.md)** —— 里程碑 M0–M4、周计划、交接点、协作规范、答辩分工。

当前进度：

| 里程碑 | 状态 |
|---|---|
| M0 启动（五项冻结） | ✅ 完成（题目待组内口头确认） |
| M1 基线打通（三人分数一致到 1e-6） | ✅ 完成（base 0.958829，P1/P2 逐位一致，P3 已独立复评） |
| M2 做出增量 | ✅ 完成（消融 / 调参敏感性 / P3 评估与稳定性 / SHAP / badcase） |
| M3 初稿齐 | ✅ 完成（三份报告初稿齐：P1 / P2 / P3） |
| M4 交付 | 🟡 进行中（仅剩 PPT 双格式导出、打印装订与彩排） |

关键交接：

| # | 交付物 | 状态 |
|---|---|---|
| H1 | 冻结划分 `splits/fold_v1.json` | ✅ 已交付（含指纹旁路） |
| H2 | 主力模型调用接口 | ✅ 已交付（`src/models/predict.py`：`train_final()` / `predict_test()`） |
| H3 | 最终模型 + submission | ✅ 已交付（`reports/submission_final.csv`，299,844 行概率） |
| H4 | 三份章节稿 | ✅ 已交付（`reports/p1_data_report.md` / `p2_model_report.md` / `p3_eval_report.md`） |

验收材料：`python scripts/verify.py --config configs/base.yaml`（**33 项**，当前 33/33 通过、0 SKIP）+ `--reproduce`（另加 1 项重跑复现，合计 **34 项**）；另有 `python -m pytest -q`（**119 passed**）。详见 [`docs/verify.md`](docs/verify.md)。
