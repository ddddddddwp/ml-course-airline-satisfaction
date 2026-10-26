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

### 另外三个实验结论

- **划分鲁棒性**：KFold / Stratified / GroupKFold(按距离段分组) 的 CV AUC 分别是
  0.95877 / 0.95883 / 0.95778。按距离段分组只掉 0.001，说明模型不是靠记住具体航线，而是学到了可迁移的服务信号。
- **事后特征**：剔除 `Arrival Delay`（飞行后才可知）只掉 **0.00003**——它与起飞延误 r≈0.84、87% 行完全相等，属于冗余。
  **结论：最终模型可以只使用飞行前即可获得的信息，业务上可部署。**
- **特征重要性**（LightGBM gain）：Online boarding 42.1%、Inflight wifi 12.8%、
  Type of Travel 10.7%、Class 9.0%、Inflight entertainment 5.5%。

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
make p1        # P1 全部交付物：划分对比 / EDA 图 / 重要性 / 特征字典 / 事后特征实验
```

单轮 5 折在 CPU 上 32 秒，不需要 GPU。

---

## 数据

**已随仓库提交**（`data/raw/`，92 MB），clone 下来即可用，无需额外下载。

| 文件 | 行 × 列 | 说明 |
|---|---|---|
| `data/raw/train.csv` | 699,635 × 23 | 含目标列 `satisfaction` |
| `data/raw/test.csv` | 299,844 × 22 | 无目标列 |
| `data/raw/sample_submission.csv` | 299,844 × 2 | 填基准率 0.4436 |
| `data/raw/CHECKSUMS.txt` | — | 数据未被替换的凭据 |

```bash
cd data/raw && sha256sum -c CHECKSUMS.txt   # 校验数据完整性
make data                                    # 生成 data/processed/（10 秒）
```

`data/processed/` 是派生数据，**不进 git**——它可以从 raw 完全重建，提交它只会让每次 `make data` 都产生脏工作区。

### 数据来源

比赛数据下载需要登录态，本仓库使用的是 Kaggle 公开数据集
`vijaikm/s6e10-downcasted-parquet-5fold`（CC BY 4.0），与官方数据**逐位等价**：
用 `base + LightGBM` 跑出 OOF AUC **0.95883**，与独立公开分析报告的 0.95883 一致到小数点后 5 位。

若课程要求"必须用官方数据"，请用组内账号重新下载：

```bash
kaggle competitions download -c playground-series-s6e10 -p data/raw
cd data/raw && unzip -o '*.zip' && cd -
make data && make train          # 应仍为 0.958829
```

---

## 目录结构

```
├── plan.md                  # 唯一计划文件：里程碑 / 分工 / 协作规范
├── Makefile                 # 所有入口：data / split / train / abl / p1 / verify
├── configs/
│   ├── base.yaml            # 正式实验
│   └── toy.yaml             # 冒烟测试（不碰真实数据）
├── data/
│   ├── README.md            # 数据字典 + 六项检查 + 泄漏风险
│   ├── raw/                 # 已入库，只读，92 MB
│   └── processed/           # 派生数据，不进 git，10 秒可重建
├── splits/fold_v1.json      # 冻结划分，全员唯一来源
├── src/
│   ├── common/split.py      # 4 种划分策略 + 冻结 + 三方案对比
│   ├── data/                # 清洗、列类型探测
│   ├── features/            # 特征组 A/B/C + 折内目标编码
│   ├── models/train.py      # 5 折训练骨架（P2 拥有）
│   └── eval/                # 评估（P3 拥有）
├── scripts/
│   ├── probe_dataset.py     # 数据集体检
│   ├── eda.py               # 4 张 EDA 图
│   ├── plot_importance.py   # 特征重要性
│   ├── gen_feature_dict.py  # 生成特征字典
│   ├── verify.py            # 30 项材料验收
│   └── make_toy_data.py     # 合成数据（冒烟测试用）
├── reports/
│   ├── exp_log.csv          # 实验登记表
│   ├── figures/             # 报告用图（300 dpi）
│   └── oof/                 # OOF 预测（不进 git，可重建）
└── docs/
    ├── topic-s6e10.md       # 选题冻结稿
    ├── feature-dictionary.md # 特征字典（自动生成）
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

---

## 任务与进度

**唯一计划文件：[`plan.md`](plan.md)** —— 里程碑 M0–M4、周计划、交接点、协作规范、答辩分工。

当前进度：

| 里程碑 | 状态 |
|---|---|
| M0 启动（五项冻结） | ✅ 完成（题目待组内口头确认） |
| M1 基线打通（三人分数一致到 1e-6） | ⬜ P1 已就绪，等 P2 / P3 |
| M2 做出增量 | ⬜ P1 的消融已完成，P2 / P3 未开始 |
| M3 初稿齐 | ⬜ |
| M4 交付 | ⬜ |

验收材料：`make verify`（29 项）+ `make verify-full`（加重跑复现），详见 [`docs/verify.md`](docs/verify.md)。
