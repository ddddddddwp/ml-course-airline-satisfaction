# P1 工作计划与闭环说明

> 负责人：P1（数据与特征）
> 目标：**先让全组的实验能跑起来，再让自己有东西可写。**
> 骨架已就绪，`make smoke` 可在无 Kaggle 数据的情况下验证全链路。

---

## 1. 闭环长什么样

```
data/raw/{train,test}.csv
        │  make data            ← 去重 / strip / 目标二值化 / 到达延误填补 / checksum
        ▼
data/processed/{train,test}.parquet
        │  make split           ← 冻结 fold_of_row（只此一份，全员共用）
        ▼
splits/fold_v1.json ──────────► P2 训练、P3 评估都从这里取折
        │
        │  make train / make abl
        ▼
特征组 A/B/C/T ──► LightGBM / 逻辑回归 ──► 5 折 OOF
        │
        ├─► reports/exp_log.csv        每个数字可追溯
        ├─► reports/oof/<run_id>.parquet  P3 的错误分析输入
        └─► reports/split_compare.csv    划分论证的证据
```

**关键：这条链上没有一步依赖 Kaggle 账号、GPU 或大数据。** 700k 行 5 折约 1–2 分钟。

---

## 2. 命令速查

| 场景 | 命令 | 耗时 |
|---|---|---|
| 真实数据全流程 | `make data && make split && make train` | ~3–5 min |
| 无数据先验链路 | `make smoke` | ~40 s |
| 划分三方案对比（P1-05） | `make split-compare` | ~2 min |
| 消融矩阵（P1-12） | `make abl` | ~15 min |
| 数据集可行性体检 | `python scripts/probe_dataset.py data/raw/train.csv --test data/raw/test.csv --time` | ~1 min |
| 清空重建 | `make clean && make data && make split` | — |

特征组开关（`--groups`）：

| 组合 | 含义 |
|---|---|
| `base` | 原始 21 特征 |
| `base,A` | + 语义修正（0=不适用，加 is_zero，0→NaN） |
| `base,B` | + 聚合/交互（隐含 A） |
| `base,C` | + 频次编码（Flight Distance、Age） |
| `base,C,T` | + 目标编码（**折内 fit**） |
| `base,A,B,C,T` | 全量 |

---

## 3. 任务清单

### 3.1 交付物（会挡住别人，必须按时）

| 编号 | 任务 | 交付物 | 卡住谁 | 截止 |
|---|---|---|---|---|
| P1-01 | 数据可用性验证 | `data/README.md` 第 4 节 | 全组 | D1 |
| **P1-05 ★** | 冻结划分 | `splits/fold_v1.json` | **P2-04、P3-03** | W1-D3 |
| P1-06 | 清洗 pipeline | `src/data/make_dataset.py` 可重建 | P2 训练 | W1-D4 |
| P1-07 | 特征 v0 | 4 组特征可开关 | P2 主力模型 | W1-D5 |
| exp_log | 实验登记表 | `reports/exp_log.csv` | 报告每个数字 | 常驻 |

### 3.2 自己写报告的部分（不挡别人，可顺延）

| 编号 | 任务 | 产出 | 进报告哪节 |
|---|---|---|---|
| P1-03 | EDA（缺失/分布/重复/异常 + 0 语义） | 4 张图 + 量化比例 | 数据描述 |
| P1-04 | 泄漏排查（事后特征 + 目标编码纪律） | 风险清单 | 数据描述 |
| P1-05 | 三方案对比表 | `reports/split_compare.csv` | 划分论证 |
| P1-08/09 | 特征组 A/B | 消融行 + 异质性结论 | 特征设计 |
| P1-10 | 特征组 C/T | 消融行 + 折内拟合证明 | 特征设计 |
| P1-11 | 重要性 top-20 | 1 张图 | 特征设计 |
| P1-12 | 消融 2 模型族 × 5 组 | 消融表 | 实验结果 |
| P1-20 | 合成 vs 真实数据一致性核对 | 1 张分布对比图 | 数据描述（加分） |
| P1-21 | 0=不适用 语义分析 | 1 张满意率曲线 | 特征设计（最亮） |

### 3.3 可以砍的

`P1-13`（特征选择重跑）、`P1-19`（采样/增强）——本数据集类别均衡、特征少，这两项收益接近 0。

---

## 4. 三道门槛（现在都能过）

| 门槛 | 现状 | 说明 |
|---|---|---|
| 要 GPU 吗 | **不要** | LightGBM CPU；700k × 21 特征 5 折 1–2 min |
| 数据大吗 | **不大** | 102 MB，raw 不进 git，processed parquet 约 60 MB |
| 要 Kaggle 账号吗 | 提交要；**开发不要** | `make smoke` 用同结构合成数据跑通全链 |
| 依赖装了吗 | **已装** | sklearn 1.7.2 / lightgbm 4.7.0 / pyarrow / pyyaml |

---

## 5. 交接契约（说一遍就要算数）

1. `splits/fold_v1.json` **冻结后只读**。谁改，谁的历史实验作废。
2. 所有实验必须 `load_folds()` 取折，**禁止自己 `train_test_split`**。
3. 目标编码只在训练折 `fit`，验证折只 `transform`。
4. 提交的是**概率**，不是 0/1 标签（30 万行 `id,satisfaction`）。
5. `reports/exp_log.csv` 里每个数字都要能按 `run_id` 查到、按 `git_commit` 复现。

**交叉校验**：P2 与 P3 各自独立跑一次 `make train`，分数必须一致到 `1e-6`。
不一致基本就是索引顺序不同——查有没有人重建了 fold 或排序了数据。

---

## 6. 建议顺序（D1–D5）

| 天 | 做什么 | 完成标志 |
|---|---|---|
| D1 | 拿数据 → `probe_dataset.py` → 填 `data/README.md` 第 4 节 → 发选题一页纸 | 全组点头 |
| D2 | `make data` → EDA 四张图 → 0=不适用 曲线 | 数据可用结论 |
| D3 | `make split-compare` → 定策略 → `make split` → **H1 交接** | P2/P3 分数一致 |
| D4 | 清洗定稿 + 特征 A/B 跑通 | `make clean && make data` 可重建 |
| D5 | 特征 C/T + 重要性图 | 特征 v0 交给 P2 跑主力模型 |
