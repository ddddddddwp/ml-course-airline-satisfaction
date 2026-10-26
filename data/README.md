# 数据说明（P1-02 交付物）

> 本文件由 P1 维护。任何数字都必须能用 `make data` 复现。
> 当前状态：**结构已按公开分析预填，标 `待复核` 的项需在真实数据到位后跑一遍确认。**

---

## 1. 数据集

| 项 | 值 |
|---|---|
| 名称 | Kaggle Playground Series S6E10 — Predicting Airline Satisfaction |
| 来源 | https://www.kaggle.com/competitions/playground-series-s6e10/data |
| 性质 | 合成数据，源自真实调查集 *Airline Passenger Satisfaction*（129,880 行，CC0） |
| 任务 | 二分类，预测 `satisfaction` 概率 |
| 指标 | ROC AUC |
| 许可 | 见比赛 Rules；课程项目内使用 |

---

## 2. 文件清单（已本机实测）

| 文件 | 行数 | 列数 | 说明 |
|---|---:|---:|---|
| `raw/train.csv` | 699,635 | 23 | 含目标列 `satisfaction` |
| `raw/test.csv` | 299,844 | 22 | 无目标列 |
| `raw/sample_submission.csv` | 299,844 | 2 | `id, satisfaction`，填基准率 0.4436 |
| `raw/CHECKSUMS.txt` | — | — | `make data` 自动生成 |

### 2.1 数据来源与还原度（重要）

比赛数据下载需要登录态，本次是从 Kaggle **公开数据集**取的同一份数据：

| 资源 | 来源 | 许可 | 用途 |
|---|---|---|---|
| 主数据 | `vijaikm/s6e10-downcasted-parquet-5fold` | CC BY 4.0 | train/test 全量，转回 CSV |
| 原始真实数据 | `arseniyshutko/binary-aviation-satisfaction-129k`（129,880 行） | CC0 | 放 `data/external/`，用于 P1-20 一致性核对 |

**还原度校验**：用这份数据跑 `base + LGBM` 5 折，OOF AUC = **0.95883**，
与公开独立分析（thomasjpfan）报告的 base 0.95883 **完全一致到小数点后 5 位**。
说明行序、类型、缺失处理与官方数据等价。

> ⚠️ 两个注意事项：
> 1. `CHECKSUMS.txt` 校验的是**本地这份导入后的文件**，不是 Kaggle 官方原文件。
>    它仍能发现"中途被替换"，但不能证明与官网字节一致。
> 2. 若课程要求"必须用比赛官方数据"，请用组内 Kaggle 账号重新下载后再跑一遍
>    `make data`，`data/README.md` 与 `exp_log` 里的数字需重算（预计一致）。

**校验**（防止中途换数据）：

```bash
cd data/raw && sha256sum -c CHECKSUMS.txt
```

---

## 3. 数据字典

### 3.1 标识与目标

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | int | 行标识。train/test 连续编号。**建模前删除** |
| `satisfaction` | bool→int8 | 目标。正例率约 44.4%（待复核） |

### 3.2 类别列（4 个，低基数）

| 列 | 取值 | 备注 |
|---|---|---|
| `Gender` | Female / Male | 近乎无信息（满意率 43.8% vs 44.9%） |
| `Customer Type` | Loyal / disloyal | Loyal 占 82.5%，满意率 49.5% vs 20.1% |
| `Type of Travel` | Business / Personal | **强信号**，个人出行满意率仅 9.7% |
| `Class` | Business / Eco / Eco Plus | **最强单列**，Business 满意率 72.5% |

### 3.3 序数评分列（13 个，0–5）

`Inflight wifi service`、`Departure/Arrival time convenient`、`Ease of Online booking`、
`Gate location`、`Food and drink`、`Online boarding`、`Seat comfort`、
`Inflight entertainment`、`On-board service`、`Leg room service`、`Baggage handling`、
`Checkin service`、`Cleanliness`

> 注意：原始真实数据集有 14 个评分列（多一个 `Inflight service`），
> 本合成版本已去掉该列。所有代码按 `schema.json` 自动探测，不写死列名。

> **0 分表示"不适用"，不是"最差"。** WiFi 评 0 的满意率 89%，评 1 只有 39%（U 型）。
> 这是本项目最重要的一条数据事实，直接决定特征怎么造。

### 3.4 数值列

| 列 | 说明 |
|---|---|
| `Age` | 非单调（19–25 岁 30%，41–60 岁 60%，>60 岁 14%） |
| `Flight Distance` | 高基数，可作航线代理，做频次/目标编码的唯一落点 |
| `Departure Delay in Minutes` | 约 90% 为 0，右偏，最大约 490 |
| `Arrival Delay in Minutes` | 与上一列 r≈0.84，87% 行完全相等；**飞行后才可知** |

---

## 4. 六项检查（P1-03 验收，已实测）

| 检查项 | 结论 | 处置 |
|---|---|---|
| 行数 / 列数 | train **699,635 × 23**，test **299,844 × 22** | 规模够撑三份报告 |
| 目标分布 | 正例 **310,339 / 699,635 = 44.36%** | 均衡，不用重采样；主指标 AUC，不看 accuracy |
| 缺失 | **仅** `Arrival Delay in Minutes`：train **292** / test **130**（0.04%） | `arr_delay_missing` 标记 + 用出发延误行内填补 |
| 重复 | **0 行**（忽略 id） | 无需去重（pipeline 仍保留兜底） |
| 异常 | 延误长尾，约 90% 为 0；`id` 唯一率 100% | `log1p` + `is_delayed`；`id` 建模前删除 |
| 时间 / 分组 | **无时间列、无实体列**，train/test 无漂移 | StratifiedKFold 主线；Flight Distance 分箱做 GroupKFold 鲁棒性检验 |

**基线结果**（`make train`，22 特征，seed 42，5 折）：

| 指标 | 值 |
|---|---|
| OOF AUC | **0.95883** |
| 折间 std | 0.00065 |
| 单折耗时 | 约 6 秒 |
| 全流程耗时 | 32 秒 |

---

## 5. 清洗规则（`src/data/make_dataset.py` 做了什么）

1. **去重**：训练集内部完全重复行（当前数据为 0 行，保留兜底）
2. **类别列 strip**：去首尾空格，防止 `"Business "` 与 `"Business"` 被当成两类
3. **目标二值化**：兼容 bool / 0-1 / `"satisfied"` 字符串三种写法
4. **到达延误**：`arr_delay_missing` 标记 + 用 `Departure Delay` **行内**填补（不跨样本，无泄漏）

**不在这里做的事**（避免把建模选择烧进数据）：

- 0 分 → NaN 的语义修正 → 放在特征组 A，才能作为一条消融轴
- 目标编码 → 放在训练折内，见 `src/features/encode.py`
- 任何标准化 / 分箱 → 放在特征或模型 pipeline

---

## 6. 已知泄漏风险

| 风险 | 状态 | 处理 |
|---|---|---|
| ID 类泄漏 | 低 | `id` 建模前删除 |
| 时间泄漏 | 不存在 | 无时间列 |
| 跨折重复 | 不存在 | 无重复行、train/test 无重叠（待复核） |
| 目标编码泄漏 | **高** | 只在训练折 fit；单独成 T 组，单独消融验证 |
| **事后特征** | **高** | `Arrival Delay` 飞行后才知道。业务口径若为"飞行前预测"，必须剔除并报告掉多少分 |
