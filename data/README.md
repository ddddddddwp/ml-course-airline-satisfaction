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

## 2. 文件清单

| 文件 | 行数 | 列数 | 说明 |
|---|---:|---:|---|
| `raw/train.csv` | 待复核（公开值 699,635） | 待复核（23） | 含目标列 `satisfaction` |
| `raw/test.csv` | 待复核（299,844） | 待复核（22） | 无目标列 |
| `raw/sample_submission.csv` | — | 2 | `id, satisfaction`，填的是基准率 0.4436 |
| `raw/CHECKSUMS.txt` | — | — | `make data` 自动生成 |

**校验**（防止中途换数据）：

```bash
cd data/raw && sha256sum -c CHECKSUMS.txt
```

`CHECKSUMS.txt` 由 `make data` 每次覆写，来源始终是当时的 raw 文件。

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

### 3.3 序数评分列（14 个，0–5）

`Inflight wifi service`、`Departure/Arrival time convenient`、`Ease of Online booking`、
`Gate location`、`Food and drink`、`Online boarding`、`Seat comfort`、
`Inflight entertainment`、`On-board service`、`Leg room service`、`Baggage handling`、
`Checkin service`、`Inflight service`、`Cleanliness`

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

## 4. 六项检查（P1-03 验收）

| 检查项 | 结论 | 处置 |
|---|---|---|
| 行数 / 列数 | 待复核 | `python scripts/probe_dataset.py data/raw/train.csv --test data/raw/test.csv` |
| 目标分布 | 正例 44.4%，均衡 | 不用重采样；不用 accuracy，用 AUC |
| 缺失 | **仅** `Arrival Delay in Minutes`，train 292 / test 130（<0.05%） | 加 `arr_delay_missing` 标记 + 用出发延误行内填补 |
| 重复 | 0 行（忽略 id） | 无需去重（pipeline 仍保留去重步骤做兜底） |
| 异常 | 延误长尾至约 490 min；`Journey` 类极端值少 | `log1p` + `is_delayed`；不做截断，树模型不敏感 |
| 时间 / 分组 | **无时间列、无实体列** | 用 StratifiedKFold 主线；用 Flight Distance 分箱做 GroupKFold 鲁棒性检验 |

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
