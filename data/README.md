# 数据说明（P1-02 交付物）

> 本文件由 P1 维护。任何数字都必须能用 `python -m src.data.make_dataset --config configs/base.yaml` 复现。
> 当前状态：**已在真实数据上实测**；数据、清洗、schema 与校验和均以本仓库为准。

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

| 文件 | 行数 | 列数 | 大小（字节） | 大小（MiB） | 说明 |
|---|---:|---:|---:|---:|---|
| `raw/train.csv` | 699,635 | 23 | 64,756,867 | 61.76 | 含目标列 `satisfaction` |
| `raw/test.csv` | 299,844 | 22 | 27,200,964 | 25.94 | 无目标列 |
| `raw/sample_submission.csv` | 299,844 | 2 | 4,197,832 | 4.00 | `id, satisfaction`，填基准率 0.4436 |
| `raw/CHECKSUMS.txt` | — | — | — | — | `make data` 的校验和闸门 |

三份数据文件合计 **96,155,663 字节 = 91.70 MiB**（约 96.16 MB）。仓库已用 `.gitattributes` 将文本统一为 LF，并重新检出 raw CSV，因此当前原始字节哈希与 `CHECKSUMS.txt` 直接一致。

### 2.1 数据来源与还原度（重要）

比赛数据下载需要登录态，本次是从 Kaggle **公开数据集**取的同一份数据：

| 资源 | 来源 | 许可 | 用途 |
|---|---|---|---|
| 主数据 | `vijaikm/s6e10-downcasted-parquet-5fold` | CC BY 4.0 | train/test 全量，转回 CSV |
| 原始真实数据 | `arseniyshutko/binary-aviation-satisfaction-129k`（129,880 行） | CC0 | 放 `data/external/`，用于 P1-20 一致性核对 |

**还原度校验**：用这份数据跑 `base + LGBM` 5 折，OOF AUC = **0.958829**，
与公开独立分析（thomasjpfan）报告的 base 0.958829 **完全一致到小数点后 5 位**。
说明行序、类型、缺失处理与官方数据等价。

> ⚠️ 两个注意事项：
> 1. `CHECKSUMS.txt` 校验的是**本地这份导入后的文件**，不是 Kaggle 官方原文件。
>    它仍能发现"中途被替换"，但不能证明与官网字节一致。
> 2. 若课程要求"必须用比赛官方数据"，请用组内 Kaggle 账号重新下载后再跑一遍
>    `python -m src.data.make_dataset --config configs/base.yaml --accept-data-change`，
>    `data/README.md` 与 `exp_log` 里的数字需重算（预计一致）。

**校验（防止中途换数据）**：

```powershell
# Windows：用项目自己的哈希（CSV 按 CRLF -> LF 归一化），或直接跑 make data / verify
$env:PYTHONPATH="."
python -c "from src.common.config import verify_checksums; from pathlib import Path; r=verify_checksums('data/raw',[Path('data/raw/train.csv'),Path('data/raw/test.csv')]); print(r); raise SystemExit(0 if r['ok'] else 1)"

# Linux / LF 检出：可以直接用 sha256sum
cd data/raw && sha256sum -c CHECKSUMS.txt
```

`CHECKSUMS.txt` 的哈希对 `.csv/.txt` 先做 **CRLF → LF 归一化**，所以同一份 CSV 在
Windows（CRLF 工作区）与 Linux/CI（LF）得到同一个摘要。仓库已统一 LF 并重检出；
若你的 clone 仍是 CRLF，可执行 `git add --renormalize data/raw && git checkout -- data/raw`
一次性迁移；不迁移也不影响 `make data` / `verify`，因为哈希会归一化。
确认是**有意更换数据**时，才用 `--accept-data-change` 重新登记校验和；否则应恢复原始文件。

---

## 3. 数据字典

### 3.1 标识与目标

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | int | 行标识。train/test 连续编号，各自唯一，**建模前删除** |
| `satisfaction` | bool→int8 | 目标。正例 **310,339 / 699,635 = 44.36%** |

### 3.2 类别列（4 个，低基数）

| 列 | 取值 | 备注 |
|---|---|---|
| `Gender` | Female / Male | 近乎无信息（满意率 43.8% vs 44.9%） |
| `Customer Type` | Loyal / disloyal | Loyal 占 82.5%，满意率 49.5% vs 20.1% |
| `Type of Travel` | Business / Personal | **强信号**，个人出行满意率仅 9.7% |
| `Class` | Business / Eco / Eco Plus | **最强单列**，Business 满意率 72.5% |

> **pandas 3 兼容修正**：pandas 3 里字符串列的 dtype 是 `str`（不再固定是 `object`）。
> `detect_schema` 改用 `is_string_dtype` / `is_object_dtype` / `CategoricalDtype` 三路判断后，
> 能稳定识别 **4 个类别列**（修复前 `cat_cols` 为空，LightGBM 直接报
> `pandas dtypes must be int,float or bool: Gender: str`）。
> `prepare_X` 再把 str/object 统一转 `category`；schema 在清洗前探测，落盘 `reports/schema_s6e10.json`。

### 3.3 序数评分列（13 个，0–5）

`Inflight wifi service`、`Departure/Arrival time convenient`、`Ease of Online booking`、
`Gate location`、`Food and drink`、`Online boarding`、`Seat comfort`、
`Inflight entertainment`、`On-board service`、`Leg room service`、`Baggage handling`、
`Checkin service`、`Cleanliness`

> 注意：原始真实数据集有 14 个评分列（多一个 `Inflight service`），
> 本合成版本已去掉该列。所有代码按 `schema.json` 自动探测，不写死列名。

> **0 分表示"不适用"，不是"最差"。** 两条证据分开写、不要混用：
> - `Inflight wifi service`：评 0 的满意率约 **89%**（88.73%），评 1 只有 **39%**（39.07%）；
> - `Online boarding`：评 0 的满意率约 **62%**（62.04%），评 1 只有约 **10.5%**（10.53%）。
>
> 两列都呈 U 型而非单调。这是本项目最重要的一条数据事实，直接决定特征怎么造
> （A 组把 0→NaN 并单独加 `is_zero`，B 组隐含 A 才有业务含义）。

### 3.4 数值列

| 列 | 说明 |
|---|---|
| `Age` | 非单调（19–25 岁 30%，41–60 岁 60%，>60 岁 14%） |
| `Flight Distance` | 高基数（3,474 个取值），可作航线代理，做频次/目标编码的唯一落点 |
| `Departure Delay in Minutes` | 约 90% 为 0，右偏，最大约 490 |
| `Arrival Delay in Minutes` | 与上一列 r≈0.84，87% 行完全相等；**飞行后才可知** |

---

## 4. 六项检查（P1-03 验收，已实测）

| 检查项 | 结论 | 处置 |
|---|---|---|
| 行数 / 列数 | train **699,635 × 23**，test **299,844 × 22** | 规模够撑三份报告 |
| 目标分布 | 正例 **310,339 / 699,635 = 44.36%** | 均衡，不用重采样；主指标 AUC，不看 accuracy |
| 缺失 | **仅** `Arrival Delay in Minutes`：train **292** / test **130**（0.04%） | `arr_delay_missing` 标记 + 用出发延误行内填补 |
| 重复 | **0 行**（按除 `id` 外的全列判重；`id` 各自唯一） | 无需去重（pipeline 仍保留兜底） |
| 异常 | 延误长尾，约 90% 为 0；`id` 唯一率 100% | `log1p` + `is_delayed`；`id` 建模前删除 |
| 时间 / 分组 | **无时间列、无实体列**；train/test 的 id 无重叠 | StratifiedKFold 主线；Flight Distance 分箱做 GroupKFold 鲁棒性检验 |

> **判重口径修正**：旧实现 `drop_duplicates()` 把每行唯一的 `id` 也算进比较键，等于永远删不掉重复行；
> 现在按**除 `id` 外的全部列**判重，`id` 是否重复单独告警。本数据上重复 0 行，但口径必须写对。

**基线结果**（`make train`，22 特征，seed 42，5 折）：

| 指标 | 值 |
|---|---|
| OOF AUC | **0.958829** |
| 折间 std | 0.000648 |
| 单折耗时 | 约 6 秒 |
| 全流程耗时 | 约 32 秒 |

> 最终主力模型为 `base+C+T`（25 特征），OOF AUC **0.960191**，run_id
> `P2_final_lgbm_CT_1031_s42_final`；P3 的 pooled AUC 为 0.960186（5 折 OOF 拼接后计算）。

---

## 5. 清洗规则（`src/data/make_dataset.py` 做了什么）

0. **校验和闸门（先于任何读取/清洗）**：`CHECKSUMS.txt` 不存在则首次生成；存在且一致则继续；
   不一致直接报错退出，只有显式 `--accept-data-change` 才用当前文件更新清单。
   这样"数据被换过"不会被静默洗进 processed。
1. **schema 探测（清洗前）**：在 raw train 上跑 `detect_schema`，得到 13 个服务评分列、
   **4 个类别列**、4 个数值列，并落盘 `reports/schema_s6e10.json`；后续 step 只按这份 schema 操作。
2. **去重**：按**除 `id` 外**的全部列检测重复（当前 0 行），`id` 重复单独告警。
3. **类别列 strip**：去首尾空格，防止 `"Business "` 与 `"Business"` 被当成两类。
4. **目标二值化**：兼容 bool / 0-1 / `"satisfied"` 字符串三种写法。
5. **到达延误**：`arr_delay_missing` 标记 + 用 `Departure Delay` **行内**填补（不跨样本，无泄漏）。

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
| 跨折重复 | 不存在 | 忽略 id 后 0 重复；train/test 的 id 无重叠 |
| 目标编码泄漏 | **高** | 只在训练折 fit；单独成 T 组，单独消融验证 |
| **事后特征** | **高** | `Arrival Delay` 飞行后才知道。业务口径若为"飞行前预测"，必须剔除并报告掉多少分 |
| 划分漂移 / 数据被换 | 中 | `splits/fold_v1.fingerprint.json` 记录 processed 数据 sha256 + fold 文件 sha256；`verify` 校验；`freeze` 已存在默认跳过、`--force` 先备份 |

> 冻结指纹 sidecar **不改 `fold_of_row`**：`splits/fold_v1.json` 仍是唯一划分来源，
> `fold_v1.fingerprint.json` 只回答"这份折是不是还绑在当前 processed 数据上"。
> `python scripts/verify.py` 会同时校验数据指纹与 fold 文件 sha256；两者任一变化都会 FAIL。

**事后特征实验的结论**：剔除 `Arrival Delay` + `arr_delay_missing` 后，`base+C+T` 的 OOF AUC
从 0.960191 降到 0.960157（**−0.000034**），几乎无损失。因此最终模型可以只使用飞行前即可获得的信息，
业务上可部署。目标编码的折内拟合纪律不变：T 组只在训练折 `fit`，验证折与 test 只 `transform`。
