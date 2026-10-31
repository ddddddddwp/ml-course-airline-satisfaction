# 材料验收指南

> 原则：**不信结论，看证据。** 能机械验的交脚本，验不了的给检查方法和合格线。

---

## 0. 一条命令

```bash
make verify        # 33 项结构校验，约 15 秒
make verify-full   # 33 + 1 项“重跑 base 并与记录对比” = 34 项，约 50 秒
```

Windows 通常没有 `make`，等价命令：

```powershell
$env:PYTHONPATH="."
python scripts/verify.py --config configs/base.yaml              # 33 项
python scripts/verify.py --config configs/base.yaml --reproduce  # 34 项
```

当前结果（2026-10-31）：

```
结论：33/33 通过，0 项跳过
（--reproduce 另加 1 项；历史实测：重跑 base 0.958829 vs 记录 0.958829，差 8.01e-08）
```

退出码非 0 就是有 FAIL，脚本会把未通过项单独列出来。`SKIP` 不算通过也不算失败，
通常是 toy 配置没有 `contract:`，或某项 P3 产物尚未生成。

---

## 1. 验收分三层

| 层 | 验什么 | 谁做 | 耗时 |
|---|---|---|---|
| **L1 机械校验** | 形状、缺失、划分完整性、字典覆盖、登记表一致、产物存在、**CHECKSUMS 实算、fold 指纹、schema 一致、P3 metrics 键** | `make verify` | 15 s |
| **L2 人工判断** | 代码是否真折内、图与文字是否一致、结论有没有越界、P3 指标与图是否同源 | 你 | 10 min |
| **L3 独立复现** | 换一个人跑同一份数据，分数是否一致到 1e-6 | P2 / P3 | 5 min |

L1 的 33 项里，本轮新增/强化的四项要特别知道：

1. **CHECKSUMS 真正重算哈希**（不是只看文件在不在）：CSV 按 CRLF → LF 归一化后比对；
2. **冻结划分指纹**：`splits/fold_v1.fingerprint.json` 校验 processed 数据 sha256 + fold 文件 sha256；
3. **保存的 schema 与 `detect_schema` 一致**：防止报告引用过期 schema；
4. **P3 `metrics.json` 含约定键**：`auc/roc_auc` 任一 + `n_rows/n/n_oof` 任一。

手工复算单个文件时：

```powershell
# 仓库已 LF 检出时，原始字节哈希与 CHECKSUMS.txt 直接一致
Get-FileHash data/raw/train.csv -Algorithm SHA256
```

> CRLF 工作区的 `Get-FileHash` 会与清单不同，这不是数据被换；请用 `scripts/verify.py`
> （内部按 CRLF → LF 归一化）或 `make data` / `make verify`。

---

## 2. L2：人工必查的 7 项

### ① 目标编码是不是真的折内 ⭐ 最关键

看 `src/models/train.py` 的 5 折循环（约 196–200 行）：

```python
for f, (tr, va) in enumerate(folds):
    Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
    if "T" in set(groups) and "Flight Distance" in Xtr.columns:
        te = TargetEncoder("Flight Distance", smooth=20.0).fit(Xtr, ytr)   # ← 必须是 Xtr
        Xtr["Flight Distance__te"] = te.transform(Xtr)
        Xva["Flight Distance__te"] = te.transform(Xva)
```

**合格**：`fit()` 的入参是 `Xtr`（训练折）。
**不合格**：写成 `fit(X, y)`（全量）——CV 会虚高，且**看不出来**（见第 4 节实测）。

### ② 消融是不是“同 fold、同模型、只换特征”

```powershell
python -c "import pandas as pd; d=pd.read_csv('reports/exp_log.csv'); print(d[d.run_id.str.contains('abl')][['run_id','model','split_ver','params_hash','cv_mean']].to_string(index=False))"
```

**合格**：同模型的行 `split_ver` 全是 `v1`，`params_hash` 相同（只差特征数时不同）。
**不合格**：有人中途重建了 fold，或换了学习率——那消融表就不能横着比。
注意 B 组已按修正后的 `dist_bin` 重跑，最新行是 `*_abl2`；与历史 B 行比较时不要混用。

### ③ 图和文字对不对得上

打开 `reports/figures/fig4_rating_vs_target.png`，分别找两列、不要混用：

- `Inflight wifi service`：**0 分处约 89%，1 分处约 39%**；
- `Online boarding`：**0 分处约 62%，1 分处约 10.5%**。

对不上就说明文字和图有一边是错的。

### ④ 结论有没有超出证据

| 结论 | 可以写 | 不能写 |
|---|---|---|
| A/B 组负增益 | "在本数据集、LightGBM 上，A/B 无增益或负增益" | "特征工程没有用" |
| C+T +0.0014 | "单 seed、固定划分下 +0.001362；3 seeds 方向一致（均值 0.960184）" | "统计显著提升"（3 seeds 只是方差估计） |
| 0=不适用 | "0 分的满意率高于 1 分，说明 0 是'不适用'" | "0 分代表服务差" |
| nl127=0.960254 | "单 seed 单点，比主模型高 0.000063，属噪声量级" | "当前最好"（对外口径是 0.960191） |

### ⑤ 数据来源能不能交代

`data/README.md` 第 2.1 节写了镜像来源与还原度证据。若课程要求官方数据，这里需要更新。

### ⑥ 报告里的数字能不能对上 run_id

抽查 3 个数字（比如 base、C、C+T），在 `reports/exp_log.csv` 里按 `run_id` 找到对应行，
`cv_mean` 必须完全一致。

注意口径：`cv_mean` 是 **5 折 AUC 的算术平均**；`reports/eval/metrics.json` 的 `auc`
是把 5 折 OOF 预测拼起来算的 **pooled AUC**。同一模型两者会差几个 1e-6（0.960191 vs 0.960186），
不是两个模型，也不是舍入错误。

### ⑦ P3 指标 / 图与 `metrics.json` 是否一致

```powershell
python -c "import json; m=json.load(open('reports/eval/metrics.json',encoding='utf-8')); print({k:m[k] for k in ['auc','ece','n','tn','fp','fn','tp']})"
```

**合格**：`p3_eval_report.md` 的主结果表、`p3_calibration.png`、`p3_confusion.png` 与上面的
`auc=0.960186`、`ece=0.002479`、`n=699635`、`tn/fp/fn/tp=367246/22050/29440/280899` 一致；
`p3_seed_variance.png` 与 `stability.json` 的 3 个 seed 一致。

---

## 3. L3：独立复现（M1 验收）

让 P2、P3 各自在自己的目录跑：

```powershell
$env:PYTHONPATH="."
python -m src.models.train --config configs/base.yaml --model lgbm --groups base
```

期望 OOF AUC = **0.958829**（`exp_log` 里 `P1_feat_lgbm_base_1026` / `P2_feat_lgbm_base_1029`）。
三个人的 `OOF AUC` 必须一致到 **1e-6**。不一致，基本就是：

- 有人重建了 fold（没读 `fold_v1.json`）
- 有人对数据做了排序/采样
- 有人改了 `configs/base.yaml`

更省事的验收是直接跑 `python scripts/verify.py --config configs/base.yaml --reproduce`，
脚本会重跑 base 并把差值与 `contract.base_auc` 对比（阈值 1e-6）。

---

## 4. 破坏性测试：验证“验收本身”有效

只会 PASS 的检查等于没有检查。下面三个测试会让你确认脚本真的在算。
命令按 **Windows PowerShell** 写；bash 用户把 `Copy-Item` / `Move-Item` 换成 `cp` / `mv`，
把 here-string 换成 `python - <<'EOF' ... EOF` 即可。全部使用仓库内的 `.tmp/` 目录。

先准备备份目录：

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
```

### 测试 1：把划分打乱 → 校验必须报 FAIL

```powershell
Copy-Item splits/fold_v1.json .tmp/fold_backup.json -Force
@'
import json, numpy as np
p = "splits/fold_v1.json"
d = json.load(open(p, encoding="utf-8"))
d["fold_of_row"] = np.random.default_rng(0).integers(0, d["n_splits"], len(d["fold_of_row"])).tolist()
json.dump(d, open(p, "w", encoding="utf-8"))
'@ | python -
python scripts/verify.py --config configs/base.yaml   # 期望 FAIL：各折正例率 / fold 指纹至少一项
Copy-Item .tmp/fold_backup.json splits/fold_v1.json -Force   # 还原
```

> 有 `fold_v1.fingerprint.json` 时，脚本可能先在“冻结划分与 processed 数据指纹一致”报 FAIL；
> 两种 FAIL 都算测试通过——重点是改动必须被发现。

### 测试 2：删一张图 → 校验必须报 FAIL

```powershell
Move-Item reports/figures/fig3_delay_dist.png .tmp/fig3_delay_dist.png
python scripts/verify.py --config configs/base.yaml   # 期望 FAIL：5 张图存在且非空
Move-Item .tmp/fig3_delay_dist.png reports/figures/fig3_delay_dist.png
```

### 测试 3：把目标编码改成全量拟合 → 观察 CV 虚高多少 ⭐ 最有价值

```powershell
Copy-Item src/models/train.py .tmp/train_backup.py -Force
@'
from pathlib import Path
p = Path("src/models/train.py")
t = p.read_text(encoding="utf-8")
old = 'TargetEncoder("Flight Distance", smooth=20.0).fit(Xtr, ytr)'
new = 'TargetEncoder("Flight Distance", smooth=20.0).fit(X, y)'
assert t.count(old) == 1, t.count(old)
p.write_text(t.replace(old, new), encoding="utf-8")
'@ | python -
python -m src.models.train --config configs/base.yaml --model lgbm --groups base,C,T --no-log
Copy-Item .tmp/train_backup.py src/models/train.py -Force   # 立即还原，别提交
```

> 只替换 `TargetEncoder(...)` 这一处，**不要**把后面的 `m.fit(Xtr, ytr)` 也换掉；
> 若训练中途报错，先执行最后一行还原 `src/models/train.py`，再排查。

**历史实测结果**（当时实现实测；B 组分箱 / 依赖版本 / 线程数变化后重跑可能略有差异）：

| TE 拟合方式 | OOF AUC |
|---|---:|
| 折内（正确） | 0.960191 |
| 全量（错误） | 0.960640 |
| 虚高 | **+0.00045** |

这个数字本身就是一个值得写进报告的发现：

> 目标编码写错（全量拟合）在这个数据集上只让 CV 虚高 0.00045。
> 原因是平滑参数 m=20 把自泄漏稀释掉了——Flight Distance 有 3,474 个取值、
> 每条约 200 行，单行的自身标签在平滑后权重很小。
> **也就是说：这类泄漏在本数据上"小到看不出来"，靠分数是发现不了的，只能靠代码审查。**

这正是你验收第 ① 项时必须看代码、而不是看分数的原因。

---

## 5. 这个脚本验不了什么（明确边界）

| 验不了 | 原因 | 你自己怎么补 |
|---|---|---|
| 数据与官方字节一致 | 用的是公开镜像 | 用官方数据重下后重跑，对比 base 是否仍为 0.958829 |
| 结论的统计显著性 | 3 seeds 只是方差估计，不是正式检验 | `python -m src.eval.stability --config configs/base.yaml --seeds 42,2026,7`；正式显著性仍需配对检验/更多划分 |
| test 上是否泄漏 | test 无标签 | 只能靠代码审查：test 只用于 count encoding 与最终预测 |
| 图好不好看、文字通不通 | 主观 | 人工过一遍 |
| 官方 Public LB | 本组没有提交记录 | 不要引用 0.9642 当作“我们的 LB” |

---

## 6. 验收记录（建议每次都留一行）

```
日期        验收人    结构校验              人工 7 项   结论
2026-10-31  P2        33/33 通过（0 SKIP）   7 项通过    可交付（待 PPT 导出 / 打印 / 彩排）
```
