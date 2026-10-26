# 材料验收指南

> 原则：**不信结论，看证据。** 能机械验的交脚本，验不了的给检查方法和合格线。

---

## 0. 一条命令

```bash
make verify        # 29 项结构校验，约 15 秒
make verify-full   # 再加一项"重跑 base 并与记录对比"，约 50 秒
```

当前结果：

```
结论：30/30 通过
重跑 base 0.958829 vs 记录 0.958829，差 8.01e-08
```

退出码非 0 就是有 FAIL，脚本会把未通过项单独列出来。

---

## 1. 验收分三层

| 层 | 验什么 | 谁做 | 耗时 |
|---|---|---|---|
| **L1 机械校验** | 形状、缺失、划分完整性、字典覆盖、登记表一致、产物存在 | `make verify` | 15 s |
| **L2 人工判断** | 代码是否真折内、图与文字是否一致、结论有没有越界 | 你 | 10 min |
| **L3 独立复现** | 换一个人跑同一份数据，分数是否一致到 1e-6 | P2 / P3 | 5 min |

---

## 2. L2：人工必查的 6 项

### ① 目标编码是不是真的折内 ⭐ 最关键

看 `src/models/train.py` 第 148–153 行：

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

### ② 消融是不是"同 fold、同模型、只换特征"

```bash
python - <<'EOF'
import pandas as pd
d = pd.read_csv("reports/exp_log.csv")
print(d[d.run_id.str.contains("abl")][["run_id","model","split_ver","params_hash","cv_mean"]].to_string(index=False))
EOF
```

**合格**：同模型的行 `split_ver` 全是 `v1`，`params_hash` 相同（只差特征数时不同）。
**不合格**：有人中途重建了 fold，或换了学习率——那消融表就不能横着比。

### ③ 图和文字对不对得上

打开 `reports/figures/fig4_rating_vs_target.png`，找 Online boarding：
**0 分处应约 62%，1 分处应约 10.5%**。对不上就说明文字和图有一边是错的。

### ④ 结论有没有超出证据

| 结论 | 可以写 | 不能写 |
|---|---|---|
| A/B 组负增益 | "在本数据集、LightGBM 上，聚合类特征无增益" | "特征工程没有用" |
| C+T +0.0014 | "单 seed、固定划分下 +0.0014" | "显著提升"（未做多 seed） |
| 0=不适用 | "0 分的满意率高于 1 分，说明 0 是'不适用'" | "0 分代表服务差" |

### ⑤ 数据来源能不能交代

`data/README.md` 第 2.1 节写了镜像来源与还原度证据。若课程要求官方数据，这里需要更新。

### ⑥ 报告里的数字能不能对上 run_id

抽查 3 个数字（比如 base、C、C+T），在 `reports/exp_log.csv` 里按 `run_id` 找到对应行，
`cv_mean` 必须完全一致。

---

## 3. L3：独立复现（M1 验收）

让 P2、P3 各自在自己的目录跑：

```bash
make train
```

三个人的 `OOF AUC` 必须一致到 **1e-6**。不一致，基本就是：
- 有人重建了 fold（没读 `fold_v1.json`）
- 有人对数据做了排序/采样
- 有人改了 `configs/base.yaml`

---

## 4. 破坏性测试：验证"验收本身"有效

只会 PASS 的检查等于没有检查。这三个测试会让你确认脚本真的在算。

### 测试 1：把划分打乱 → 校验必须报 FAIL

```bash
cp splits/fold_v1.json /tmp/fold_backup.json
python - <<'EOF'
import json, numpy as np
p = "splits/fold_v1.json"
d = json.load(open(p))
d["fold_of_row"] = np.random.default_rng(0).integers(0, d["n_splits"], len(d["fold_of_row"])).tolist()
json.dump(d, open(p, "w"))
EOF
make verify          # 期望：FAIL「各折正例率一致」
cp /tmp/fold_backup.json splits/fold_v1.json   # 还原
```

### 测试 2：删一张图 → 校验必须报 FAIL

```bash
mv reports/figures/fig3_delay_dist.png /tmp/
make verify          # 期望：FAIL「5 张图存在且非空」
mv /tmp/fig3_delay_dist.png reports/figures/
```

### 测试 3：把目标编码改成全量拟合 → 观察 CV 虚高多少 ⭐ 最有价值

```bash
cp src/models/train.py /tmp/train_backup.py
sed -i 's/\.fit(Xtr, ytr)/.fit(X, y)/' src/models/train.py
python3 -m src.models.train --config configs/base.yaml --model lgbm --groups base,C,T --no-log
cp /tmp/train_backup.py src/models/train.py
```

**实测结果**（我跑过）：

| TE 拟合方式 | OOF AUC |
|---|---:|
| 折内（正确） | 0.96019 |
| 全量（错误） | 0.96064 |
| 虚高 | **+0.00045** |

这个数字本身就是一个值得写进报告的发现：

> 目标编码写错（全量拟合）在这个数据集上只让 CV 虚高 0.00045。
> 原因是平滑参数 m=20 把自泄漏稀释掉了——Flight Distance 有 3200 个取值、
> 每个约 200 行，单行的自身标签在平滑后权重很小。
> **也就是说：这类泄漏在本数据上"小到看不出来"，靠分数是发现不了的，只能靠代码审查。**

这正是你验收第 ① 项时必须看代码、而不是看分数的原因。

---

## 5. 这个脚本验不了什么（明确边界）

| 验不了 | 原因 | 你自己怎么补 |
|---|---|---|
| 数据与官方字节一致 | 用的是公开镜像 | 用官方数据重下后重跑，对比 base 是否仍为 0.958829 |
| 结论的统计显著性 | 只有单 seed | 3 seeds × 5 folds（`make seed-check` 尚未实现） |
| test 上是否泄漏 | test 无标签 | 只能靠代码审查：test 只用于 count encoding 与最终预测 |
| 图好不好看、文字通不通 | 主观 | 人工过一遍 |

---

## 6. 验收记录（建议每次都留一行）

```
日期        验收人    make verify      人工 6 项   结论
2026-10-26  P1        30/30 通过       通过        可交付
```
