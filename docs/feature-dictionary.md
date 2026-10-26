# 特征字典 v1

> 由 `python scripts/gen_feature_dict.py` 自动生成，列清单直接取自训练时的特征矩阵。
> 改动特征后重新生成即可，**不需要手工维护**——这就是「字段全覆盖」的保证。

## 汇总

| 组 | 含义 | 列数 |
|---|---|---:|
| base | 原始字段 | 22 |
| A | 语义修正（0 = 不适用） | 13 |
| B | 聚合 / 交互 | 11 |
| C | 频次编码 | 2 |
| T | 目标编码（折内拟合） | 1 |
| **合计** | | **49** |

## 字段明细

| 列名 | 组 | 类型 | 唯一值 | 含义 | 处理策略 |
|---|---|---|---:|---|---|
| `Age` | base | int64 | 75 | int | 乘客年龄 7–85 |
| `Flight Distance` | base | int64 | 3474 | int | 航段距离（英里），高基数，可作航线代理 |
| `Inflight wifi service` | base | int64 | 6 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `Departure/Arrival time convenient` | base | int64 | 6 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `Ease of Online booking` | base | int64 | 6 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `Gate location` | base | int64 | 6 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `Food and drink` | base | int64 | 6 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `Online boarding` | base | int64 | 6 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `Seat comfort` | base | int64 | 6 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `Inflight entertainment` | base | int64 | 6 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `On-board service` | base | int64 | 6 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `Leg room service` | base | int64 | 6 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `Baggage handling` | base | int64 | 5 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `Checkin service` | base | int64 | 6 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `Cleanliness` | base | int64 | 6 | 0 = 不适用（不是最差）；1–5 评分 | 0 单独标记 + 0→NaN |
| `Departure Delay in Minutes` | base | int64 | 167 | int | 起飞延误（分钟），约 90% 为 0 |
| `Arrival Delay in Minutes` | base | float64 | 172 | float | 到达延误（分钟），0.04% 缺失；**飞行后才可知** |
| `Gender` | base | object | 2 | str | 性别 |
| `Customer Type` | base | object | 2 | str | 常旅客 / 非常旅客 |
| `Type of Travel` | base | object | 2 | str | 商务出行 / 个人出行 |
| `Class` | base | object | 3 | str | 舱位 Business / Eco / Eco Plus |
| `arr_delay_missing` | base | int8 | 2 | int8 | 到达延误是否缺失 |
| `Inflight wifi service__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 Inflight wifi service） | 把「不适用」从「最差」里拆出来 |
| `Departure/Arrival time convenient__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 Departure/Arrival time convenient） | 把「不适用」从「最差」里拆出来 |
| `Ease of Online booking__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 Ease of Online booking） | 把「不适用」从「最差」里拆出来 |
| `Gate location__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 Gate location） | 把「不适用」从「最差」里拆出来 |
| `Food and drink__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 Food and drink） | 把「不适用」从「最差」里拆出来 |
| `Online boarding__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 Online boarding） | 把「不适用」从「最差」里拆出来 |
| `Seat comfort__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 Seat comfort） | 把「不适用」从「最差」里拆出来 |
| `Inflight entertainment__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 Inflight entertainment） | 把「不适用」从「最差」里拆出来 |
| `On-board service__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 On-board service） | 把「不适用」从「最差」里拆出来 |
| `Leg room service__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 Leg room service） | 把「不适用」从「最差」里拆出来 |
| `Baggage handling__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 Baggage handling） | 把「不适用」从「最差」里拆出来 |
| `Checkin service__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 Checkin service） | 把「不适用」从「最差」里拆出来 |
| `Cleanliness__is_zero` | A | int8 |  | 1 表示该评分填了 0（服务不适用）（基于 Cleanliness） | 把「不适用」从「最差」里拆出来 |
| `n_zeros` | B | int16 |  | 一行里评 0 的列数 | 整体「不适用」程度 |
| `n_fives` | B | int16 |  | 一行里评 5 的列数 | 整体满意度 |
| `svc_mean` | B | float64 |  | 13 项评分中 1–5 部分的均值（0 已转为 NaN） | 服务综合水平 |
| `svc_min` | B | float64 |  | 13 项评分中的最低分 | 短板效应 |
| `svc_std` | B | float64 |  | 13 项评分的标准差 | 服务一致性 |
| `travel_class` | B | category |  | Type of Travel × Class 交叉 | 类别交互（线性模型必需） |
| `total_delay` | B | float64 |  | 起飞延误 + 到达延误 | 总延误 |
| `arr_minus_dep` | B | float64 |  | 到达延误 − 起飞延误 | 空中追回/损失的时间 |
| `is_delayed` | B | int8 |  | 起飞是否延误 > 0 | 延误与否的二值化 |
| `delay_log` | B | float64 |  | log1p(总延误) | 压长尾，线性模型友好 |
| `dist_bin` | B | float32 |  | Flight Distance 的 10 分位分箱 | 距离的非线性 |
| `Flight Distance__count` | C | float32 |  | 该取值在全量（train+test）出现的次数（基于 Flight Distance） | 频次编码：只用特征、不用标签，无泄漏 |
| `Age__count` | C | float32 |  | 该取值在全量（train+test）出现的次数（基于 Age） | 频次编码：只用特征、不用标签，无泄漏 |
| `Flight Distance__te` | T | float64 |  | 目标编码（平滑 m=20），**只在训练折内拟合**（基于 Flight Distance） | 高基数特征的均值编码 |

## 使用约定

1. `base` 永远可用；A/B/C/T 通过 `--groups` 组合。
2. `B` 隐含 `A`：不做 0→NaN 的语义修正，聚合特征没有业务含义。
3. `T` 的目标编码**只在训练折 fit**，验证折与 test 只 transform。
4. `Arrival Delay in Minutes` 是飞行后才可知的事后特征，业务口径若为「飞行前预测」，必须剔除并报告损失。
