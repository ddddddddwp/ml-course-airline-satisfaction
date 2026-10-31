# CHANGELOG

## 2026-10-31 · 可靠性与 P3 交付修复

本轮把项目从“P1/P2 可用、P3 缺失、若干约束靠口头约定”修到“可测试、可验收、可复现”的状态。
主模型分数不变：`base=0.958829`、`base+C+T=0.960191`；最终 submission 与修复前逐位相同。

### 1. 兼容性与正确性

- **pandas 3 兼容**：pandas 3 把字符串列读成 `str` 而非 `object`，原 `detect_schema` 识别不到 4 个类别列，
  LightGBM 直接报 `pandas dtypes must be int, float or bool`。现在 `detect_schema` / `prepare_X`
  统一用 `is_string_dtype` / `is_object_dtype` / `CategoricalDtype` 判断并转 `category`。
- **Windows LightGBM 导入顺序**：`lightgbm` 先于 `sklearn` 加载会触发 OpenMP/DLL 冲突
  （`LGBM_DatasetSetField` access violation）。`diag.py` 已改为先加载 sklearn / 后加载 lightgbm，并加注释。
- **去重修正**：原 `drop_duplicates()` 把唯一 `id` 算进比较键，等于没做；现在按除 `id` 外全列去重，
  `id` 重复单独告警。
- **B 组分箱一致性**：`FeatureBuilder` 在全量 train 上固化为 Flight Distance 的 10 分位边界，
  逐折/验证/测试只 `transform`。默认 `base+C+T` 不含 B，主分数不变；B/ABCT 已用 `_abl2` 重跑。

### 2. 可复现与数据防护

- **CHECKSUMS 真正生效**：`.csv/.txt` 流式 CRLF→LF 归一化后哈希；`make data` 先校验，
  不一致直接报错，只有 `--accept-data-change` 才更新；`verify` 重算哈希而不是只看文件存在。
- **换行统一**：新增 `.gitattributes`（csv/txt/json/py/md/yml/yaml/toml/Makefile → LF；parquet/png 等 binary），
  并把 `data/raw/*.csv` 重检出为 LF。当前原始字节哈希与 `CHECKSUMS.txt` **直接一致**。
- **冻结划分防覆盖**：`freeze` 已存在时默认打印提示并 exit 0（`make all` 可重复），
  `--force` 先备份 `.bak.<timestamp>`；新增 `stamp` 写 `splits/fold_v1.fingerprint.json`
  （data_fingerprint + fold_file_sha256，不改 `fold_of_row`）。
- **run_id 含 seed**：`{owner}_{stage}_{model}_{feat}_{MMDD}_s{seed}[_{tag}]`，
  多 seed 不再互相覆盖 OOF；`predict_test` 也写 `exp_log`（`stage=final`）和 `reports/oof/{run_id}.parquet`。
- **git_commit 不再为空**：优先 `GIT_COMMIT` 环境变量，其次 git，最后 `unknown`；新增 `git_dirty()` / `code_version()`。

### 3. P3 评估体系（此前为空）

- `src/eval/metrics.py`：AUC / LogLoss / Brier / ECE / 阈值指标 / bootstrap CI / 可靠性曲线。
- `src/eval/evaluate.py`：OOF 主评估、校准图、分布图、混淆矩阵、22 个分群表、`metrics.json` / `summary.md`。
- `src/eval/stability.py`：3 seeds × 固定 5 折，输出 `stability.json` / `stability.md` / seed 方差图。
- `src/eval/interpret.py`：LightGBM `pred_contrib=True` 的 TreeSHAP；T 组用 **OOF 目标编码**，避免自泄漏。
- `src/eval/badcase.py`：高置信 FP/FN top-50 + 六分群错误率 + badcase 图。
- `reports/p3_eval_report.md`：完整评估章节；`reports/figures/p3_*.png` 为 300 dpi 图。

**最终数字**：pooled OOF AUC **0.960186**（95% CI 0.959715–0.960605），LogLoss 0.225458，
Brier 0.060432，ECE 0.002479；3 seeds 0.960191 / 0.960159 / 0.960203（均值 0.960184，
seed std 0.000023，平均折内 std 0.000631）。

### 4. 测试、入口与工程配套

- `tests/` 新增 119 项 pytest（数据/schema/划分/特征/TE/predict/指标/评估）。
- `Makefile`：`PY ?= python`；新增 `test / eval / stability / shap / badcase / p3 / seed-check / all`；
  `clean` 改为跨平台 `python scripts/clean.py`。
- 新增 `LICENSE`（MIT）、`pyproject.toml`、`requirements-dev.txt`、
  `.github/workflows/tests.yml`（pytest + compileall）、`scripts/clean.py`。
- `scripts/eda.py` / `plot_importance.py` / `gen_feature_dict.py` / `make_toy_data.py` 全部走 config/`path_of`；
  `figures_p2.py` 跨平台中文字体 + 300 dpi + 基准线动态读 `exp_log`。
- 文档全面对齐：README、plan、data/README、docs/verify、docs/feature-dictionary、PPT 统稿、P2 报告。

### 5. 行为变化 / 迁移提示

- `run_id` 带 `_s{seed}`，历史 run_id 保持原样；报告引用新实验时用新 id。
- `make split` 默认跳过已存在的 `fold_v1.json`；需要重划时显式 `--force`。
- `make data` 在数据被替换时会失败；确认后加 `--accept-data-change`。
- B 组 `dist_bin` 语义修正，B/ABCT 的 1e-5 量级数值变化属预期；base/A/C/T/CT 不变。
- 若你的 clone 仍是 CRLF：`git add --renormalize data/raw && git checkout -- data/raw`；
  不迁移也不影响 `make data` / `verify`，哈希会归一化。

### 6. 答辩 PPT（2026-10-31）

- 新增 `reports/airline_satisfaction_ppt.pptx`：**12 页、16:9、可编辑文本/表格、300 dpi 图、逐页演讲备注**；
  全篇最小字号 **9.0 pt**。
- 新增生成脚本 `scripts/make_ppt.py`（python-pptx）；`requirements-dev.txt` 增加 `python-pptx>=1.0`。
- 结构检查：12 页、9 张图、5 个表格、12 页备注齐全；文本溢出启发式检查 0 处。
- **未完成**：PDF 导出、双面打印、预答辩彩排。当前机器的 LibreOffice Kit 渲染器即使打开最小 PPTX 也会崩溃
  （`0xC0000005`），因此未生成 PDF/预览图；请在 PowerPoint 中打开检查后另存 PDF 再打印。

### 7. 验收

```text
python -m pytest -q                                      # 119 passed
python scripts/verify.py --config configs/base.yaml      # 33/33 通过
python scripts/verify.py --config configs/base.yaml --reproduce  # 34/34，base 差 8.01e-08
```

未做/保留：不做模型集成（按 plan §8 风险预案与交接说明）；3 seeds 只是方差估计，不是正式显著性检验；
数据为公开镜像，没有官方 Public LB 提交记录。答辩材料中 PPTX 已生成，PDF/双面打印/彩排仍待完成。
