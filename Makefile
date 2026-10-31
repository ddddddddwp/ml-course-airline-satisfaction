# 跨平台 Makefile
#   Windows：PY 默认 python（系统里通常没有 python3）
#   Linux/macOS：若只有 python3，用 make <target> PY=python3
PY ?= python
CONFIG ?= configs/base.yaml
export PYTHONPATH := .

.PHONY: help data split split-compare train abl eda importance fdict drop-arrival \
        p1 test eval stability shap badcase p3 seed-check all \
        verify verify-full smoke clean

help:
	@echo "用法: make <target> [CONFIG=configs/base.yaml] [PY=python]"
	@echo ""
	@echo "数据 / 划分"
	@echo "  make data          # raw -> processed（清洗 + 校验和）"
	@echo "  make split         # 冻结 folds 到 splits/fold_v1.json（P1-05）"
	@echo "  make split-compare # 三种划分策略对比（P1-05 交付物）"
	@echo ""
	@echo "训练 / 特征 / 图表"
	@echo "  make train         # 单次 5 折训练（默认 lgbm + base 特征）"
	@echo "  make abl           # 消融矩阵：2 模型族 x 6 特征组（P1-12）"
	@echo "  make eda           # 4 张 EDA 图（P1-03 / P1-21）"
	@echo "  make importance    # 特征重要性 top-20（P1-11）"
	@echo "  make fdict         # 生成特征字典（P1-07）"
	@echo "  make drop-arrival  # 剔除事后特征 Arrival Delay（P1-04）"
	@echo "  make p1            # 上面 P1 交付物全跑一遍（约 15 分钟）"
	@echo ""
	@echo "P3 评估"
	@echo "  make eval          # 最终评估：混淆矩阵 / ROC / PR / 校准"
	@echo "  make stability     # 稳定性：bootstrap / seed 方差"
	@echo "  make shap          # 可解释性：SHAP summary"
	@echo "  make badcase       # 误差分析：badcase top-50"
	@echo "  make p3            # eval + stability + shap + badcase"
	@echo "  make seed-check    # 等价于 stability（seed 方差自检）"
	@echo ""
	@echo "回归 / 验收 / 清理"
	@echo "  make test          # pytest -q"
	@echo "  make verify        # 交付物验收（约 15 秒）"
	@echo "  make verify-full   # 验收 + 重跑 base 复现（约 50 秒）"
	@echo "  make smoke         # 用合成数据跑通全链路（不需要 Kaggle）"
	@echo "  make clean         # 跨平台清理派生数据（python scripts/clean.py）"
	@echo "  make all           # data split train p1 p3 verify 一条龙"
	@echo ""
	@echo "换配置: make smoke CONFIG=configs/toy.yaml"
	@echo "Windows: PY 默认 python。Linux/macOS 若只有 python3：make data PY=python3"
	@echo "没装 make 时：把 make xxx 换成 README 里对应的 python 命令即可。"

data:
	$(PY) -m src.data.make_dataset --config $(CONFIG)

split:
	$(PY) -m src.common.split freeze --config $(CONFIG)

split-compare:
	$(PY) -m src.common.split compare --config $(CONFIG) --score

train:
	$(PY) -m src.models.train --config $(CONFIG) --model lgbm --groups base

abl:
	@for m in lgbm lr; do \
	  for g in base base,A base,B base,C base,C,T base,A,B,C,T; do \
	    $(PY) -m src.models.train --config $(CONFIG) --model $$m --groups $$g --tag abl || exit 1; \
	  done; \
	done

eda:
	$(PY) scripts/eda.py --config $(CONFIG)

importance:
	$(PY) scripts/plot_importance.py --config $(CONFIG) --groups base,C,T

fdict:
	$(PY) scripts/gen_feature_dict.py --config $(CONFIG)

drop-arrival:
	$(PY) -m src.models.train --config $(CONFIG) --model lgbm --groups base,C,T \
	  --drop-cols "Arrival Delay in Minutes,arr_delay_missing" --tag noarr

p1: split-compare eda importance fdict drop-arrival
	@echo "== P1 交付物已全部生成：划分对比 / EDA 图 / 重要性 / 特征字典 / 事后特征实验 =="

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m src.eval.evaluate --config $(CONFIG)

stability:
	$(PY) -m src.eval.stability --config $(CONFIG)

shap:
	$(PY) -m src.eval.interpret --config $(CONFIG)

badcase:
	$(PY) -m src.eval.badcase --config $(CONFIG)

p3: eval stability shap badcase
	@echo "== P3 交付物已全部生成：评估 / 稳定性 / 可解释性 / 误差分析 =="

seed-check: stability
	@echo "== seed-check 完成：seed 方差与稳定性结论见 reports/ =="

all: data split train p1 p3 verify
	@echo "== 全链路完成：data -> split -> train -> p1 -> p3 -> verify =="

verify:
	$(PY) scripts/verify.py --config $(CONFIG)

verify-full:
	$(PY) scripts/verify.py --config $(CONFIG) --reproduce

smoke:
	@echo "== 1/4 生成合成数据（结构与 S6E10 一致）=="
	$(PY) scripts/make_toy_data.py --out data/toy/raw
	@echo "== 2/4 清洗 =="
	$(PY) -m src.data.make_dataset --config configs/toy.yaml
	@echo "== 3/4 冻结 folds =="
	$(PY) -m src.common.split freeze --config configs/toy.yaml
	@echo "== 4/4 训练（base+A+B+C）=="
	$(PY) -m src.models.train --config configs/toy.yaml --model lgbm --groups base,A,B,C --tag smoke
	@echo "== 闭环通过：raw -> processed -> folds -> oof AUC -> exp_log =="

clean:
	$(PY) scripts/clean.py
