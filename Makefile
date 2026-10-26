PY ?= python3
CONFIG ?= configs/base.yaml
export PYTHONPATH := .

.PHONY: help data split split-compare train abl smoke clean

help:
	@echo "make data          # raw -> processed（清洗 + 校验和）"
	@echo "make split         # 冻结 folds 到 splits/fold_v1.json（P1-05）"
	@echo "make split-compare # 三种划分策略对比（P1-05 交付物）"
	@echo "make train         # 单次 5 折训练（默认 lgbm + base 特征）"
	@echo "make abl           # 消融矩阵：2 模型族 x 5 特征组（P1-12）"
	@echo "make smoke         # 用合成数据跑通全链路（不需要 Kaggle）"
	@echo ""
	@echo "换数据集: make smoke CONFIG=configs/toy.yaml"

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
	rm -rf data/toy data/processed data/interim reports/oof
	rm -f splits/toy_fold.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
