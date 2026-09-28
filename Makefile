.PHONY: help setup preprocess baseline xgboost ablation sporadic figures test all clean

PY := python

help:
	@echo "setup       install dependencies"
	@echo "preprocess  raw PPMI  ->  data/processed/ppmi_tdpigd_long.csv"
	@echo "baseline    majority / logistic / HistGradientBoosting baselines"
	@echo "xgboost     XGBoost baseline"
	@echo "ablation    feature-group ablation study"
	@echo "sporadic    hardest test: sporadic PD, baseline visit only"
	@echo "figures     regenerate every figure in reports/figures"
	@echo "test        run the test suite"
	@echo "all         preprocess -> baseline -> xgboost -> ablation -> figures"

setup:
	$(PY) -m pip install -r requirements.txt

preprocess:
	$(PY) src/trace_pd/data/preprocess.py

baseline:
	$(PY) src/trace_pd/models/train_baseline.py

xgboost:
	$(PY) src/trace_pd/models/train_xgboost.py

ablation:
	$(PY) src/trace_pd/evaluation/ablation.py

sporadic:
	$(PY) src/trace_pd/evaluation/sporadic_baseline.py

figures:
	$(PY) src/trace_pd/viz/dataset_figures.py
	$(PY) src/trace_pd/viz/architecture_hld.py
	$(PY) src/trace_pd/viz/problem_chart.py
	$(PY) src/trace_pd/viz/gap_chart.py
	$(PY) src/trace_pd/viz/impact_diagram.py

test:
	$(PY) -m pytest tests -v

all: preprocess baseline xgboost ablation sporadic figures

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
