.PHONY: fcx c3 split compare rounds c3plus help setup preprocess baseline xgboost ablation sporadic figures docs test all clean

PY := python

help:
	@echo "setup       install dependencies"
	@echo "preprocess  raw PPMI  ->  data/processed/ppmi_tdpigd_long.csv"
	@echo "baseline    majority / logistic / HistGradientBoosting baselines"
	@echo "xgboost     XGBoost baseline"
	@echo "ablation    feature-group ablation study"
	@echo "sporadic    hardest test: sporadic PD, baseline visit only"
	@echo "figures     regenerate every figure in reports/figures"
	@echo "fcx         FCX explainability framework: evaluate all three components + figure"
	@echo "docs        regenerate docs/data_dictionary.md from the processed data"
	@echo "split       create the locked 20% patient test split"
	@echo "compare     step 2: 12-config subtype classifier bake-off"
	@echo "rounds      step 3: per-round models + FCX C1"
	@echo "c3plus      step 4: E5/E6 transition models + FCX C3"
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

fcx:
	$(PY) src/trace_pd/evaluation/evaluate_fcx.py
	$(PY) src/trace_pd/viz/fcx_figures.py

c3:
	$(PY) src/trace_pd/evaluation/validate_c3.py

# --- session additions (need trace_pd importable: pip install -e . or PYTHONPATH=src)
split:
	$(PY) -m trace_pd.evaluation.splits

compare:
	$(PY) -m trace_pd.evaluation.compare_subtype_models

rounds:
	$(PY) -m trace_pd.evaluation.evaluate_c1_rounds

c3plus:
	$(PY) -m trace_pd.evaluation.improve_c3

docs:
	$(PY) src/trace_pd/data/export_dictionary.py

conformal:
	$(PY) -m trace_pd.models.conformal

transition:
	$(PY) -m trace_pd.models.train_transition

demo:
	$(PY) -m trace_pd.inference --demo

test:
	$(PY) -m pytest tests -v

all: preprocess baseline xgboost ablation sporadic fcx conformal transition figures docs

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
