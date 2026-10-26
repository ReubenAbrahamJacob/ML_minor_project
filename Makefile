PY=.venv/bin/python
export SSL_CERT_FILE := $(shell $(PY) -c "import certifi;print(certifi.where())" 2>/dev/null)

setup:
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -e . 
data:
	$(PY) scripts/download_data.py
	@echo "XJTU rar parts need: unar -o data/raw/xjtu/extracted_unar data/raw/xjtu/download/Data/XJTU-SY_Bearing_Datasets.part01.rar"
features:
	$(PY) scripts/build_features.py
train:
	$(PY) scripts/run_cwru.py && $(PY) scripts/run_femto.py && $(PY) scripts/run_uncertainty.py
eval: train
	$(PY) scripts/run_explain.py && $(PY) scripts/run_cross.py && $(PY) scripts/run_cmapss.py
	$(PY) scripts/make_figures.py && $(PY) scripts/make_report.py
app:
	.venv/bin/streamlit run app/streamlit_app.py
test:
	$(PY) -m pytest -q && .venv/bin/ruff check src scripts tests app
.PHONY: setup data features train eval app test
