PY ?= /local/MQICHU/envs/l2606_simplemask_refact/bin/python

.PHONY: ui test lint

# Regenerate the Qt UI module from the Designer .ui source (adds the license header).
# Also runnable without make: python src/pysimplemask/gui/view/compile_ui.py
ui:
	$(PY) src/pysimplemask/gui/view/compile_ui.py

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check src tests
