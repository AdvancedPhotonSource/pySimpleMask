PY ?= /local/MQICHU/envs/l2606_simplemask_refact/bin/python
UIC ?= /local/MQICHU/envs/l2606_simplemask_refact/bin/pyside6-uic

.PHONY: ui test lint

# Regenerate the Qt UI module from the Designer .ui source.
# Also runnable without make: python src/pysimplemask/gui/view/compile_ui.py
ui:
	$(UIC) src/pysimplemask/gui/view/mask.ui -o src/pysimplemask/gui/view/ui_mask.py

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check src tests
