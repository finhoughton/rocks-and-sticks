PYTHON ?= python3

.PHONY: test
test:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pytest -q
