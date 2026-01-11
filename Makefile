PYTHON ?= python3

.PHONY: build
build-ext-release:
	CFLAGS="-O3 -DNDEBUG" CXXFLAGS="-O3 -DNDEBUG" $(PYTHON) -m pip install -e .

.PHONY: test
test: build-ext-release
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pytest -q
