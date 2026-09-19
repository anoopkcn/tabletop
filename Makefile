PYTHON ?= python3
WORKERS ?= 10
DATA ?= data

.PHONY: help dataset clean-grid smoke mini figures preview verify test lint all

help:
	@echo "tabletop"
	@echo
	@echo "  make dataset      render the full grid   -> $(DATA)/tabletop.h5   (294,912 states, ~5 GB, ~6 min on $(WORKERS) cores)"
	@echo "  make clean-grid   the nuisance-free twin -> $(DATA)/tabletop_clean.h5"
	@echo "  make smoke        the 6,144-state subset -> $(DATA)/smoke.h5"
	@echo "  make mini         the 512-state subset   -> $(DATA)/mini.h5      (the one tracked in git)"
	@echo "  make all          all four of the above"
	@echo
	@echo "  make figures      redraw the README figures into assets/"
	@echo "  make preview      the one-row-per-factor sheet -> assets/preview.png"
	@echo "  make verify       digests of every grid in $(DATA), and a re-render spot check"
	@echo "  make test         run the test suite"
	@echo
	@echo "  WORKERS=$(WORKERS)  DATA=$(DATA)"

dataset: $(DATA)/tabletop.h5
$(DATA)/tabletop.h5:
	$(PYTHON) -m tabletop --out $@ --workers $(WORKERS)

clean-grid: $(DATA)/tabletop_clean.h5
$(DATA)/tabletop_clean.h5:
	$(PYTHON) -m tabletop --out $@ --workers $(WORKERS) --no-nuisance

smoke: $(DATA)/smoke.h5
$(DATA)/smoke.h5:
	$(PYTHON) -m tabletop --out $@ --workers $(WORKERS) --smoke

mini: $(DATA)/mini.h5
$(DATA)/mini.h5:
	$(PYTHON) -m tabletop --out $@ --workers $(WORKERS) --mini

all: dataset clean-grid smoke mini

figures:
	$(PYTHON) scripts/make_demo_images.py

preview:
	$(PYTHON) -m tabletop --preview assets/preview.png

verify:
	$(PYTHON) scripts/verify.py $(wildcard $(DATA)/*.h5) --against CHECKSUMS.txt --rerender 32

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check .
