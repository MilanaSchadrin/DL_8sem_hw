PYTHON=python
TEST = pytest
PIP = pip

.PHONY: all data train evaluate summarize visualize test clean

all: data train evaluate visualize

data:
    $(PYTHON) data.py

train:
    $(PYTHON) train.py\
		--pretrained_embeddings=$(PRETRAINED_EMBEDDINGS) \
		--use_label_smoothing=$(USE_LABEL_SMOOTHING)

evaluate:
    $(PYTHON) eval.py

summarize:
    $(PYTHON) sum.py

visualize:
    $(PYTHON) visualization.py

test:
	@echo "Test"
	$(TEST) tests.py

clean:
    rm -rf saved_data/*
	rm -rf results/*

requirements:
	@echo "Installing requirements"
	$(PIP) install -r requirements.txt