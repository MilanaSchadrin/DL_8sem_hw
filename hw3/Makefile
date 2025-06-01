PYTHON=python
TEST = pytest
PIP = pip
PRETRAINED_EMBEDDINGS = false
USE_LABEL_SMOOTHING = false
MODEL_PATH = "./results/model_not_label_sm.pt"

.PHONY: all test clean

all: requirements data train evaluate summarize visualize

data:
    $(PYTHON) data.py

train:
    $(PYTHON) main.py\
		--pretrained_embeddings=$(PRETRAINED_EMBEDDINGS) \
		--use_label_smoothing=$(USE_LABEL_SMOOTHING)

evaluate:
    $(PYTHON) eval.py --model_path=$(MODEL_PATH)

summarize:
    $(PYTHON) sum.py --model_path=$(MODEL_PATH)


visualize:
    $(PYTHON) visualization.py --model_path=$(MODEL_PATH)

test:
	@echo "Test"
	$(TEST) tests.py

clean:
    rm -rf saved_data/*
	rm -rf results/*
	rm -rf metrics/*
	rm -rf vis/
	rm -rf 1/ 2/ 3/

requirements:
	@echo "Installing requirements"
	$(PIP) install -r requirements.txt