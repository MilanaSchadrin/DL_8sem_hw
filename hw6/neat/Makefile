PYTHON = python
TEST = pytest


train:
	@echo "Training model"
	$(PYTHON) app.py --mode train

play-human:
	@echo "Human play mode"
	$(PYTHON) app.py --mode human

play-ai:
	@echo "Model plays"
	$(PYTHON) app.py --mode AI

test:
	@echo "Tests"
	$(TEST) test.py

dvc-run:
	@echo "Reproducing DVC pipeline"
	dvc repro

clean:
	@echo "Cleaning"
	rm -f best_genome.pkl
	rm -rf graphs
	rm -f *.json