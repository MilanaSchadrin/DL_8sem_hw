SHELL := /bin/bash

install:
	poetry install

hooks:
	poetry run pre-commit install --install-hooks

load:
	@echo "Downloading and unpacking data..."
	poetry run python load.py

train:
	@echo "Training completed"
	poetry run python train.py

test:
	@echo "Running test"
	poetry run python test.py

sim:
	@echo "Running SIMCLR"
	poetry run python sim.py

pytest:
	@echo "Running pytest"
	poetry run pytest tests.py

clean:
	@echo "Cleaning"
	rm -rf __pycache__
	rm -rf .pytest_cache
