PYTHON=python
TEST = pytest
PIP = pip

train:
	@echo "Training"
	$(PYTHON) agent.py

play:
	@echo "Play"
	$(PYTHON) app.py

test:
	@echo "Test"
	$(TEST) test.py

clean:
	@echo "Cleaning files"
	rm -f flappy_model.pth training_rewards.png

requirements:
	@echo "Installing requirements"
	$(PIP) install -r requirements.txt