PYTHON = python
TEST = pytest
PIP = pip

download_data:
   @echo "Download data"
   $(PYTHON) get_data.py
run: 
   @echo "Run everething"
   $(PYTHON) main.py

test:
	@echo "Tests"
	$(TEST) test.py

clean:
    @echo "Cleaning files"
	find . -name "*.pyc" -exec rm -f {} \;
	find . -name "__pycache__" -exec rm -rf {} \;

requirements:
	@echo "Installing requirements"
	$(PIP) install -r requirements.txt
