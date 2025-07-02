# create venv

.venv:
	python3 -m venv .venv

.PHONY: install
install: 
	@pip install -r requirements.txt

.PHONY: clean
clean:
	@rm -f .flor.json

