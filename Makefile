# create venv

train:
	cloudexe --gpuspec H100x1 -- /root/xp-layoutlmv3/.venv/bin/python3 train.py 

.venv:
	python3 -m venv .venv

.PHONY: install
install: 
	@pip install -r requirements.txt

.PHONY: clean
clean:
	@rm -f .flor.json

