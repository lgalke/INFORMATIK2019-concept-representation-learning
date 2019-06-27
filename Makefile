all: experiment

venv:
	virtualenv -p /usr/bin/python3 venv

deps: venv
	. venv/bin/activate && pip install -r requirements.txt



sample_data:
	mkdir -p data/sample
	test -z data/sample/sample.zip || cp /home/lgalke/git/lab/qgraph/data/rel/ECONIS-STW-eng-100k/econbiz-100k.zip data/sample/sample.zip
	cd data/sample && unzip -u sample.zip


experiment: sample_data deps
	mkdir -p out
	. venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python3 main.py lsa data/sample --no-cuda -o out/gcn
	. venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python3 main.py deepwalk data/sample --no-cuda -o out/gcn
	. venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python3 main.py gcn_cv_sc data/sample --no-cuda -o out/gcn
