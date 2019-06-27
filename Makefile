VENV_CMD = virtualenv -p /usr/bin/python3
GOLD_FILE = gold/stw/concept2subthesaurus.csv

all: experiment

venv:
	$(VENV_CMD) venv

deps: venv
	. venv/bin/activate && pip install -r requirements-stable.txt


$(GOLD_FILE):
	dirname $(GOLD_FILE) | xargs mkdir -p 
	cp /data21/lgalke/qaktiv/gold/concept2subthesaurus.csv $(GOLD_FILE)

sample_data:
	mkdir -p data/sample
	test -z data/sample/sample.zip || cp /home/lgalke/git/lab/qgraph/data/rel/ECONIS-STW-eng-100k/econbiz-100k.zip data/sample/sample.zip
	cd data/sample && unzip -u sample.zip


train: sample_data deps
	mkdir -p out
	. venv/bin/activate && python3 train.py lsa data/sample --no-cuda -o out/lsa
	. venv/bin/activate && python3 train.py deepwalk data/sample --no-cuda -o out/deepwalk
	. venv/bin/activate && python3 train.py gcn_cv_sc data/sample --no-cuda -o out/gcn
	
eval: train $(GOLD_FILE)
	. venv/bin/activate && python3 cluster.py -s $(GOLD_FILE) -r 10 --normalize --plot out/lsa/embedding.csv out/deepwalk/embedding.csv out/gcn/embedding.csv
	. venv/bin/activate && python3 classify.py $(GOLD_FILE) --cv 10 --normalize out/lsa/embedding.csv out/deepwalk/embedding.csv out/gcn/embedding.csv


experiment: eval
