VENV_CMD = virtualenv -p /usr/bin/python3
GOLD_FILE = gold/stw/stw-concept2subthesaurus.csv

DATA_URL = "https://ndownloader.figshare.com/files/15620744"
GOLD_URL = "https://ndownloader.figshare.com/files/15620792"


all: experiment

venv:
	$(VENV_CMD) venv

deps: venv
	. venv/bin/activate && pip install -r requirements-stable.txt

$(GOLD_FILE):
	dirname $(GOLD_FILE) | xargs mkdir -p 
	wget $(GOLD_URL) -O $(GOLD_FILE)

fetch_sample_data:
	mkdir -p data/sample
	wget $(DATA_URL) -O data/sample/sample.zip
	cd data/sample && unzip -u sample.zip

data/sample/annotation.csv: fetch_sample_data

data/sample/authorship.csv: fetch_sample_data

data/sample/paper.csv: fetch_sample_data

out/lsa/embedding.csv: data/sample/paper.csv data/sample/annotation.csv data/sample/paper.csv deps
	. venv/bin/activate && python3 train.py lsa data/sample --no-cuda -o out/lsa


out/deepwalk/embedding.csv: data/sample/paper.csv data/sample/annotation.csv data/sample/paper.csv deps
	. venv/bin/activate && python3 train.py deepwalk data/sample --no-cuda -o out/deepwalk --epochs 5


out/gcn/embedding.csv: data/sample/paper.csv data/sample/annotation.csv data/sample/paper.csv deps
	. venv/bin/activate && python3 train.py gcn_cv_sc data/sample --no-cuda -o out/gcn --epochs 400


eval: out/lsa/embedding.csv out/deepwalk/embedding.csv out/gcn/embedding.csv $(GOLD_FILE)
	. venv/bin/activate && python3 cluster.py -s $(GOLD_FILE) -r 10 --normalize --plot out/lsa/embedding.csv out/deepwalk/embedding.csv out/gcn/embedding.csv
	. venv/bin/activate && python3 classify.py $(GOLD_FILE) --cv 10 --normalize out/lsa/embedding.csv out/deepwalk/embedding.csv out/gcn/embedding.csv


experiment: eval
