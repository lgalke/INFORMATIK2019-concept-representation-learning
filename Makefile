all: deps

venv:
	virtualenv -p /usr/bin/python3 venv
	source venv/bin/activate


deps: venv
	pip install -r requirements.txt

