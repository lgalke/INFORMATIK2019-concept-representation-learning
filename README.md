# Quick start

Run `make` to setup a virtual environment, download sample data, and run the experiments.
If there are any issues, or the methods should be applied to other data, follow the detailed steps below.

# Setup

Set up a virtual environment (i.e. `virtualenv -p /usr/bin/python3 venv && source venv/bin/activate`) and then run:

```sh
pip install -r requirements.txt
```

# Preparation

A graph directory `graph_dir` with three csv files:

- paper.csv with columns: paper\_id, year, title
- annotation.csv with columns: paper\_id, subject
- authorship.csv with columns: paper\_id, author


# Usage


```python3 main.py gcn_cv_sc graph_dir```

For more information on hyperparameters, consult `python3 main.py -h`.

