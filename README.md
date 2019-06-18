# Installation

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

