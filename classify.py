#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: evaluate_embedding.py
Author: Lukas Galke
Email: vim@lpag.de
Github: https://github.com/lgalke
Description: Evaluates an embedding by running a classifier on top.
"""

import pandas as pd

from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score

from utils import load_embedding

def main(args):
    df_gold = pd.read_csv(args.goldstandard, index_col=0)

    for emb_path in args.embedding:
        print("=" * 78)
        print("Processing embedding file:", emb_path)
        print("-" * 78)
        df_embedding = load_embedding(emb_path, as_dataframe=True)


        # align embedding and gold standard
        df = df_gold.join(df_embedding, how='inner')
        # df = pd.merge(df_gold, df_embedding, left_index=True, right_index=True, how='inner')

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(df['top'].values)

        # First column is label column
        X = df[df.columns[1:]].values

        print("N examples", X.shape[0])
        print("N targets", len(le.classes_))

        if args.normalize:
            print("Normalizing...")
            X = preprocessing.normalize(X, norm='l2')

        # Linear SVM with default parameters
        clf = svm.SVC(kernel=args.kernel)

        print("Running {}-cross-validated SVM with {} kernel...".format(args.cv, args.kernel))
        scores = cross_val_score(clf, X, y, cv=args.cv)

        print("Accuracy scores", scores)

        print("Accuracy mean/std:", scores.mean(), scores.std())
        print("=" * 78)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("goldstandard", help="Path to file created by `find_subthesauri_for_concepts`")
    parser.add_argument("embedding", nargs="+", help="Path to embedding file")
    parser.add_argument("--cv", default=10, type=int, help="Cross-validation number")
    parser.add_argument("--kernel", default='linear', type=str, help="SVM Kernal",
                        choices=['linear','poly','rbf', 'sigmoid'])
    parser.add_argument("--normalize", default=False, action='store_true', help="Normalize embedding")

    args = parser.parse_args()
    main(args)
