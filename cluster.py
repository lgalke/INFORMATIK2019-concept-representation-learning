#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
File: plot_tsne
Author: Lukas Galke
Email: vim@lpag.de
Github: https://github.com/lgalke
Description: Script to visualize an embedding via TSNE
"""

import os
import argparse
from collections import defaultdict


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, LabelEncoder


# STW has 101 concepts on first-level after subthesauri roots

from utils import load_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddingfile", nargs='+',
                        help="Path to embedding file")
    # The default of 7 is derived from the number of subthesauri in STW
    parser.add_argument("-k", type=int, default=7,
                        help="Number of clusters (7)")
    parser.add_argument("-r", "--repetitions", type=int, default=1,
                        help="Number of repetitions (1)")
    parser.add_argument("--plot", action='store_true', default=False,
                        help="Draw TSNE plots")
    parser.add_argument("--normalize", action='store_true', default=False,
                        help="Normalize vectors to unit L2-length before anything.")
    parser.add_argument("-o", "--outdir", default=None,
                        help="Put outfiles (plots) in this dir.")
    parser.add_argument("-p", "--perplexity", default=30.0, type=float,
                        help="Perplexity for TSNE visualization")
    parser.add_argument("-s", "--supervised", default=None, help="Compute V-measure with respect to a ground truth")
    args = parser.parse_args()

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if args.supervised:
        print("~ Supervised clustering eval against %s ~" % args.supervised)
    else:
        print("~ Applying k-Means with %d clusters ~" % args.k)
    print("~ Unit L2 norm: %s ~" % args.normalize)
    print("~ Mean and std dev of %d repetitions ~" % args.repetitions)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for i, embedding_file in enumerate(args.embeddingfile):
        print("=" * 78)
        print("File:", embedding_file)
        df_embedding = load_embedding(embedding_file, as_dataframe=True)

        if args.supervised:
            df_gold = pd.read_csv(args.supervised, index_col=0)
            df = df_gold.join(df_embedding, how='inner')
            le = LabelEncoder()
            y = le.fit_transform(df['top'].values)
            X = df[df.columns[1:]].values
            is_supervised = True
            print("Inferring num clusters from ground truth.")
            n_clusters = len(le.classes_)
            print("n_clusters = %d" % n_clusters)
        else:
            X = df_embedding.values
            is_supervised = False
            n_clusters = args.k

        if args.normalize:
            X = normalize(X, norm='l2', axis=1)
        scores = defaultdict(list)
        for run in range(args.repetitions):
            # clustering
            kmeans = KMeans(n_clusters=n_clusters)
            # print(kmeans)
            kmeans.fit(X)
            s = metrics.silhouette_score(X, kmeans.labels_)
            ch = metrics.calinski_harabasz_score(X, kmeans.labels_)

            scores['s'].append(s)
            scores['ch'].append(ch)

            if is_supervised:
                h, c, v = metrics.homogeneity_completeness_v_measure(y, kmeans.labels_)
                scores['h'].append(h)
                scores['c'].append(c)
                scores['v'].append(v)
                r = metrics.adjusted_rand_score(y, kmeans.labels_)
                scores['r'].append(r)

        scores = {m: (np.mean(r), np.std(r)) for m, r in scores.items()}

        ext = ".cluster-k{}-r{}".format(n_clusters, args.repetitions)
        if args.normalize:
            ext += "-l2"
        print("Silhouette: {:.4f} (SD: {:.4f})".format(*scores['s']))
        print("Calinski-Harabasz: {:.4f} (SD: {:.4f})".format(*scores['ch']))
        if is_supervised:
            print("Homogeneity: {:.4f} (SD: {:.4f})".format(*scores['h']))
            print("Completeness: {:.4f} (SD: {:.4f})".format(*scores['c']))
            print("V measure: {:.4f} (SD: {:.4f})".format(*scores['v']))
            print("Adjusted Rand index: {:.4f} (SD: {:.4f})".format(*scores['r']))
        results_path = embedding_file + ext
        with open(results_path,'w') as results_file:
            print("Silhouette: {:.4f} (SD: {:.4f})".format(*scores['s']), file=results_file)
            print("Calinski-Harabasz: {:.4f} (SD: {:.4f})".format(*scores['ch']), file=results_file)
            if is_supervised:
                print("Homogeneity: {:.4f} (SD: {:.4f})".format(*scores['h']), file=results_file)
                print("Completeness: {:.4f} (SD: {:.4f})".format(*scores['c']), file=results_file)
                print("V measure: {:.4f} (SD: {:.4f})".format(*scores['v']), file=results_file)
                print("Adjust Rand index: {:.4f} (SD: {:.4f})".format(*scores['r']), file=results_file)

        if args.plot:
            # 2d visualization via tsne
            tsne = TSNE(n_components=2, perplexity=args.perplexity, learning_rate=200.0)
            print(tsne)
            # print("Applying t-distributed Stochastic Neighbor Embedding")
            x_2d = tsne.fit_transform(X)
            print("KL Divergence: %.4f" % tsne.kl_divergence_)


            plot_path = embedding_file + '-k{}'.format(n_clusters)
            if args.normalize:
                plot_path += "-l2"
            plot_path += '-p{}.png'.format(int(args.perplexity))
            plt.figure(i)
            # Labels from final kmeans iteration
            plt.scatter(x_2d[:, 0], x_2d[:, 1], s=1, c=kmeans.labels_, marker=",")
            plt.savefig(plot_path)
        print("=" * 78)

