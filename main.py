#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
File: unsupervised
Author: Lukas Galke
Email: vim@lpag.de
Github: https://github.com/lgalke
Description: Script to run supervised experiments on bibliographic data
"""

import itertools as it
import logging
import os
import random
import pickle
from collections import defaultdict

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.contrib.sampling.sampler import NeighborSampler
from gensim.models import Word2Vec
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from data import PAPER_TYPE, SUBJECT_TYPE, load_data
from models.gcn_cv_sc import GCNSampling
from models.deepwalk import graph as deepwalk_graph
from models.lsa import LSAEmbedding, class_prototypes
from preprocessing import AlphaNumericTextPreprocessor
from utils import save_embedding

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEBUG = False


# Model directory structure
MDS = {
    "embedding_file": "embedding.csv",
    "args_file": "args.txt",
    "loss_file": "losses.txt",
    "accuracy_file": "accuracy.txt",
    "temporal_embeddings_dir": "temporal_embeddings",
    "ccount_file": "cumulative_counts.csv",
    "checkpoints_dir": "checkpoints"
}

def preprocess_text(args, data):
    """ Preprocess text """
    print("Preprocessing text...")
    preprocessor = AlphaNumericTextPreprocessor(max_features=args.max_features,
                                                lowercase=True,
                                                max_length=args.max_length,
                                                stop_words=ENGLISH_STOP_WORDS,
                                                drop_unknown=True,
                                                dtype=torch.tensor)
    titles = data.paper_features['title'].values
    text_features = preprocessor.fit_transform(titles)
    return text_features, preprocessor


def embed_lsa(args, data):
    """ Embeds data via latent semantic analysis """
    lsa = LSAEmbedding(n_components=args.representation_size,
                       max_features=args.max_features,
                       stop_words='english')
    titles = data.paper_features['title'].fillna('')
    x_papers = lsa.fit_transform(titles)
    print("Explained variance {:.4%}".format(lsa.svd.explained_variance_ratio_.sum()))
    y_labels = data.label_indicator_matrix()

    subject_embedding = class_prototypes(x_papers, y_labels)
    subject_ids = data.ndata[data.ndata.type == SUBJECT_TYPE]["identifier"].values
    return subject_ids, subject_embedding


def embed_control_variate(args, data):
    """embed_control_variate
    Learns an embedding with GCNs with control variate sampling and skip-connections.
    Valid args.model keys are 'graphsage_cv' or 'gcn_cv_sc'.

    :param args: Namespace for command line arguments
    :param data: BiblioGraph instance
    """
    # Can go to command line args later
    num_neighbors = args.num_neighbors
    n_layers = args.n_layers
    dropout = args.dropout
    emb_size = args.embedding_dim
    n_hidden = args.n_hidden
    rep_size = args.representation_size

    device = torch.device("cuda") if args.use_cuda else torch.device("cpu")
    globals_device = device if not args.globals_on_cpu else torch.device("cpu")

    # Preparing lists of vertices
    subjects = data.ndata[data.ndata.type == SUBJECT_TYPE]
    subject_vs = subjects.index.values
    paper_vs = data.ndata[data.ndata.type == PAPER_TYPE].index.values
    print(subject_vs)
    print(len(subject_vs))

    print("Nxgraph")
    print("Number of nodes:", data.graph.number_of_nodes())
    print("Number of edges:", data.graph.number_of_edges())

    if args.warm_start:
        try:
            with open(os.path.join(args.out, "preprocessor.pkl", 'rb')) as fh:
                preprocessor = pickle.load(fh)
            text_features = preprocessor.transform(data.paper_features['title'].values)
        except FileNotFoundError:
            print("Warning: warm start without restoring preprocessor...")
            print("Vocabulary will be recreated.")
            text_features, preprocessor = preprocess_text(args, data)
    else:
        text_features, preprocessor = preprocess_text(args, data)


    print("Text feature dims", text_features.size())


    with open(os.path.join(args.out, "preprocessor.pkl"), 'wb') as fh:
        print("Saving preprocessor to", fh.name)
        pickle.dump(preprocessor, fh)



    print("Creating DGL graph ...")
    g = dgl.DGLGraph(data.graph, readonly=True)
    g.set_n_initializer(dgl.init.zero_initializer)
    print("DGL Graph")
    print("Number of nodes:", g.number_of_nodes())
    print("Number of edges:", g.number_of_edges())

    print("Is multigraph:",g.is_multigraph())
    print("Setting zero initializer")
    g.set_n_initializer(dgl.init.zero_initializer)

    ### INIT 'features'
    print("Adding features")
    features = torch.zeros(g.number_of_nodes(), text_features.size(1),
                           device=text_features.device, dtype=text_features.dtype)
    features[paper_vs] = text_features
    g.ndata['features'] = features.to(globals_device)
    print("Feats size", features.size())

    print("Mapping subject nids to class label")
    print("Subj values", subject_vs)
    subject2classlabel = defaultdict(lambda: -1, {nid:c for c, nid in enumerate(subject_vs)})
    # print("Map", subject2classlabel)
    n_classes = len(subject2classlabel)
    targets = torch.zeros(g.number_of_nodes(), dtype=torch.int64) - 1
    # -1 as default since paper nodes shouldnt be subject to optim target anyways
    for nid in subject_vs:
        targets[nid] = subject2classlabel[nid]

    targets = targets.to(device)

    print("Subject targets:", targets[subject_vs])
    print("Number of classes", n_classes)

    g.ndata['h_{}'.format(0)] = torch.zeros(g.number_of_nodes(),
                                            emb_size,
                                            device=globals_device)
    for i in range(1, n_layers):
        g.ndata['h_{}'.format(i)] = torch.zeros(g.number_of_nodes(),
                                                n_hidden,
                                                device=globals_device)


    # penultimate skip-connection layer
    if n_layers > 1:
        # We don't use skip connections when there is only one layer
        g.ndata['h_{}'.format(n_layers-1)] = torch.zeros(g.number_of_nodes(),
                                                         2*n_hidden,
                                                         device=globals_device)
    # For two layers
    # h_0 : [N, emb_size]
    # h_1 : [N, 2*n_hidden]

    ### INIT 'norm' (both gcn/graphsage use this norm)
    print("Computing global norm...")
    norm = 1./g.in_degrees().float().unsqueeze(1)
    norm[torch.isinf(norm)] = 0.  # Prevent INF values
    # print("Spurious rows", data.ndata[norm.numpy() == 0.])
    print("Norm", norm, sep='\n')
    print("Norm size", norm.size())
    g.ndata['norm'] = norm.to(globals_device)

    text_encoder = nn.Embedding(len(preprocessor.vocabulary_) + 1,
                                emb_size,
                                sparse=(not args.scale_grad_by_freq), padding_idx=0,
                                max_norm=args.max_norm,
                                scale_grad_by_freq=args.scale_grad_by_freq)

    if args.embedding_dropout:
        text_encoder = nn.Sequential(text_encoder, nn.Dropout(args.embedding_dropout))

    model = GCNSampling(emb_size,
                        n_hidden,
                        rep_size,
                        n_layers,
                        F.relu,
                        dropout)

    # Linear decoder
    decoder = nn.Linear(rep_size, n_classes, bias=args.decoder_bias)

    if args.representation_dropout \
            or args.representation_activation \
            or args.representation_layer_norm:
        pp_modules = []
        if args.representation_dropout and args.representation_layer_norm:
            print("warning: dropout and layer norm might not go well together")
        if args.representation_dropout:
            pp_modules.append(nn.Dropout(args.representation_dropout))
        if args.representation_activation:
            act = getattr(torch.nn, args.representation_activation)()
            pp_modules.append(act)
        if args.representation_layer_norm:
            pp_modules.append(nn.LayerNorm(rep_size, elementwise_affine=False))
        model = nn.Sequential(model, *pp_modules)

    if args.warm_start:
        print("Loading model checkpoint from", args.out)
        text_encoder.load_state_dict(torch.load(os.path.join(args.out, 'best_text_encoder.pkl')))
        model.load_state_dict(torch.load(os.path.join(args.out, 'best_model.pkl')))
        decoder.load_state_dict(torch.load(os.path.join(args.out, 'best_decoder.pkl')))

    if args.use_cuda:
        # Keep large embedding on CPU when globals are on CPU
        text_encoder = text_encoder.to(globals_device)
        model = model.to(device)
        decoder = decoder.to(device)




    loss_fcn = nn.CrossEntropyLoss()

    if args.scale_grad_by_freq:
        embed_optimizer = optim.Adam(text_encoder.parameters(), lr=args.lr)
    else:
        embed_optimizer = optim.SparseAdam(text_encoder.parameters(), lr=args.lr)
    model_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(text_encoder)
    print(model)
    print(decoder)

    ############
    # Training #
    ############

    def validate(text_encoder, model, decoder, g, subject_vs, targets):
        # Closure, uses args of outer scope
        num_acc = 0.
        val_loss = 0.
        text_encoder.eval()
        model.eval()
        decoder.eval()
        # Evaluate accuracy
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
                                                       g.number_of_nodes(),  # expand factor
                                                       neighbor_type='in',
                                                       num_hops=n_layers,
                                                       seed_nodes=subject_vs,
                                                       add_self_loop=False,
                                                       num_workers=args.workers,
                                                       prefetch=True):
            # Copy data from global graph
            node_embed_names = [['features']]
            for i in range(n_layers):
                node_embed_names.append(['norm'])

            nf.copy_from_parent(node_embed_names=node_embed_names)

            with torch.no_grad():
                nf.apply_layer(0, lambda node:
                               {'embed': text_encoder(node.data['features']).mean(1)})
                z = model(nf)
                pred = decoder(z)
                batch_nids = nf.layer_parent_nid(-1)
                batch_targets = targets[batch_nids]
                loss = loss_fcn(pred, batch_targets)
                num_acc += (torch.argmax(pred, dim=1) == batch_targets).sum().item()
                val_loss += loss.detach().item() * nf.layer_size(-1)
                # batch_labels = torch.tensor(
                #  [subject2classlabel[nid.item()] for nid in batch_nids],
                #                             dtype=torch.long,
                #                             device=device)

        accuracy = num_acc / len(subjects)
        val_loss = val_loss / len(subjects)

        return val_loss, accuracy

    losses = []
    step = 0
    if args.early_stopping:
        # Init early stopping 
        cnt_wait = 0
        best = 1e9
        best_t = 0

    for epoch in range(args.epochs):
        # Make sure everything is in train mode!!!
        text_encoder.train()
        model.train()
        decoder.train()
        epoch_loss = 0.
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                       num_neighbors,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_hops=n_layers,
                                                       add_self_loop=False,
                                                       seed_nodes=subject_vs,
                                                       num_workers=args.workers,
                                                       prefetch=True):
            step += 1
            # Fill aggregate history from neighbors
            for i in range(n_layers):
                agg_history_str = 'agg_h_{}'.format(i)
                g.pull(nf.layer_parent_nid(i+1), fn.copy_src(src='h_{}'.format(i), out='m'),
                       fn.sum(msg='m', out=agg_history_str),
                       lambda node: {agg_history_str: node.data[agg_history_str] * node.data['norm']})
            
            # Copy data from parent
            node_embed_names = [['features', 'h_0']]
            for i in range(1, n_layers):
                node_embed_names.append(['h_{}'.format(i), 'agg_h_{}'.format(i-1)])
            node_embed_names.append(['agg_h_{}'.format(n_layers-1)])
            nf.copy_from_parent(node_embed_names=node_embed_names)

            # forward
            model_optimizer.zero_grad()
            embed_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # First, apply the text encoder
            # nf.layers[0].data['features'] = text_encoder(nf.layers[0].data['features'])
            nf.apply_layer(0, lambda node: {'embed': text_encoder(node.data['features']).mean(1)})
            z = model(nf)
            pred = decoder(z)

            batch_nids = nf.layer_parent_nid(-1)
            batch_targets = targets[batch_nids]
            loss = loss_fcn(pred, batch_targets)
            # backward
            loss.backward()

            model_optimizer.step()
            embed_optimizer.step()
            decoder_optimizer.step()

            node_embed_names = [['h_{}'.format(i)] for i in range(n_layers)]
            node_embed_names.append([])

            nf.copy_to_parent(node_embed_names=node_embed_names)

            # Loss is sample-averaged, for epoch loss, we need to multiply it
            # back in and divide later
            epoch_loss += loss.detach().item() * nf.layer_size(-1)
            # print("Step {:7d} | Epoch {:4d} | Loss {:.4f}".format(step, epoch+1, loss.item()))


        avg_epoch_loss = epoch_loss / len(subjects)
        # Now expand to all nodes for getting final represenations
        with open(os.path.join(args.out, MDS['loss_file']), 'a') as lossfile:
            print("{:.4f}".format(avg_epoch_loss), file=lossfile)


        if args.early_stopping:
            if avg_epoch_loss < best:
                best = avg_epoch_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(text_encoder.state_dict(), os.path.join(args.out, 'best_text_encoder.pkl'))
                torch.save(model.state_dict(), os.path.join(args.out, 'best_model.pkl'))
                torch.save(decoder.state_dict(), os.path.join(args.out, 'best_decoder.pkl'))
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print("Early stopping!")
                # Break out of main training loop if early-stopping criterion is met
                break

        if args.fastmode:
            # In fast mode, only evaluate accuracy in final epoch!
            print("Step {:7d} | Epoch {:4d} | Train Loss: {:.4f}".format(step,
                                                                         epoch,
                                                                         avg_epoch_loss))
            # Skip per-epoch computation of validation accuracy/loss
            continue

        val_loss, accuracy = validate(text_encoder, model, decoder, g, subject_vs, targets)
        with open(os.path.join(args.out, MDS['accuracy_file']), 'a') as accfile:
            print("Step {:7d} | Epoch {:4d} | Train loss: {:.4f} | Eval loss: {:.4f} | Accuracy {:.4f}"
                  .format(step, epoch, avg_epoch_loss, val_loss, accuracy), file=accfile)
        print("Step {:7d} | Epoch {:4d} | Train Loss: {:.4f} | Eval loss: {:.4f} | Accuracy {:.4f}"
              .format(step, epoch, avg_epoch_loss, val_loss, accuracy))

    if args.early_stopping:
        print('Loading {}th epoch'.format(best_t))
        text_encoder.load_state_dict(torch.load(os.path.join(args.out, 'best_text_encoder.pkl')))
        model.load_state_dict(torch.load(os.path.join(args.out, 'best_model.pkl')))
        decoder.load_state_dict(torch.load(os.path.join(args.out, 'best_decoder.pkl')))
        # For logging purposes
        epoch = best_t



    print("Shift models on cpu...")
    # PUT EVERYTHING STILL NEEDED ON CPU
    # WE DONT WANT TO RUN INTO MEM ISSUES HERE
    text_encoder = text_encoder.cpu()
    model = model.cpu()
    decoder = decoder.cpu()

    g.ndata['features'] = g.ndata['features'].cpu()
    g.ndata['norm'] = g.ndata['norm'].cpu()
    targets = targets.cpu()

    for i in range(n_layers):
        g.ndata.pop('h_{}'.format(i))
        g.ndata.pop('agg_h_{}'.format(i))

    # Put stuff in eval mode, no dropout and stuff
    text_encoder.eval()
    model.eval()
    decoder.eval()

    # and comp accuracy on the fly
    print("Computing final decoding accuracy")
    val_loss, accuracy = validate(text_encoder, model, decoder, g, subject_vs, targets)
    with open(os.path.join(args.out, MDS['accuracy_file']), 'a') as accfile:
        print("Step {:7d} | Epoch {:4d} | Eval loss: {:.4f} | Accuracy {:.4f}"
              .format(step, epoch, val_loss, accuracy), file=accfile)
    print("Step {:7d} | Epoch {:4d} | Eval loss: {:.4f} | Accuracy {:.4f}"
          .format(step, epoch, val_loss, accuracy))



    # Preprocess text encoding
    # Save representation
    def embedding_fn(features, graph, node_ids):
        # Closure to compute embedding from subgraph
        embedding = torch.zeros(graph.number_of_nodes(), rep_size)
        model.eval()
        graph.ndata['embed'] = features
        print("Computing norm...")
        print(graph)
        norm = 1./graph.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0.  # Prevent INF values
        graph.ndata['norm'] = norm

        # For now, we only do it for subject_vs
        for nf in dgl.contrib.sampling.NeighborSampler(graph, args.test_batch_size,
                                                       graph.number_of_nodes(),  #expand factor
                                                       neighbor_type='in',
                                                       num_hops=n_layers,
                                                       seed_nodes=node_ids,
                                                       num_workers=args.workers,
                                                       prefetch=True,
                                                       add_self_loop=True):  # Self-loop required for temporal stuff
            # Copy data from global graph
            node_embed_names = [['embed']]
            for i in range(n_layers):
                node_embed_names.append(['norm'])
            nf.copy_from_parent(node_embed_names=node_embed_names)
            with torch.no_grad():
                z = model(nf)
                embedding[nf.layer_parent_nid(-1)] = z
        # Cleanup
        graph.ndata.pop('embed')
        graph.ndata.pop('norm')
        return embedding[node_ids]

    features = text_encoder(g.ndata.pop('features')).mean(1)

    print("g just before temporal embeddings (after dropping features):", g, sep='\n')

    concepts = data.ndata[data.ndata.type == SUBJECT_TYPE]["identifier"]
    concept_nids, descriptors = concepts.index.values, concepts.values


    print("Computing global embeddings...")
    # Global embedding
    representation = embedding_fn(features, g, concept_nids).numpy()
    print("Done.")
    return descriptors, representation



def embed_deepwalk(args, data):

    # BEGIN QND
    # TODO make this program args
    args.seed = 'seed'
    # END QND

    epochs = args.epochs

    g = deepwalk_graph.from_networkx(data.graph, undirected=True)

    is_concept_node = lambda n: data.ndata.type[int(n)] == SUBJECT_TYPE
    concept_node_count = len([n for n in g.nodes() if is_concept_node(n)])

    print("Number of nodes: {}".format(concept_node_count))

    num_walks = concept_node_count * args.number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * args.walk_length

    print("Data size (walks*length): {}".format(data_size))

    print("Walking...")
    walks = deepwalk_graph.build_deepwalk_corpus(g, num_paths=args.number_walks,
                                                 path_length=args.walk_length, alpha=0,
                                                 rand=random.Random(args.seed),
                                                 node_start_condition=is_concept_node)
    print("Training...")
    model = Word2Vec(walks,
                     size=args.representation_size,
                     window=args.window_size,
                     min_count=1,
                     sg=1,
                     hs=1,
                     workers=args.workers,
                     sorted_vocab=0,
                     sample=0,
                     iter=epochs)

    # DONE word2vec seems to build an own vocabulary. We need to map it back to our indices
    # subject_ids = data.ndata[data.ndata.type == SUBJECT_TYPE]["identifier"].values
    # We don't actually need all vectors, but just concept vectors
    # subject_vs = data.ndata[data.ndata.type == SUBJECT_TYPE].index.values

    subjects = data.ndata[data.ndata.type == SUBJECT_TYPE]["identifier"]

    # Init final embedding
    # subject_embedding = np.zeros((len(subjects), args.representation_size))

    subject_embedding = model.wv[subjects.index.values.astype('U')]

    subject_ids = subjects.values

    return subject_ids, subject_embedding



def embed_random(args, data):
    df_annotation = pd.read_csv(os.path.join(args.graphdir, 'annotation.csv'))
    concepts = df_annotation['subject'].unique()
    embedding = np.random.rand(len(concepts), args.representation_size)
    return concepts, embedding

def main(args):
    if args.use_cuda and not torch.cuda.is_available():
        print("Cuda not available, falling back to CPU")


    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    print("Loading data...")
    data = load_data(args.graphdir, supervised=False,
                     with_authors=args.use_authors,
                     collate_coauthorship=(not args.first_class_authors),
                     undirected=True)

    # Switch case on main training function
    labels, embedding = {
        'random': embed_random,
        'lsa': embed_lsa,
        'deepwalk': embed_deepwalk,
        'gcn_cv_sc': embed_control_variate
    }[args.model](args, data)


    with open(os.path.join(args.out, MDS['args_file']), 'w') as argsfile:
        print(args, file=argsfile)


    embedding_file = os.path.join(args.out, MDS['embedding_file'])
    save_embedding(labels, embedding, embedding_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model',
                        choices=['lsa',
                                 'gae',
                                 'deepwalk',
                                 'dgi',
                                 'gcn_cv_sc',
                                 'random'],
                        help="Select model for representation learning")
    parser.add_argument('graphdir',
                        help='path to graph dir')
    parser.add_argument('--min-count', type=int, default=5,
                        help="Minimum count for words")
    parser.add_argument('--max-features', type=int, default=50000,
                        help="Vocabulary size for words")
    parser.add_argument('--max-length', type=int, default=None,
                        help="Maximum word count of titles")
    parser.add_argument('-s', '--representation-size', type=int, default=32,
                        help="Target embedding dimension")

    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="Weight decay")
    parser.add_argument('-o', '--out',
                        help='Output directory')
    parser.add_argument('--epochs', default=200, type=int,
                        help="Number of epochs")
    parser.add_argument('--no-cuda', dest='use_cuda', default=True,
                        action='store_false',
                        help='Do not use GPU processing!')
    # Deepwalk-specific params
    parser.add_argument("--number-walks", default=5, type=int, help="Number of walks per node in Deepwalk")
    parser.add_argument("--walk-length", default=3, type=int, help="Walk length in Deepwalk")
    parser.add_argument("--window-size", default=3, type=int, help="Window size for skip-gram in Deepwalk")
    parser.add_argument("--workers", default=4, type=int, help="CPU workers for Deepwalk")

    # Batching specific
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--test-batch-size", default=None, type=int, help="Batch size for testing")

    # Text Embedding specific
    parser.add_argument('--embedding-dim', type=int, default=64,
                        help="Text embedding dimension")
    parser.add_argument('--embedding-dropout', type=float, default=0,
                        help="Text embedding dimension")
    parser.add_argument("--max-norm", default=None, help="Max norm for text embedding", type=float)

    # Sampling specific
    parser.add_argument("--num-neighbors", default=2, type=int,
                        help="How many neighbors to sample in each layer")

    # GCN specific
    parser.add_argument("--n-layers", default=2, type=int,
                        help="How many GCN layers to use")
    parser.add_argument("--n-hidden", default=16, type=int,
                        help="How many GCN layers to use")
    parser.add_argument("--dropout", default=0.5, type=float,
                        help="Dropout factor within GCN layers")
    parser.add_argument("--representation-dropout", default=0, type=float,
                        help="Apply dropout on the representation layer")
    parser.add_argument("--representation-activation", default=None, type=str,
                        help="Apply dropout on the representation layer")
    parser.add_argument("--representation-layer-norm", default=False,
                        action='store_true',
                        help="Apply layer norm on the representation layer")
    parser.add_argument("--decoder-bias", default=False, help="Enable bias in decoder",
                        action='store_true')
    parser.add_argument("-f", "--fastmode", default=False, help="Only compute final accuracy.",
                        action='store_true')
    parser.add_argument("--scale-grad-by-freq", default=False, help="Scale grad by batch IDF",
                        action='store_true')

    # Memory management
    parser.add_argument("--globals-on-cpu", default=False, help="Store globals on CPU memory.",
                        action='store_true')

    # Dataset specs
    parser.add_argument("--first-class-authors", default=False, help="Do not collate authorship.",
                        action='store_true')
    parser.add_argument("--no-authors", dest='use_authors', default=True,
                        action='store_false', help="Do not use author data at all.")

    # Early stopping
    parser.add_argument("--early-stopping", default=False, action='store_true', help="Use early stopping")
    parser.add_argument("--patience", default=20, type=int, help="Patience for early stopping")
    parser.add_argument("--warm-start",
                        default=False,
                        action='store_true',
                        help="Restore model checkpoint in outdir for warm start")


    # parser.add_argument("--thesaurus", default=None, help="Thesaurus")

    ARGS = parser.parse_args()
    ARGS.test_batch_size = ARGS.batch_size if ARGS.test_batch_size is None else ARGS.test_batch_size
    print(ARGS)

    if not ARGS.out:
        # Guess some tmp directory to put output
        ARGS.out = os.path.join("/tmp/",
                                os.path.basename(os.path.dirname(ARGS.graphdir)),
                                ARGS.model)

    print("Storing outputs to", ARGS.out)
    os.makedirs(ARGS.out, exist_ok=True)

    main(ARGS)
