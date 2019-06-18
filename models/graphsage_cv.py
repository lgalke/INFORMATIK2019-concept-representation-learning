import argparse, time, math
import numpy as np
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden,
                 out_feats,
                 dropout,
                 last=False,
                 **kwargs):
        super(GraphSAGELayer, self).__init__(**kwargs)
        self.last = last
        self.dropout = nn.Dropout(dropout) if dropout else None

        self.dense1 = nn.Linear(in_feats, hidden)
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.dense2 = nn.Linear(hidden, out_feats)
        if not self.last:
            self.layer_norm2 = nn.LayerNorm(out_feats)

    def forward(self, h):
        h = self.dense1(h)
        h = self.layer_norm1(h)
        h = F.relu(h)
        if self.dropout:
            h = self.dropout(h)
        h = self.dense2(h)
        if not self.last:
            h = self.layer_norm2(h)
            h = F.relu(h)
        return h


class NodeUpdate(nn.Module):
    def __init__(self, layer_id, in_feats, out_feats, hidden, dropout,
                 last=False):
        super(NodeUpdate, self).__init__()
        self.layer_id = layer_id
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.last = last
        self.layer = GraphSAGELayer(in_feats, hidden, out_feats, dropout, last)

    def forward(self, node):
        h = node.data['h']
        norm = node.data['norm']
        # activation from previous layer of myself
        self_h = node.data['self_h']

        if not self.training:
            h = (h - self_h) * norm
            # graphsage
            h = torch.cat([h, self_h], dim=1)
        else:
            agg_history_str = 'agg_h_{}'.format(self.layer_id-1)
            agg_history = node.data[agg_history_str]
            # normalization constant
            subg_norm = node.data['subg_norm']
            # delta_h (h - history) from previous layer of myself
            self_delta_h = node.data['self_delta_h']
            # control variate
            h = (h - self_delta_h) * subg_norm + agg_history * norm
            # graphsage
            h = torch.cat([h, self_h], dim=1)
            if self.dropout:
                h = self.dropout(h)

        h = self.layer(h)

        return {'activation': h}



class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 **kwargs):
        super(GraphSAGE, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(NodeUpdate(1, 2*in_feats, n_hidden, n_hidden, dropout))
        # hidden layers
        for i in range(2, n_layers):
            self.layers.append(NodeUpdate(i, 2*n_hidden, n_hidden, n_hidden, dropout))

        # output layer
        self.layers.append(NodeUpdate(n_layers, 2*n_hidden, n_classes, n_hidden, dropout, last=True))

    def forward(self, nf):
        h = nf.layers[0].data['embed']


        if self.training:
            for i, layer in enumerate(self.layers):
                parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
                layer_nid = nf.map_from_parent_nid(i, parent_nid)
                self_h = h[layer_nid]
                # activation from previous layer of myself, used in graphSAGE
                nf.layers[i+1].data['self_h'] = self_h

                # new_history = h.clone().detach()
                new_history = h.clone().detach()
                history_str = 'h_{}'.format(i)
                history = nf.layers[i].data[history_str]
                # delta_h used in control variate
                delta_h = h - history
                # delta_h from previous layer of the nodes in (i+1)-th layer, used in control variate
                nf.layers[i+1].data['self_delta_h'] = delta_h[layer_nid]

                nf.layers[i].data['h'] = delta_h
                nf.block_compute(i,
                                 fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'),
                                 layer)
                h = nf.layers[i+1].data.pop('activation')
                # update history
                if i < nf.num_layers-1:
                    nf.layers[i].data[history_str] = new_history
        else:
            # Inference mode
            for i, layer in enumerate(self.layers):
                nf.layers[i].data['h'] = h
                parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
                layer_nid = nf.map_from_parent_nid(i, parent_nid)
                # activation from previous layer of the nodes in (i+1)-th layer, used in graphSAGE
                self_h = h[layer_nid]
                nf.layers[i+1].data['self_h'] = self_h
                nf.block_compute(i,
                                 fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'),
                                 layer)
                h = nf.layers[i+1].data.pop('activation')

        return h
