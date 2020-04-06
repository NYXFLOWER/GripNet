from src.layers import *
from src.decoder import multiInnerProductDecoder
from data.utils import process_edge_multirelational
from torch_geometric.data import Data
import sys, time
import pandas as pd


# ###################################
# data processing
# ###################################
# load data
dd = torch.load('data/pose/drug-drug.pt')
gd = torch.load('data/pose/gene-drug.pt')
gg = torch.load('data/pose/gene-gene.pt')

# ###################################
# dataset integration
# ###################################
# add dd graph
index = dd.edge_index.copy()
ttt = dd.edge_type.copy()
# add gg graph
tmp = gg.edge_index + dd.n_node
index.append(tmp)
tmp = torch.tensor([len(index)] * gg.n_edge)
ttt.append(tmp)
# add gd graph
tmp = gd.edge_index
tmp[0] = tmp[0] + dd.n_node
index.append(tmp)
tmp = torch.tensor([len(index)] * gd.n_edge)
ttt.append(tmp)

data = Data.from_dict({
    'n_node': dd.n_node + gg.n_node,
    'n_node_type': 2,
    'n_edge': dd.n_edge + gd.n_edge + gg.n_edge,
    'n_edge_type': dd.n_edge_type + gd.n_edge_type + gg.n_edge_type,
    'node_type': [('drug', 'drug'), ('gene', 'drug'), ('gene', 'gene')],

    'edge_index': index,
    'edge_type': ttt,
    'edge_weight': None,

    'description': 'reindex node [drug]+[gene]'
})

torch.save(data, './data/pose_comb.pt')


