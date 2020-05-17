import pandas as pd
import torch
from torch_geometric.data import Data
from data.utils import process_node_multilabel

n = 2
l = ['business', 'location', 'organization']

bb = torch.from_numpy(pd.read_csv('data/freebase/book_{}/index_separated/book-book.dat'.format(l[n]), sep='\t', header=None).to_numpy().T)
bp = torch.from_numpy(pd.read_csv('data/freebase/book_{}/index_separated/book-{}.dat'.format(l[n], l[n]), sep='\t', header=None).to_numpy().T)
pp = torch.from_numpy(pd.read_csv('data/freebase/book_{}/index_separated/{}-{}.dat'.format(l[n], l[n], l[n]), sep='\t', header=None).to_numpy().T)

train = torch.from_numpy(pd.read_csv('data/freebase/book_{}/index_separated/label.dat.train'.format(l[n]), sep='\t', header=None).to_numpy().T)
test = torch.from_numpy(pd.read_csv('data/freebase/book_{}/index_separated/label.dat.test'.format(l[n]), sep='\t', header=None).to_numpy().T)


data = Data()
data.aa_edge_idx = bb
data.pa_edge_idx = torch.tensor([bp[1].tolist(), bp[0].tolist()])
data.pp_edge_idx = pp

data.n_a_node = (max(bb.max(), bp[0].max()) + 1).tolist()
data.n_p_node = (max(pp.max(), bp[1].max()) + 1).tolist()

data.n_a_type = (train[1].max() + 1).tolist()

data.n_aa_edge = bb.shape[1]
data.n_pp_edge = pp.shape[1]
data.n_pa_edge = bp.shape[1]

idx, sam = [], []
for nn in range(data.n_a_type):
    tr = train[0][train[1] == nn]
    te = test[0][test[1] == nn]
    idx.append(torch.cat((tr, te)))
    sam.append(idx[nn].__len__())


data.train_node_idx, data.train_node_class, data.train_range, data.test_node_idx, data.test_node_class, data.test_range = process_node_multilabel(idx)
data.test_sample = [(nn[1] - nn[0]).tolist() for nn in data.test_range]
data.train_sample = [(nn[1] - nn[0]).tolist() for nn in data.train_range]

torch.save(data, 'datasets-freebase/book-{}.pt'.format(l[n]))
