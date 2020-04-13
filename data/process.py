from torch_geometric.data import Data
from data.utils import process_edge_multirelational, process_node_multilabel
import pickle
import torch
import numpy as np
torch.manual_seed(1111)
np.random.seed(1111)

# ###################################
# MIP-pose dataset
# ###################################
with open("data/tipexp_data.pkl", 'rb') as f:
    data = pickle.load(f)

dd_edge_index = torch.cat([data.train_idx, data.test_idx], dim=1)
dd_edge_type = torch.cat([data.train_et, data.test_et])

index = []
ttt = []
for i in range(len(data.train_range)):
    print(i)
    idx = torch.cat([data.train_idx[:, data.train_range[i][0]: data.train_range[i][1]],
                     data.test_idx[:, data.test_range[i][0]: data.test_range[i][1]]], dim=1)
    tt = torch.cat([data.train_et[data.train_range[i][0]: data.train_range[i][1]],
                    data.test_et[data.test_range[i][0]: data.test_range[i][1]]])
    index.append(idx)
    ttt.append(tt)

drug = Data.from_dict({
    'n_node': data.n_drug,
    'n_node_type': 1,
    'n_edge': sum(data.n_edges_per_type),
    'n_edge_type': data.n_et,
    'node_type': ('drug', 'drug'),

    'edge_index': index,
    'edge_type': ttt,
    'edge_weight': None
})
torch.save(drug, './data/pose/drug-drug.pt')

gene = Data.from_dict({
    'n_node': data.n_prot,
    'n_node_type': 1,
    'n_edge': data.pp_index.shape[1],
    'n_edge_type': 1,
    'node_type': ('gene', 'gene'),

    'edge_index': data.pp_index,
    'edge_type': None,
    'edge_weight': None
})
torch.save(gene, './data/pose/gene-gene.pt')

gene_drug = Data.from_dict({
    'n_node': data.n_prot + data.n_drug,
    'n_node_type': 2,
    'n_edge': data.pd_index.shape[1],
    'n_edge_type': 1,
    'node_type': ('gene', 'drug'),

    'edge_index': data.pd_index,
    'edge_type': None,
    'edge_weight': None
})
torch.save(gene_drug, './data/pose/gene-drug.pt')

idx_to_id_dict = {
    'drug': data.drug_idx_to_id,
    'prot': data.prot_idx_to_id,
    'side_effect': data.side_effect_idx_to_name
}

with open('./data/pose/pose_map.pkl', 'wb') as f:
    pickle.dump(idx_to_id_dict, f)


# ###################################
# comb-pose dataset
# ###################################
# load data
dd = torch.load('data/pose/drug-drug.pt')
gd = torch.load('data/pose/gene-drug.pt')
gg = torch.load('data/pose/gene-gene.pt')

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

# train and test set
values = process_edge_multirelational(index, p=0.9)

# construction
data = Data.from_dict({
    'n_node': dd.n_node + gg.n_node,
    'n_drug': dd.n_node,
    'n_gene': gg.n_node,
    'n_node_typed': 2,
    'n_edge': dd.n_edge + gd.n_edge + gg.n_edge,
    'n_edge_type': dd.n_edge_type + gd.n_edge_type + gg.n_edge_type,
    'node_type': [('drug', 'drug'), ('gene', 'drug'), ('gene', 'gene')],

    'edge_index': index,
    'edge_type': ttt,
    'edge_weight': None,

    'train_idx': values[0],
    'train_et': values[1],
    'train_range': values[2],
    'test_idx': values[3],
    'test_et': values[4],
    'test_range': values[5],

    'description': 'reindex node [drug]+[gene]'
})

tmp = data.train_idx[:, data.train_range[-1][0]:data.train_range[-1][1]].shape[1]
tmp = -int(tmp/2)
data.train_idx = data.train_idx[:, :tmp]
data.train_et = data.train_et[:tmp]
data.train_range[-1][1] += tmp

tmp = data.test_idx[:, data.test_range[-1][0]:data.test_range[-1][1]].shape[1]
tmp = -int(tmp/2)
data.test_idx = data.test_idx[:, :tmp]
data.test_et = data.test_et[:tmp]
data.test_range[-1][1] += tmp

torch.save(data, './data/pose-1-combl.pt')


# ###################################
# citation network dataset
# ###################################
aa = torch.load("data/citation_network/author-author.pt")
pa = torch.load("data/citation_network/paper-author.pt")
pp = torch.load("data/citation_network/paper-paper.pt")

# tmp = aa.node_label
# label = [[] for i in range(aa.n_node_type)]
# for idx, lab in tmp:
#     label[lab].append(idx)
# label = [torch.tensor(i, dtype=torch.int) for i in label]
# aa.typed_node_label = label
# torch.save(aa, "data/citation_network/author-author.pt")

data = Data()
data.train_idx, data.train_label, data.train_range, data.test_idx, data.test_label, data.test_range = process_node_multilabel(aa.typed_node_label)

data.aa_edge_index, data.pp_edge_index, data.pa_edge_index = aa.edge_index, pp.edge_index, pa.edge_index
data.a_node_label = aa.typed_node_label
data.n_a_node = len(set(aa.edge_index.unique().tolist()) | set(pa.edge_index[1].unique().tolist()))
data.n_a_node_type = aa.n_node_type
data.n_aa_edge = aa.n_edge
data.n_aa_edge_type = 1

data.n_p_node = pp.n_node

un = pa.edge_index[1].unique()
tmp = []
for i in range(aa.edge_index.shape[1]):
    x, y = aa.edge_index[:, i].tolist()
    if x in un and y in un:
        tmp.append([x, y])

tmp = np.array(tmp).reshape((2, -1))
data.aa_edge_index = torch.tensor(tmp)

no_connect_a = set(data.pa_edge_index[1].unique().tolist()) - set(data.aa_edge_index.unique().tolist())
data.a_no_connect = no_connect_a

no_connect_p = set(data.pp_edge_index.unique().tolist()) - set(data.pa_edge_index[0].unique().tolist())
data.p_no_connect = no_connect_p

data.n_p_node = 46704
data.n_pp_edge = pp.edge_index.shape[1]
data.n_pa_edge = pa.edge_index.shape[1]

torch.save(data, "./data/auta-0.pt")


# ###################################
# pose-0
# ###################################
data = torch.load("./datasets/pose-2.pt")
g_node = set(data.gd_edge_index[0].unique().tolist())
data.n_g_node = len(g_node)

pp = data.gg_edge_index.numpy()
tmp = []
for i in range(pp.shape[1]):
    x, y = pp[:, i]
    if x in g_node and y in g_node:
        tmp.append([x, y])
data.gg_edge_index = torch.tensor(tmp, dtype=torch.int).reshape((2, -1))
data.n_gg_edge = data.gg_edge_index.shape[1]

with open('./datasets/protein-map.pkl', 'rb') as f:
    gmap = pickle.load(f)
gmap = {v:k for k,v in gmap.items()}
mmp = dict()
data.g_idx_to_id = dict()
for i, n in enumerate(g_node):
    mmp[n] = i
    data.g_idx_to_id[i] = gmap[n]

tmp = []
for i in range(data.gd_edge_index.shape[1]):
    data.gd_edge_index[0, i] = mmp[data.gd_edge_index[0, i].tolist()]

for i in range(data.gg_edge_index.shape[1]):
    data.gg_edge_index[0, i] = mmp[data.gg_edge_index[0, i].tolist()]
    data.gg_edge_index[1, i] = mmp[data.gg_edge_index[1, i].tolist()]

torch.save(data,  './datasets/pose-0.pt')

