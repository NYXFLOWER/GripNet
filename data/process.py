from torch_geometric.data import Data
from data.utils import *
import pickle
import torch
import numpy as np
torch.manual_seed(1111)
np.random.seed(1111)



#
# # ###################################
# # MIP-pose-1 dataset
# # ###################################
# with open("data/tipexp_data.pkl", 'rb') as f:
#     data = pickle.load(f)
#
# dd_edge_index = torch.cat([data.train_idx, data.test_idx], dim=1)
# dd_edge_type = torch.cat([data.train_et, data.test_et])
#
# index = []
# ttt = []
# for i in range(len(data.train_range)):
#     print(i)
#     idx = torch.cat([data.train_idx[:, data.train_range[i][0]: data.train_range[i][1]],
#                      data.test_idx[:, data.test_range[i][0]: data.test_range[i][1]]], dim=1)
#     tt = torch.cat([data.train_et[data.train_range[i][0]: data.train_range[i][1]],
#                     data.test_et[data.test_range[i][0]: data.test_range[i][1]]])
#     index.append(idx)
#     ttt.append(tt)
#
# drug = Data.from_dict({
#     'n_node': data.n_drug,
#     'n_node_type': 1,
#     'n_edge': sum(data.n_edges_per_type),
#     'n_edge_type': data.n_et,
#     'node_type': ('drug', 'drug'),
#
#     'edge_index': index,
#     'edge_type': ttt,
#     'edge_weight': None
# })
# torch.save(drug, './data/pose-1/drug-drug.pt')
#
# gene = Data.from_dict({
#     'n_node': data.n_prot,
#     'n_node_type': 1,
#     'n_edge': data.pp_index.shape[1],
#     'n_edge_type': 1,
#     'node_type': ('gene', 'gene'),
#
#     'edge_index': data.pp_index,
#     'edge_type': None,
#     'edge_weight': None
# })
# torch.save(gene, './data/pose-1/gene-gene.pt')
#
# gene_drug = Data.from_dict({
#     'n_node': data.n_prot + data.n_drug,
#     'n_node_type': 2,
#     'n_edge': data.pd_index.shape[1],
#     'n_edge_type': 1,
#     'node_type': ('gene', 'drug'),
#
#     'edge_index': data.pd_index,
#     'edge_type': None,
#     'edge_weight': None
# })
# torch.save(gene_drug, './data/pose-1/gene-drug.pt')
#
# idx_to_id_dict = {
#     'drug': data.drug_idx_to_id,
#     'prot': data.prot_idx_to_id,
#     'side_effect': data.side_effect_idx_to_name
# }
#
# with open('./data/pose-1/pose_map.pkl', 'wb') as f:
#     pickle.dump(idx_to_id_dict, f)
#
#
# # ###################################
# # comb-pose-1 dataset
# # ###################################
# # load data
# dd = torch.load('data/pose-1/drug-drug.pt')
# gd = torch.load('data/pose-1/gene-drug.pt')
# gg = torch.load('data/pose-1/gene-gene.pt')
#
# # add dd graph
# index = dd.edge_index.copy()
# ttt = dd.edge_type.copy()
#
# # add gg graph
# tmp = gg.edge_index + dd.n_node
# index.append(tmp)
# tmp = torch.tensor([len(index)] * gg.n_edge)
# ttt.append(tmp)
#
# # add gd graph
# tmp = gd.edge_index
# tmp[0] = tmp[0] + dd.n_node
# index.append(tmp)
# tmp = torch.tensor([len(index)] * gd.n_edge)
# ttt.append(tmp)
#
# # train and test set
# values = process_edge_multirelational(index, p=0.9)
#
# # construction
# data = Data.from_dict({
#     'n_node': dd.n_node + gg.n_node,
#     'n_drug': dd.n_node,
#     'n_gene': gg.n_node,
#     'n_node_typed': 2,
#     'n_edge': dd.n_edge + gd.n_edge + gg.n_edge,
#     'n_edge_type': dd.n_edge_type + gd.n_edge_type + gg.n_edge_type,
#     'node_type': [('drug', 'drug'), ('gene', 'drug'), ('gene', 'gene')],
#
#     'edge_index': index,
#     'edge_type': ttt,
#     'edge_weight': None,
#
#     'train_idx': values[0],
#     'train_et': values[1],
#     'train_range': values[2],
#     'test_idx': values[3],
#     'test_et': values[4],
#     'test_range': values[5],
#
#     'description': 'reindex node [drug]+[gene]'
# })
#
# tmp = data.train_idx[:, data.train_range[-1][0]:data.train_range[-1][1]].shape[1]
# tmp = -int(tmp/2)
# data.train_idx = data.train_idx[:, :tmp]
# data.train_et = data.train_et[:tmp]
# data.train_range[-1][1] += tmp
#
# tmp = data.test_idx[:, data.test_range[-1][0]:data.test_range[-1][1]].shape[1]
# tmp = -int(tmp/2)
# data.test_idx = data.test_idx[:, :tmp]
# data.test_et = data.test_et[:tmp]
# data.test_range[-1][1] += tmp
#
# torch.save(data, './data/pose-1-1-combl.pt')
#
#
# # ###################################
# # citation network dataset
# # ###################################
# aa = torch.load("data/citation_network/author-author.pt")
# pa = torch.load("data/citation_network/paper-author.pt")
# pp = torch.load("data/citation_network/paper-paper.pt")
#
# # tmp = aa.node_label
# # label = [[] for i in range(aa.n_node_type)]
# # for idx, lab in tmp:
# #     label[lab].append(idx)
# # label = [torch.tensor(i, dtype=torch.int) for i in label]
# # aa.typed_node_label = label
# # torch.save(aa, "data/citation_network/author-author.pt")
#
# data = Data()
# data.train_idx, data.train_label, data.train_range, data.test_idx, data.test_label, data.test_range = process_node_multilabel(aa.typed_node_label)
#
# data.aa_edge_index, data.pp_edge_index, data.pa_edge_index = aa.edge_index, pp.edge_index, pa.edge_index
# data.a_node_label = aa.typed_node_label
# data.n_a_node = len(set(aa.edge_index.unique().tolist()) | set(pa.edge_index[1].unique().tolist()))
# data.n_a_node_type = aa.n_node_type
# data.n_aa_edge = aa.n_edge
# data.n_aa_edge_type = 1
#
# data.n_p_node = pp.n_node
#
# un = pa.edge_index[1].unique()
# tmp = []
# for i in range(aa.edge_index.shape[1]):
#     x, y = aa.edge_index[:, i].tolist()
#     if x in un and y in un:
#         tmp.append([x, y])
#
# tmp = np.array(tmp).reshape((2, -1))
# data.aa_edge_index = torch.tensor(tmp)
#
# no_connect_a = set(data.pa_edge_index[1].unique().tolist()) - set(data.aa_edge_index.unique().tolist())
# data.a_no_connect = no_connect_a
#
# no_connect_p = set(data.pp_edge_index.unique().tolist()) - set(data.pa_edge_index[0].unique().tolist())
# data.p_no_connect = no_connect_p
#
# data.n_p_node = 46704
# data.n_pp_edge = pp.edge_index.shape[1]
# data.n_pa_edge = pa.edge_index.shape[1]
#
# torch.save(data, "./data/auta-0.pt")
#
#
# # ###################################
# # pose-1-0
# # ###################################
# data = torch.load("./datasets/pose-1-2.pt")
# g_node = set(data.gd_edge_index[0].unique().tolist())
# data.n_g_node = len(g_node)
#
# pp = data.gg_edge_index.numpy()
# tmp = []
# for i in range(pp.shape[1]):
#     x, y = pp[:, i]
#     if x in g_node and y in g_node:
#         tmp.append([x, y])
# data.gg_edge_index = torch.tensor(tmp, dtype=torch.int).reshape((2, -1))
# data.n_gg_edge = data.gg_edge_index.shape[1]
#
# with open('./datasets/protein-map.pkl', 'rb') as f:
#     gmap = pickle.load(f)
# gmap = {v:k for k,v in gmap.items()}
# mmp = dict()
# data.g_idx_to_id = dict()
# for i, n in enumerate(g_node):
#     mmp[n] = i
#     data.g_idx_to_id[i] = gmap[n]
#
# tmp = []
# for i in range(data.gd_edge_index.shape[1]):
#     data.gd_edge_index[0, i] = mmp[data.gd_edge_index[0, i].tolist()]
#
# for i in range(data.gg_edge_index.shape[1]):
#     data.gg_edge_index[0, i] = mmp[data.gg_edge_index[0, i].tolist()]
#     data.gg_edge_index[1, i] = mmp[data.gg_edge_index[1, i].tolist()]
#
# torch.save(data,  './datasets/pose-1-0.pt')


# ###################################
# comb-pose-1 dataset
# ###################################
# ddd = 2
# # load data
# data = torch.load('./datasets/pose-{}.pt'.format(ddd))
#
# # add dd graph
# index = data.dd_edge_index.copy()
# ttt = data.dd_edge_type.copy()
#
# # add gg graph
# tmp = data.gg_edge_index + data.n_d_node
# index.append(tmp)
# tmp = torch.tensor([len(index)] * data.n_gg_edge, dtype=torch.long)
# ttt.append(tmp)
#
# # add gd graph
# tmp = data.gd_edge_index
# tmp[0] = tmp[0] + data.n_d_node
# index.append(tmp)
# tmp = torch.tensor([len(index)] * data.n_gd_edge, dtype=torch.long)
# ttt.append(tmp)

# # train and test set
# index[-2] = torch.tensor(index[-2], dtype=torch.long)
# index[-1] = torch.tensor(index[-1], dtype=torch.long)
# data.train_idx, data.train_et, data.train_range, data.test_idx, data.test_et, data.test_range = process_edge_multirelational(index, p=0.9)
#
# # construction
# data.n_node = data.n_d_node + data.n_g_node
# data.n_drug = data.n_d_node
# data.n_gene = data.n_g_node
# data.n_node_typed = 2
# data.n_edge = data.n_dd_edge + data.n_gg_edge+ data.n_gd_edge
# data.n_edge_type = data.n_dd_edge_type + 2
# data.node_type = [('drug', 'drug'), ('gene', 'drug'), ('gene', 'gene')]
# data.edge_index = index
# data.edge_type = ttt
#
# tmp = data.train_idx[:, data.train_range[-1][0]:data.train_range[-1][1]].shape[1]
# tmp = -int(tmp/2)
# data.train_idx = data.train_idx[:, :tmp]
# data.train_et = data.train_et[:tmp]
# data.train_range[-1][1] += tmp
#
# tmp = data.test_idx[:, data.test_range[-1][0]:data.test_range[-1][1]].shape[1]
# tmp = -int(tmp/2)
# data.test_idx = data.test_idx[:, :tmp]
# data.test_et = data.test_et[:tmp]
# data.test_range[-1][1] += tmp
#
# torch.save(data, './datasets/pose-{}-combl.pt'.format(ddd))

def pkl_load(path):
    with open(path, 'rb') as f:
        out = pickle.load(f)
    return out

def pkl_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj,  f)
    return
#
# # ###################################
# # Auta dataset
# # ###################################
# aa = torch.load('data/citation_network_2/author-author.pt')
# pa = torch.load('data/citation_network_2/paper-author-all.pt')
# pp = torch.load('data/citation_network_2/paper-paper-all.pt')
#
# pmap = pkl_load('data/citation_network_2/paper_all_map.pkl')
# amap = pkl_load('data/citation_network_2/author_map.pkl')
# pmap = {v: k for k, v in pmap.items()}
# amap = {v: k for k, v in amap.items()}
#
# # reindex author and paper
# ppmap, aamap, p_old_to_new, a_ole_to_new = dict(), dict(), dict(), dict()
#
# # a1, a2 = [set(aa.edge_index[i].unique().tolist()) for i in (0, 1)]
# linked_author = set(aa.edge_index.unique().tolist())
# unlinked_author = set(range(aa.n_node)) - linked_author
# for i, a in enumerate(linked_author):
#     aamap[i] = amap[a]
#     a_ole_to_new[a] = i
# pre = linked_author.__len__()
# for i, a in enumerate(unlinked_author):
#     aamap[pre+i] = amap[a]
#     a_ole_to_new[a] = pre+i
#
# pkl_save(aamap, 'datasets-auta/map-author.pkl')
#
# paper_with_linked_author = set([p for p, a in pa.edge_index.reshape((-1, 2)).tolist() if a in linked_author])
# paper_with_author = set(pa.edge_index[0].unique().tolist())
# paper_without_author = set(range(pp.n_node)) - paper_with_author
# for i, p in enumerate(paper_with_author):
#     ppmap[i] = pmap[p]
#     p_old_to_new[p] = i
# pre = len(paper_with_author)
# for i, p in enumerate(paper_without_author):
#     ppmap[pre+i] = pmap[p]
#     p_old_to_new[p] = pre+i
#
# pkl_save(ppmap, 'datasets-auta/map-paper.pkl')
#
# id = [[a_ole_to_new[i], a_ole_to_new[j]] for i, j in aa.edge_index.reshape((-1, 2)).tolist() if i != j]
# idd = [[i, j] for i, j in aa.edge_index.reshape((-1, 2)).tolist() if i != j]
# un_id = remove_bidirection(torch.tensor(id).reshape((2, -1)), None)
# aa.edge_index = to_bidirection(un_id)
#
# node_label_list = [set([]) for i in range(7)]
# for n, l in aa.node_label:
#     node_label_list[l].add(a_ole_to_new[n])
# node_label_list = [torch.tensor(list(i)) for i in node_label_list]
#
# aa.node_label_list = node_label_list
# aa.n_edge = aa.edge_index.shape[1]
#
# id = [[p_old_to_new[i], p_old_to_new[j]] for i, j in pp.edge_index.reshape((-1, 2)).tolist() if i != j]
# un_id = remove_bidirection(torch.tensor(id).reshape((2, -1)), None)
# pp.edge_index = to_bidirection(un_id)
#
# id = [p_old_to_new[i] for i in pa.edge_index[0].tolist()]
# id2  = [a_ole_to_new[i] for i in pa.edge_index[1].tolist()]
#
# pa.edge_index = torch.tensor([id, id2])
#
# aa.n_a_linked = len(linked_author)
# pp.n_p_linked = len(paper_with_author)
#
# pp.n_edge = pp.edge_index.shape[1]
#
# torch.save(aa, 'datasets-auta/aa.pt')
# torch.save(pa, 'datasets-auta/pa.pt')
# torch.save(pp, 'datasets-auta/pp.pt')

aa = torch.load('datasets-auta/aa.pt')
pa = torch.load('datasets-auta/pa.pt')
pp = torch.load('datasets-auta/pp.pt')

# auta-2 -- all the data
data = Data()
data.aa_edge_idx = aa.edge_index
data.pa_edge_idx = pa.edge_index
data.pp_edge_idx = pp.edge_index
data.n_a_node = aa.n_node
data.n_p_node = pp.n_node
data.n_aa_edge = aa.n_edge
data.n_pa_edge = pa.n_edge
data.n_pp_edge = pp.n_edge
data.train_node_idx, data.train_node_class, data.train_range, data.test_node_idx, data.test_node_class, data.test_range = process_node_multilabel(aa.node_label_list)
data.n_a_type = 7

torch.save(data, 'datasets-auta/auta-2.pt')


# auta-0 -- no unlinked paper
p_linked = set(range(pp.n_p_linked))
p1, p2 = [], []
for i, j in pp.edge_index.reshape((-1, 2)).tolist():
    if i in p_linked and j in p_linked:
        p1.append(i)
        p2.append(j)
data.pp_edge_idx = torch.tensor([p1, p2])
data.n_pp_edge = data.pp_edge_idx.shape[1]
data.n_p_node = pp.n_p_linked
torch.save(data, 'datasets-auta/auta-0.pt')

# auta-1 -- no unlinked paper
data = torch.load('./datasets-auta/auta-2.pt')
a_linked = set(range(aa.n_a_linked))
a1, a2 = [], []
for i, j in aa.edge_index.reshape((-1, 2)).tolist():
    if i in a_linked or j in a_linked:
        a1.append(i)
        a2.append(j)
data.aa_edge_idx = torch.tensor([a1, a2])
data.n_aa_edge = data.aa_edge_idx.shape[1]
data.n_a_node = aa.n_a_linked

node_label_list = [i[i < aa.n_a_linked] for i in aa.node_label_list]
data.train_node_idx, data.train_node_class, data.train_range, data.test_node_idx, data.test_node_class, data.test_range = process_node_multilabel(node_label_list)

p, a = [],  []
for i, j in pa.edge_index.reshape((-1, 2)).tolist():
    if j in a_linked:
        p.append(i)
        a.append(j)
data.pa_edge_idx = torch.tensor([p, a])
data.n_pa_edge = data.pa_edge_idx.shape[1]
torch.save(data, 'datasets-auta/auta-1.pt')