import pickle
import torch

with open("data/tipexp_data.pkl", 'rb') as f:
    data = pickle.load(f)

from torch_geometric.data import Data

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

