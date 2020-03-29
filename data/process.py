import pickle
import torch

with open("data/tipexp_data.pkl", 'rb') as f:
    data = pickle.load(f)

from torch_geometric.data import Data

drug = Data.from_dict({
    'n_node': data.n_drug,
    'n_node_type': 1,
    'n_edge': sum(data.n_edges_per_type),
    'n_edge_type': data.n_et,
    'node_type': ('drug', 'drug'),

    'edge_index': data.dd_edge_index,
    'edge_type': data.dd_edge_type,
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

