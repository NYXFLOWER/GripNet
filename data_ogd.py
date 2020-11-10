import torch

data = torch.load("datasets-pose/pose-0-combl.pt")

split_edge = {}
head = ['drug']*data.train_range[-2][0] + ['gene']*(data.train_range[-1][1]-data.train_range[-2][0])

split_edge['train'] = {'head_type': ['drug']*data.train_range[-2][0] +
                                    ['gene']*(data.train_range[-1][1]-data.train_range[-2][0]),
                       'head': data.train_idx[0].numpy(),
                       'relation': data.train_et.numpy(),
                       'tail_type': ['drug']*data.train_range[-2][0] +
                                    ['gene']*(data.train_range[-1][0]-data.train_range[-2][0]) +
                                    ['drug']*(data.train_range[-1][1]-data.train_range[-1][0]),
                       'tail': data.train_idx[1].numpy()}

split_edge['test'] = {'head_type': ['drug']*data.test_range[-2][0] +
                                   ['gene']*(data.test_range[-1][1]-data.test_range[-2][0]),
                       'head': data.test_idx[0].numpy(),
                       'relation': data.test_et.numpy(),
                       'tail_type': ['drug']*data.test_range[-2][0] +
                                    ['gene']*(data.test_range[-1][0]-data.test_range[-2][0]) +
                                    ['drug']*(data.test_range[-1][1]-data.test_range[-1][0]),
                       'tail': data.test_idx[1].numpy()}

split_edge['valid'] = split_edge['test']


split_edge['nentity'] = data.n_drug+data.n_gene
split_edge['entity_dict'] = {'drug': (0, data.n_drug), 'gene': (data.n_drug, split_edge['nentity'])}

import pickle
with open('data_ogd.pkl', 'wb') as f:
    pickle.dump(split_edge, f)


