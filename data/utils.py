import torch
import numpy as np


def remove_bidirection(edge_index, edge_type=None):
    mask = edge_index[0] > edge_index[1]
    keep_set = mask.nonzero().view(-1)

    if edge_type is None:
        return edge_index[:, keep_set]
    else:
        return edge_index[:, keep_set], edge_type[keep_set]


def to_bidirection(edge_index, edge_type=None):
    tmp = edge_index.clone()
    tmp[0, :], tmp[1, :] = edge_index[1, :], edge_index[0, :]
    if edge_type is None:
        return torch.cat([edge_index, tmp], dim=1)
    else:
        return torch.cat([edge_index, tmp], dim=1), torch.cat([edge_type, edge_type])


def get_range_list(edge_list, is_node=False):
    idx = 0 if is_node else 1
    tmp = []
    s = 0
    for i in edge_list:
        tmp.append((s, s + i.shape[idx]))
        s += i.shape[idx]
    return torch.tensor(tmp)


def process_edge(raw_edges):
    indices = remove_bidirection(raw_edges, None)
    n_edge = indices.shape[1]

    rd = np.random.binomial(1, 0.9, n_edge)
    train_mask = rd.nonzero()[0]
    test_mask = (1 - rd).nonzero()[0]

    train_indices = indices[:, train_mask]
    train_indices = to_bidirection(train_indices, None)

    test_indices = indices[:, test_mask]
    test_indices = to_bidirection(test_indices, None)

    return train_indices, test_indices


def process_edge_multirelational(raw_edge_list, p=0.9):
    train_list = []
    test_list = []
    train_label_list = []
    test_label_list = []

    for i, idx in enumerate(raw_edge_list):
        train_mask = np.random.binomial(1, p, idx.shape[1])
        test_mask = 1 - train_mask
        train_set = train_mask.nonzero()[0]
        test_set = test_mask.nonzero()[0]

        train_list.append(idx[:, train_set])
        test_list.append(idx[:, test_set])

        train_label_list.append(torch.ones(2 * train_set.size, dtype=torch.long) * i)
        test_label_list.append(torch.ones(2 * test_set.size, dtype=torch.long) * i)

    train_list = [to_bidirection(idx) for idx in train_list]
    test_list = [to_bidirection(idx) for idx in test_list]

    train_range = get_range_list(train_list)
    test_range = get_range_list(test_list)

    train_edge_idx = torch.cat(train_list, dim=1)
    test_edge_idx = torch.cat(test_list, dim=1)

    train_et = torch.cat(train_label_list)
    test_et = torch.cat(test_label_list)

    return train_edge_idx, train_et, train_range, test_edge_idx, test_et, test_range


def process_node(raw_nodes, p=0.9):
    rd = np.random.binomial(1, 0.9, len(raw_nodes))
    train_mask = rd.nonzero()[0]
    test_mask = (1 - rd).nonzero()[0]

    train_indices = raw_nodes[train_mask]
    test_indices = raw_nodes[test_mask]

    return train_indices, test_indices


def process_node_multilabel(raw_nodes_list):
    train_list = []
    test_list = []
    train_label_list = []
    test_label_list = []

    for i, idx in enumerate(raw_nodes_list):
        train_indices, test_indices = process_node(idx)

        train_list.append(train_indices)
        test_list.append(test_indices)

        train_label_list.append(torch.tensor([i]*train_indices.shape[0], dtype=torch.long))
        test_label_list.append(torch.tensor([i]*test_indices.shape[0], dtype=torch.long))

    train_range = get_range_list(train_list, is_node=True)
    test_range = get_range_list(test_list, is_node=True)

    train_node_idx = torch.cat(train_list)
    test_node_idx = torch.cat(test_list)

    train_node_class = torch.cat(train_label_list)
    test_node_class = torch.cat(test_label_list)

    return train_node_idx, train_node_class, train_range, test_node_idx, test_node_class, test_range


def process_data_multiclass(torch_tensor, n_class):
    node_idx, node_class, sample, range1 = [], [], [], [0]
    for i in range(n_class):
        idx = torch_tensor[0][torch_tensor[1] == i]
        node_idx.append(idx)
        sample.append(idx.shape[0])
        range1.append(idx.shape[0]+range1[i])
    return torch.cat(node_idx), \
           torch.cat([torch.tensor([i]*sample[i], dtype=torch.int64) for i in range(n_class)]), \
           [[range1[i], range1[i+1]] for i in range(n_class)]