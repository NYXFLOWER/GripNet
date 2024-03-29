import torch
import numpy as np
import pickle
from sklearn import metrics
from sklearn.metrics import accuracy_score


torch.manual_seed(1111)
np.random.seed(1111)
EPS = 1e-13


def normalize(input):
    norm_square = (input ** 2).sum(dim=1)
    return input / torch.sqrt(norm_square.view(-1, 1))


def sparse_id(n):
    idx = [[i for i in range(n)], [i for i in range(n)]]
    val = [1 for i in range(n)]
    i = torch.LongTensor(idx)
    v = torch.FloatTensor(val)
    shape = (n, n)

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def auprc_auroc_ap(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    auroc, ap = metrics.roc_auc_score(y, pred), metrics.average_precision_score(y, pred)
    y, xx, _ = metrics.precision_recall_curve(y, pred)
    auprc = metrics.auc(xx, y)

    return auprc, auroc, ap


def micro_macro(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    micro, macro = (
        metrics.f1_score(y, pred, average="micro"),
        metrics.f1_score(y, pred, average="macro"),
    )

    return micro, macro


def acc(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    return accuracy_score(y, pred)


def load_graph(pt_file_path="./sample_graph.pt"):
    """
    Parameters
    ----------
    pt_file_path : file path

    Returns
    -------
    graph : torch_geometric.data.Data
        - data.n_node: number of nodes
        - data.n_node_type: number of node types == (1 or 2)
        - data.n_edge: number of edges
        - data.n_edge_type: number of edge types
        - data.node_type: (source_node_type, target_node_type)

        - data.edge_index: [list of] torch.Tensor, int, shape (2, n_edge), [indexed by edge type]
            [0, :] : source node index
            [1, :] : target node index
        - data.edge_type: None or list of torch.Tensor, int, shape (n_edge,), indexed by edge type
        - data.edge_weight: None or list of torch.Tensor, float, shape (n_edge,)

        - data.source_node_idx_to_id: dict {idx : id}
        - data.target_node_idx_to_id: dict {idx : id}
    """

    return torch.load(pt_file_path)


def load_node_idx_to_id_dict(pkl_file_path="./data/pose-1/map.pkl"):
    """
    Parameters:
    -----------
    The path of index maps in the dataset directory

    Returns:
    --------
    a dictionary of map from node index to entity id/name
    """
    with open(pkl_file_path, "rb") as f:
        out = pickle.load(f)
    return out


def negative_sampling(pos_edge_index, num_nodes):
    idx = pos_edge_index[0] * num_nodes + pos_edge_index[1]
    idx = idx.to(torch.device("cpu"))

    perm = torch.tensor(np.random.choice(num_nodes ** 2, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(np.random.choice(num_nodes ** 2, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp
        rest = mask.nonzero().view(-1)

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0).long().to(pos_edge_index.device)


def typed_negative_sampling(pos_edge_index, num_nodes, range_list):
    tmp = []
    for start, end in range_list:
        tmp.append(negative_sampling(pos_edge_index[:, start:end], num_nodes))
    return torch.cat(tmp, dim=1)


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

        train_label_list.append(
            torch.tensor([i] * train_indices.shape[0], dtype=torch.long)
        )
        test_label_list.append(
            torch.tensor([i] * test_indices.shape[0], dtype=torch.long)
        )

    train_range = get_range_list(train_list, is_node=True)
    test_range = get_range_list(test_list, is_node=True)

    train_node_idx = torch.cat(train_list)
    test_node_idx = torch.cat(test_list)

    train_node_class = torch.cat(train_label_list)
    test_node_class = torch.cat(test_label_list)

    return (
        train_node_idx,
        train_node_class,
        train_range,
        test_node_idx,
        test_node_class,
        test_range,
    )


def process_data_multiclass(torch_tensor, n_class):
    node_idx, sample, range1 = [], [], [0]
    for i in range(n_class):
        idx = torch_tensor[0][torch_tensor[1] == i]
        node_idx.append(idx)
        sample.append(idx.shape[0])
        range1.append(idx.shape[0] + range1[i])
    return (
        torch.cat(node_idx),
        torch.cat(
            [torch.tensor([i] * sample[i], dtype=torch.int64) for i in range(n_class)]
        ),
        [[range1[i], range1[i + 1]] for i in range(n_class)],
    )
