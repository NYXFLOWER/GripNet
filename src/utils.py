import torch


def load_graph(pt_file_path='./sample_graph.pt'):
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
        - data.edge_index: torch.Tensor, int, shape (2, n_edge)
            [0, :] : source node index
            [1, :] : target node index
        - data.edge_type: torch.Tensor, int, shape (n_edge,)
        - data.edge_weight: torch.Tensor, float, shape (n_edge,)

        - data.source_node_idx_to_id: dict {idx : id}
        - data.target_node_idx_to_id: dict {idx : id}
    """

    return torch.load(pt_file_path)