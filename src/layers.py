from src.utils import *
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import torch.nn.functional as F


class myGCN(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 **kwargs):
        super(myGCN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = np.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.fill_(0)

        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class homoGraph(Module):

    def __init__(self, in_dim, nhid_list, requires_grad=True):
        super(homoGraph, self).__init__()
        self.out_dim = nhid_list[-1]

        self.embedding = torch.nn.Parameter(torch.Tensor(in_dim, nhid_list[0]))

        self.conv_list = torch.nn.ModuleList(
            [myGCN(nhid_list[i], nhid_list[i + 1], cached=True) for i in range(len(nhid_list) - 1)]
        )


        # TODO:
        self.embedding.requires_grad = requires_grad

        # TODO:

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.data.normal_()

    def forward(self, x, homo_edge_index, edge_weight):
        tmp = []

        x = self.embedding

        # TODO
        # x = normalize(x)
        # TODO

        tmp.append(x)

        for net in self.conv_list[:-1]:
            x = net(x, homo_edge_index, edge_weight)

            # TODO
            # x = normalize(x)
            x = F.relu(x, inplace=True)
            # TODO

            tmp.append(x)

        x = self.conv_list[-1](x, homo_edge_index, edge_weight)


        # TODO
        # x = normalize(x)
        x = F.relu(x, inplace=True)
        # TODO

        tmp.append(x)

        # TODO
        # print([torch.abs(a).detach().mean().tolist() for a in tmp])
        # self.tmp = tmp
        # [a.retain_grad() for a in self.tmp]
        # TODO

        # TODO
        return torch.cat(tmp, dim=1)
        # TODO


class interGraph(Module):

    def __init__(self, source_dim, target_dim, n_target, target_feat_dim=32):
        super(interGraph, self).__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.target_feat_dim = target_feat_dim
        self.n_target = n_target
        self.target_feat = torch.nn.Parameter(torch.Tensor(n_target, target_feat_dim))

        # TODO:
        self.target_feat.requires_grad = False
        # TODO:

        self.conv = myGCN(source_dim, target_dim, cached=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.target_feat.data.normal_()

    def forward(self, x, inter_edge_index, edge_weight=None):

        n_source = x.shape[0]
        tmp = inter_edge_index + 0
        tmp[1, :] += n_source

        x = torch.cat([x, torch.zeros((self.n_target, x.shape[1])).to(x.device)], dim=0)
        x = self.conv(x, tmp, edge_weight)[n_source:, :]
        x = F.relu(x)
        x = torch.cat([x, torch.abs(self.target_feat)], dim=1)
        return x