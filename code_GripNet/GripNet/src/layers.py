import torch.nn.functional as F

from pytorch_memlab import profile
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add

from src.utils import *


torch.manual_seed(1111)
np.random.seed(1111)


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
            edge_weight = torch.ones((edge_index.size(1),),
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


class myRGCN(MessagePassing):
    r"""
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 num_bases,
                 after_relu,
                 bias=False,
                 **kwargs):
        super(myRGCN, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.after_relu = after_relu

        self.basis = Parameter(
            torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Parameter(torch.Tensor(num_relations, num_bases))
        self.root = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        self.att.data.normal_(std=1 / np.sqrt(self.num_bases))

        if self.after_relu:
            self.root.data.normal_(std=2 / self.in_channels)
            self.basis.data.normal_(std=2 / self.in_channels)

        else:
            self.root.data.normal_(std=1 / np.sqrt(self.in_channels))
            self.basis.data.normal_(std=1 / np.sqrt(self.in_channels))

        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, edge_index, edge_type, range_list):
        """"""
        return self.propagate(
            edge_index, x=x, edge_type=edge_type, range_list=range_list)

    def message(self, x_j, edge_index, edge_type, range_list):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        # w = w[edge_type, :, :]
        # out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        out_list = []
        for et in range(range_list.shape[0]):
            start, end = range_list[et]

            tmp = torch.matmul(x_j[start: end, :], w[et])

            # xxx = x_j[start: end, :]
            # tmp = checkpoint(torch.matmul, xxx, w[et])

            out_list.append(tmp)

        # TODO: test this
        return torch.cat(out_list)

    def update(self, aggr_out, x):

        out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)


class homoGraph(Module):

    def __init__(self, nhid_list, requires_grad=True, start_graph=False,
                 in_dim=None, multi_relational=False, n_rela=None, n_base=32):
        super(homoGraph, self).__init__()
        self.multi_relational = multi_relational
        self.start_graph = start_graph
        self.out_dim = nhid_list[-1]
        self.n_cov = len(nhid_list) - 1

        if start_graph:
            self.embedding = torch.nn.Parameter(torch.Tensor(in_dim, nhid_list[0]))
            self.embedding.requires_grad = requires_grad
            self.reset_parameters()

        if multi_relational:
            assert n_rela is not None
            after_relu = [False if i == 0 else True for i in
                          range(len(nhid_list) - 1)]
            self.conv_list = torch.nn.ModuleList([
                myRGCN(nhid_list[i], nhid_list[i + 1], n_rela, n_base, after_relu[i])
                for i in range(len(nhid_list) - 1)])
        else:
            self.conv_list = torch.nn.ModuleList([
                myGCN(nhid_list[i], nhid_list[i + 1], cached=True)
                for i in range(len(nhid_list) - 1)])


    def reset_parameters(self):
        self.embedding.data.normal_()

    def forward(self, x, homo_edge_index, edge_weight=None, edge_type=None,
                range_list=None, if_catout=False):
        if self.start_graph:
            x = self.embedding

        if if_catout:
            tmp = []
            tmp.append(x)

        if self.multi_relational:
            assert edge_type is not None
            assert range_list is not None

        # ---- start: no check point version ----
        for net in self.conv_list[:-1]:
            x = net(x, homo_edge_index, edge_type, range_list) \
                if self.multi_relational \
                else net(x, homo_edge_index, edge_weight)
            x = F.relu(x, inplace=True)
            if if_catout:
                tmp.append(x)

        x = self.conv_list[-1](x, homo_edge_index, edge_type, range_list) \
            if self.multi_relational \
            else self.conv_list[-1](x, homo_edge_index, edge_weight)
        # ---- end: no check point version ----

        # if self.multi_relational:
        #     for net in self.conv_list[:-1]:
        #         x = checkpoint(net, x, homo_edge_index, edge_type, range_list)
        #         x = F.relu(x, inplace=True)
        #         tmp.append(x)
        #     x = checkpoint(self.conv_list[-1], x, homo_edge_index, edge_type, range_list)
        # else:
        #     for net in self.conv_list[:-1]:
        #         x = net(x, homo_edge_index, edge_weight)
        #         x = F.relu(x, inplace=True)
        #         tmp.append(x)
        #     x = self.conv_list[-1](x, homo_edge_index, edge_weight)

        # TODO
        # x = normalize(x)
        x = F.relu(x, inplace=True)
        # TODO
        if if_catout:
            tmp.append(x)
            x = torch.cat(tmp, dim=1)

        # TODO
        # print([torch.abs(a).detach().mean().tolist() for a in tmp])
        # self.tmp = tmp
        # [a.retain_grad() for a in self.tmp]
        # TODO

        # TODO
        return x
        # TODO


class interGraph(Module):

    def __init__(self, source_dim, target_dim, n_target, target_feat_dim=32,
                 requires_grad=True):
        super(interGraph, self).__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.target_feat_dim = target_feat_dim
        self.n_target = n_target
        self.target_feat = torch.nn.Parameter(
            torch.Tensor(n_target, target_feat_dim))

        # TODO:
        self.target_feat.requires_grad = requires_grad

        if target_dim != target_feat_dim:
            self.target_feat_down = torch.nn.Parameter(torch.Tensor(self.target_feat_dim, target_dim))
            self.target_feat_down.requires_grad = requires_grad
            self.target_feat_down.data.normal_()
        # TODO:

        self.conv = myGCN(source_dim, target_dim, cached=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.target_feat.data.normal_()

    def forward(self, x, inter_edge_index, edge_weight=None, if_relu=True, mod='cat'):
        n_source = x.shape[0]
        tmp = inter_edge_index + 0
        tmp[1, :] += n_source

        x = torch.cat(
            [x, torch.zeros((self.n_target, x.shape[1])).to(x.device)], dim=0)
        x = self.conv(x, tmp, edge_weight)[n_source:, :]
        if if_relu:
            x = F.relu(x)
        if mod == 'cat':
            x = torch.cat([x, torch.abs(self.target_feat)], dim=1)
        else:
            if x.shape[1] == self.target_feat.shape[1]:
                x = (x + torch.abs(self.target_feat)) / 2
            else:

                x = (x + F.relu(torch.matmul(self.target_feat, self.target_feat_down))) / 2
        # x = torch.cat([x, F.relu(self.target_feat)], dim=1)

        return x
