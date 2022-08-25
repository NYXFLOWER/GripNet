from torch.nn import Module, Parameter
import torch
import numpy as np


torch.manual_seed(1111)
np.random.seed(1111)


class multiRelaInnerProductDecoder(Module):
    def __init__(self, in_dim, num_et):
        super(multiRelaInnerProductDecoder, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Parameter(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, z, edge_index, edge_type, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]] * self.weight[edge_type]).sum(
            dim=1
        )
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))


class multiClassInnerProductDecoder(Module):
    def __init__(self, in_dim, num_class):
        super(multiClassInnerProductDecoder, self).__init__()
        self.num_class = num_class
        self.in_dim = in_dim
        self.weight = Parameter(torch.Tensor(self.in_dim, self.num_class))

        self.reset_parameters()

    def forward(self, z, node_list, softmax=True):
        # value = (z[node_list] * self.weight[node_label]).sum(dim=1)
        # value = torch.sigmoid(value) if sigmoid else value

        pred = torch.matmul(z[node_list], self.weight)
        pred = torch.softmax(pred, dim=1) if softmax else pred

        return pred

    def reset_parameters(self):
        stdv = np.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        # self.weight.data.normal_()
