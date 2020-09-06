from sklearn import metrics
from torch.nn import Parameter, Module
import torch
import numpy as np


def micro_macro(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    micro, macro = metrics.f1_score(y, pred, average='micro'), metrics.f1_score(y, pred, average='macro')
    return micro, macro


class multiClassInnerProductDecoder(Module):
    def __init__(self, in_dim, num_class):
        super(multiClassInnerProductDecoder, self).__init__()
        self.num_class = num_class
        self.in_dim = in_dim
        self.weight = Parameter(torch.Tensor(self.in_dim, self.num_class))

        self.reset_parameters()

    def forward(self, z, softmax=True):
        # value = (z[node_list] * self.weight[node_label]).sum(dim=1)
        # value = torch.sigmoid(value) if sigmoid else value

        pred = torch.matmul(z, self.weight)
        pred = torch.log_softmax(pred, dim=1) if softmax else pred

        return pred

    def reset_parameters(self):
        stdv = np.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        # self.weight.data.normal_()

def sparse_id(n):
    idx = [[i for i in range(n)], [i for i in range(n)]]
    val = [1 for i in range(n)]
    i = torch.LongTensor(idx)
    v = torch.FloatTensor(val)
    shape = (n, n)
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
