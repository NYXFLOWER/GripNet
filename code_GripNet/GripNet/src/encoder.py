from src.utils import *
from src.layers import myRGCN
from src.decoder import multiRelaInnerProductDecoder


class RGCN(Module):
    def __init__(self, feat_dim, r1_in_dim, r1_out_dim, r2_out_dim, n_relations, n_bases):
        super(RGCN, self).__init__()
        self.embedding = Parameter(torch.Tensor(feat_dim, r1_in_dim))
        self.embedding.data.normal_()
        self.rgcn1 = myRGCN(r1_in_dim, r1_out_dim, n_relations, n_bases, after_relu=False)
        self.rgcn2 = myRGCN(r1_out_dim, r2_out_dim, n_relations, n_bases, after_relu=True)

    def forward(self, x, edge_index, edge_et, edge_range):
        x = torch.matmul(x, self.embed)
        x = self.rgcn1(x, edge_index, edge_et, edge_range)
        x = self.rgcn2(x, edge_index, edge_et, edge_range)

        return x
