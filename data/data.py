import pandas as pd
import torch
from torch_geometric.data import Data

bb = torch.from_numpy(pd.read_csv('data/freebase/book_lo_business/index_separated/link_book-book_1.dat', usecols=[0, 1], sep='\t', header=None).to_numpy().T)
bp = torch.from_numpy(pd.read_csv('data/freebase/book_lo_business/index_separated/link_book-business_4.dat', usecols=[0, 1], sep='\t', header=None).to_numpy().T)
pp = torch.from_numpy(pd.read_csv('data/freebase/book_lo_business/index_separated/link_business-business_5.dat', usecols=[0, 1], sep='\t', header=None).to_numpy().T)
bq = torch.from_numpy(pd.read_csv('data/freebase/book_lo_business/index_separated/link_book-location&organization_2.dat', usecols=[0, 1, 2], sep='\t', header=None).to_numpy().T)
qq = torch.from_numpy(pd.read_csv('data/freebase/book_lo_business/index_separated/link_location&organization_3.dat', usecols=[0, 1, 2], sep='\t', header=None).to_numpy().T)


# train = torch.from_numpy(pd.read_csv('data/freebase_org/book_{}/index_separated/label.dat.train'.format(l[n]), sep='\t', header=None).to_numpy().T)
# test = torch.from_numpy(pd.read_csv('data/freebase/book_{}/index_separated/label.dat.test'.format(l[n]), sep='\t', header=None).to_numpy().T)


data = Data()
data.aa_edge_idx = bb
data.pa_edge_idx = torch.tensor([bp[1].tolist(), bp[0].tolist()])
data.pp_edge_idx = pp

data.n_a_node = (max(bb.max(), bp[0].max()) + 1).tolist()
data.n_p_node = (max(pp.max(), bp[1].max()) + 1).tolist()

data.n_a_type = 8

data.n_aa_edge = bb.shape[1]
data.n_pp_edge = pp.shape[1]
data.n_pa_edge = bp.shape[1]

torch.save(data, 'datasets-freebase/business.pt')


data.qa_edge_idx = torch.tensor([bq[1].tolist(), bq[0].tolist()])
data.qq_edge_idx = qq
data.n_q_node = (max(qq.max(), bq[1].max()) + 1).tolist()
data.n_qq_edge = qq.shape[1]
data.n_qa_edge = bq.shape[1]

torch.save(data, 'datasets-freebase/org3.pt')

org_map = {}
qqq = qq[0:2, qq[2]==4]
bqq = bq[0:2, bq[2]==2]

j = 1
for i in range(bqq.shape[1]):
    if not org_map.get(bqq[j, i].tolist()):
        org_map[bqq[j, i].tolist()] = org_map.__len__()
    bqq[j, i] = org_map[bqq[j, i].tolist()]

qq = qqq
bq = bqq