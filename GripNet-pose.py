from gripnet.utils import (
    EPS,
    sparse_id,
    typed_negative_sampling,
    auprc_auroc_ap,
    negative_sampling,
)
from gripnet.layers import homoGraph, interGraph
import numpy as np
from torch.nn import Module
import torch
from gripnet.decoder import multiRelaInnerProductDecoder
from torch_geometric.data import Data
import sys
import time
import os
import pandas as pd
from pytorch_memlab import profile
from torch.utils.checkpoint import checkpoint

print()
print("========================================================")
print(
    "run: {} === PoSE-{} === {}".format(int(sys.argv[-3]), int(sys.argv[-2]), "GripNet")
)
print("========================================================")


# ###################################
# data processing
# ###################################
# load data
ddd = int(sys.argv[-2])
data = torch.load("datasets/pose/pose-{}.pt".format(ddd))

# d = torch.load('gripNet_baselines/data/book_data_0.pt')

# output path
out_dir = "./out/pose-nneg-{}/".format(ddd)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# node feature vector initialization
data.g_feat = sparse_id(data.n_g_node)
data.d_feat = sparse_id(data.n_d_node)
data.edge_weight = torch.ones(data.n_gg_edge)
data.gd_edge_index = torch.tensor(data.gd_edge_index, dtype=torch.long)
data.gg_edge_index = torch.tensor(data.gg_edge_index, dtype=torch.long)
data.n_edges_per_type = [(i[1] - i[0]).data.tolist() for i in data.test_range]


# output dictionary
keys = ("train_record", "test_record", "train_out", "test_out")
out = Data.from_dict({k: {} for k in keys})

# sent to device
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(device_name)
device = torch.device(device_name)
data = data.to(device)
out = out.to(device)


# ###################################
# Model
# ###################################
class Model(Module):
    def forward(self, *input):
        pass

    def __init__(self, gg, gd, dd, dmt):
        super(Model, self).__init__()
        self.gg = gg
        self.gd = gd
        self.dd = dd
        self.dmt = dmt


# hyper-parameter setting
gg_nhids_gcn = [32, 16, 16]
# gd_gcn = 16
gd_out = [16, 32]
dd_nhids_gcn = [sum(gd_out), int(sys.argv[-1])]
learning_rate = 0.01
EPOCH_NUM = 100

# model init
model = Model(
    homoGraph(gg_nhids_gcn, start_graph=True, in_dim=data.n_g_node),
    interGraph(sum(gg_nhids_gcn), gd_out[0], data.n_d_node, target_feat_dim=gd_out[-1]),
    homoGraph(dd_nhids_gcn, multi_relational=True, n_rela=data.n_dd_edge_type),
    multiRelaInnerProductDecoder(sum(dd_nhids_gcn), data.n_dd_edge_type),
).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

z = 0


# ###################################
# Train and Test
# ###################################
@profile
def train(epoch):
    model.train()
    optimizer.zero_grad()

    z = model.gg(
        data.g_feat, data.gg_edge_index, edge_weight=data.edge_weight, if_catout=True
    )
    z = model.gd(z, data.gd_edge_index, mod="cat", if_relu=True)
    z = model.dd(
        z,
        data.train_idx,
        edge_type=data.train_et,
        range_list=data.train_range,
        if_catout=True,
    )

    pos_index = data.train_idx
    # neg_index = typed_negative_sampling(data.train_idx, data.n_d_node, data.train_range).to(device)
    neg_index = negative_sampling(data.train_idx, data.n_d_node).to(device)

    pos_score = checkpoint(model.dmt, z, pos_index, data.train_et)
    neg_score = checkpoint(model.dmt, z, neg_index, data.train_et)

    pos_loss = -torch.log(pos_score + EPS).mean()
    neg_loss = -torch.log(1 - neg_score + EPS).mean()
    loss = pos_loss + neg_loss

    loss.backward()

    optimizer.step()

    record = np.zeros((3, data.n_dd_edge_type))  # auprc, auroc, ap
    for i in range(data.train_range.shape[0]):
        [start, end] = data.train_range[i]
        p_s = pos_score[start:end]
        n_s = neg_score[start:end]

        pos_target = torch.ones(p_s.shape[0])
        neg_target = torch.zeros(n_s.shape[0])

        score = torch.cat([p_s, n_s])
        target = torch.cat([pos_target, neg_target])

        record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)

    out.train_record[epoch] = record
    [auprc, auroc, ap] = record.mean(axis=1)
    out.train_out[epoch] = [auprc, auroc, ap]

    print(
        "{:3d}   loss:{:0.4f}   auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}".format(
            epoch, loss.tolist(), auprc, auroc, ap
        )
    )

    return z, loss


test_neg_index = typed_negative_sampling(
    data.test_idx, data.n_d_node, data.test_range
).to(device)


def test(z):
    model.eval()

    record = np.zeros((3, data.n_dd_edge_type))

    pos_score = model.dmt(z, data.test_idx, data.test_et)
    neg_score = model.dmt(z, test_neg_index, data.test_et)

    for i in range(data.test_range.shape[0]):
        [start, end] = data.test_range[i]
        p_s = pos_score[start:end]
        n_s = neg_score[start:end]

        pos_target = torch.ones(p_s.shape[0])
        neg_target = torch.zeros(n_s.shape[0])

        score = torch.cat([p_s, n_s])
        target = torch.cat([pos_target, neg_target])

        record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)

    return record


# if __name__ == '__main__':
# hhh

print("model training ...")

# train and test
for epoch in range(EPOCH_NUM):
    time_begin = time.time()

    z, loss = train(epoch)

    record_te = test(z)
    [auprc, auroc, ap] = record_te.mean(axis=1)

    print(
        "{:3d}   loss:{:0.4f}   auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}    time:{:0.2f}\n".format(
            epoch, loss.tolist(), auprc, auroc, ap, (time.time() - time_begin)
        )
    )

    out.test_record[epoch] = record_te
    out.test_out[epoch] = [auprc, auroc, ap]

# model name
name = "{}-{}-{}-{}".format(sys.argv[-3], gg_nhids_gcn, gd_out, dd_nhids_gcn)

if device == "cuda":
    data = data.to("cpu")
    model = model.to("cpu")
    out = out.to("cpu")

# save model and record
torch.save(model.state_dict(), out_dir + name + "-model.pt")
torch.save(out, out_dir + name + "-record.pt")

# save record to csv
last_record = out.test_record[EPOCH_NUM - 1].T
et_index = np.array(range(data.test_range.shape[0]), dtype=int).reshape(-1, 1)
combine = np.concatenate(
    [et_index, np.array(data.n_edges_per_type, dtype=int).reshape(-1, 1), last_record],
    axis=1,
)
df = pd.DataFrame(
    combine, columns=["side_effect", "n_instance", "auprc", "auroc", "ap"]
)
df.astype({"side_effect": "int32"})
df.to_csv(out_dir + name + "-record.csv", index=False)

print("The trained model and the result record have been saved!")

with open(out_dir + name + ".txt", "w") as f:
    f.write(str(out.test_out[EPOCH_NUM - 1]))

print("The trained model and the result record have been saved!")

torch.save(z, out_dir + name + "-weight.pt")
