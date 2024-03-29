from gripnet.decoder import multiClassInnerProductDecoder
from torch_geometric.data import Data
from pytorch_memlab import profile

from gripnet.utils import EPS, sparse_id, process_data_multiclass, micro_macro
from gripnet.layers import homoGraph
import numpy as np
from torch.nn import Module
import torch
import sys
import time
import os
import pandas as pd

torch.manual_seed(1111)
np.random.seed(1111)

print()
print("========================================================")
print("run: {} epochs === Freebase-a === {}".format(int(sys.argv[1]), "GripNet"))
print("========================================================")


# ###################################
# data processing
# ###################################
lll = int(sys.argv[-1])
data = torch.load("datasets/freebase/freebase-b.pt")
train_set = torch.from_numpy(
    pd.read_csv(
        "datasets/freebase/train_test_split/label.dat.train_{}".format(lll),
        sep="\t",
        header=None,
    )
    .to_numpy()
    .T
)
test_set = torch.from_numpy(
    pd.read_csv(
        "datasets/freebase/train_test_split/label.dat.test_{}".format(lll),
        sep="\t",
        header=None,
    )
    .to_numpy()
    .T
)
data.train_node_idx, data.train_node_class, data.train_range = process_data_multiclass(
    train_set, data.n_a_type
)
data.test_node_idx, data.test_node_class, data.test_range = process_data_multiclass(
    test_set, data.n_a_type
)

# output path
out_dir = "./out/freebase-a/book-{}/".format(lll)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# node feature vector initialization
data.a_feat = sparse_id(data.n_a_node)
data.p_feat = sparse_id(data.n_p_node)
data.aa_edge_weight = torch.ones(data.n_aa_edge)
data.pp_edge_weight = torch.ones(data.n_pp_edge)
data.n_node_per_type = [(i[1] - i[0]) for i in data.test_range]
data.n_node_per_type_train = [(i[1] - i[0]) for i in data.train_range]


# output dictionary
keys = ("train_out", "test_out")
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

    def __init__(self, pp, mcip):
        super(Model, self).__init__()
        self.pp = pp
        self.mcip = mcip


# hyper-parameter setting
pp_nhids_gcn = [256, 128, 128]

# pp_nhids_gcn = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]

learning_rate = 0.01

# model init
model = Model(
    homoGraph(pp_nhids_gcn, start_graph=True, in_dim=data.n_a_node),
    multiClassInnerProductDecoder(pp_nhids_gcn[-1], data.n_a_type),
).to(device)

print(model)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ###################################
# Train and Test
# ###################################
@profile
def train(epoch):
    model.train()
    optimizer.zero_grad()

    z = model.pp(data.a_feat, data.aa_edge_idx, edge_weight=data.aa_edge_weight)

    score = model.mcip(z, data.train_node_idx)
    pred = torch.argmax(score, dim=1)

    loss = -torch.log(score[range(score.shape[0]), data.train_node_class] + EPS).mean()
    loss.backward()
    optimizer.step()

    micro, macro = micro_macro(data.train_node_class, pred)

    out.train_out[epoch] = np.array([micro, macro])

    print(
        "{:3d}   loss:{:0.4f}   micro:{:0.4f}   macro:{:0.4f}".format(
            epoch, loss.tolist(), micro, macro
        )
    )

    return z, loss


def test(z):
    model.eval()

    score = model.mcip(z, data.test_node_idx)
    pred = torch.argmax(score, dim=1)

    micro, macro = micro_macro(data.test_node_class, pred)

    return micro, macro


EPOCH_NUM = int(sys.argv[1])
print("model training ...")
z = 0

# train and test
for epoch in range(EPOCH_NUM):
    time_begin = time.time()

    z, loss = train(epoch)

    micro, macro = test(z)

    print(
        "{:3d}   loss:{:0.4f}   micro:{:0.4f}   macro:{:0.4f}    time:{:0.1f}\n".format(
            epoch, loss.tolist(), micro, macro, (time.time() - time_begin)
        )
    )

    out.test_out[epoch] = np.array([micro, macro])

# model name
name = "-{}".format(pp_nhids_gcn)

if device == "cuda":
    data = data.to("cpu")
    model = model.to("cpu")
    out = out.to("cpu")

# save model and record
torch.save(model.state_dict(), out_dir + str(EPOCH_NUM) + name + "-model.pt")
torch.save(out, out_dir + str(EPOCH_NUM) + name + "-record.pt")

with open(out_dir + str(EPOCH_NUM) + name + ".txt", "w") as f:
    f.write(str(out.test_out[EPOCH_NUM - 1]))

print("The trained model and the result record have been saved!")

torch.save(z, out_dir + str(EPOCH_NUM) + name + "-weight.pt")
