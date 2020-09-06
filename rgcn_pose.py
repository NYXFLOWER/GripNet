from src.layers import *
from src.decoder import multiRelaInnerProductDecoder
from data.utils import process_edge_multirelational
from torch_geometric.data import Data
import sys, time, os
import pandas as pd

print()
print("========================================================")
print("run: {} === PoSE-{} === {} === embedding dim: {}".format(int(sys.argv[-4]), int(sys.argv[-2]), sys.argv[-3], int(sys.argv[-1])))
print("========================================================")

# ###################################
# data processing
# ###################################
# load data
ddd = int(sys.argv[-2])
data = torch.load('./datasets-pose/pose-{}-combl.pt'.format(ddd))
out_dir = './out_baseline/pose-{}-baselines/'.format(ddd)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# node feature vector initialization
data.feat = sparse_id(data.n_node)
data.n_edges_per_type = [(i[1] - i[0]).data.tolist() for i in data.test_range]

# output dictionary
keys = ('train_record', 'test_record', 'train_out', 'test_out')
out = Data.from_dict({k: {} for k in keys})

# sent to device
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_name)
device = torch.device(device_name)
data = data.to(device)
out = out.to(device)


# ###################################
# Model
# ###################################
# hyper-parameter setting
r1_in_dim, r1_out_dim, r2_out_dim = 64, 32, int(sys.argv[-1])
n_relations, n_bases = data.n_edge_type, 16
learning_rate = 0.01
EPOCH_NUM = 100

# model and initialization
class Model(Module):
    def forward(self, *input):
        pass

    def __init__(self, r1, r2, dmt):
        super(Model, self).__init__()
        self.embedding = Parameter(torch.Tensor(data.n_node, r1_in_dim))
        self.embedding.data.normal_()
        self.rgcn1 = r1
        self.rgcn2 = r2
        self.dmt = dmt


model = Model(
    myRGCN(r1_in_dim, r1_out_dim, n_relations, n_bases, after_relu=False),
    myRGCN(r1_out_dim, r2_out_dim, n_relations, n_bases, after_relu=True),
    multiRelaInnerProductDecoder(r2_out_dim, data.n_edge_type)
).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ###################################
# Train and Test
# ###################################
@profile
def train(epoch):
    model.train()
    optimizer.zero_grad()

    z = torch.matmul(data.feat, model.embedding)
    z = model.rgcn1(z, data.train_idx, data.train_et, data.train_range)
    z = model.rgcn2(z, data.train_idx, data.train_et, data.train_range)

    pos_index = data.train_idx

    neg_index = typed_negative_sampling(data.train_idx, data.n_node, data.train_range).to(device)
    tmp_index = typed_negative_sampling(data.train_idx, data.n_drug,
                                        data.train_range[:-2]).to(device)
    neg_index = torch.cat([tmp_index, neg_index[:, tmp_index.shape[1]:]], dim=1)

    pos_score = model.dmt(z, pos_index, data.train_et)
    neg_score = model.dmt(z, neg_index, data.train_et)

    pos_loss = -torch.log(pos_score + EPS).mean()
    neg_loss = -torch.log(1 - neg_score + EPS).mean()
    loss = pos_loss + neg_loss

    loss.backward()

    optimizer.step()

    record = np.zeros((3, data.n_edge_type))  # auprc, auroc, ap

    model.eval()
    neg_index = typed_negative_sampling(data.train_idx, data.n_drug, data.train_range[:-2]).to(device)
    neg_score = model.dmt(z, neg_index, data.train_et[:neg_index.shape[1]])

    for i in range(data.train_range.shape[0] - 2):
        [start, end] = data.train_range[i]
        p_s = pos_score[start: end]
        n_s = neg_score[start: end]

        pos_target = torch.ones(p_s.shape[0])
        neg_target = torch.zeros(n_s.shape[0])

        score = torch.cat([p_s, n_s])
        target = torch.cat([pos_target, neg_target])

        record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)

    out.train_record[epoch] = record
    [auprc, auroc, ap] = record.mean(axis=1)
    out.train_out[epoch] = [auprc, auroc, ap]

    print('{:3d}   loss:{:0.4f}   auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}'
          .format(epoch, loss.tolist(), auprc, auroc, ap))

    return z, loss


test_neg_index = typed_negative_sampling(data.test_idx, data.n_drug, data.test_range[:-2]).to(device)


def test(z):
    model.eval()

    record = np.zeros((3, data.n_edge_type))

    pos_score = model.dmt(z, data.test_idx, data.test_et)
    neg_score = model.dmt(z, test_neg_index, data.test_et[:test_neg_index.shape[1]])
    for i in range(data.test_range.shape[0] - 2):
        [start, end] = data.test_range[i]
        p_s = pos_score[start: end]
        n_s = neg_score[start: end]

        pos_target = torch.ones(p_s.shape[0])
        neg_target = torch.zeros(n_s.shape[0])

        score = torch.cat([p_s, n_s])
        target = torch.cat([pos_target, neg_target])

        record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)

    return record


# if __name__ == '__main__':
# hhh


print('model training ...')

# train and test
for epoch in range(EPOCH_NUM):
    time_begin = time.time()

    z, loss = train(epoch)

    record_te = test(z)
    [auprc, auroc, ap] = record_te.mean(axis=1)

    print(
        '{:3d}   loss:{:0.4f}   auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}    time:{:0.1f}\n'
        .format(epoch, loss.tolist(), auprc, auroc, ap,
                (time.time() - time_begin)))

    out.test_record[epoch] = record_te
    out.test_out[epoch] = [auprc, auroc, ap]

# model name
name = '{}-{}-{}'.format(int(sys.argv[-4]), 'RGCN', int(sys.argv[-1]))

if device == 'cuda':
    data = data.to('cpu')
    model = model.to('cpu')
    out = out.to('cpu')

# save model and record
torch.save(model.state_dict(), out_dir + name + '-model.pt')
torch.save(out, out_dir + name + '-record.pt')

# save record to csv
last_record = out.test_record[EPOCH_NUM-1].T
et_index = np.array(range(data.test_range.shape[0]), dtype=int).reshape(-1, 1)
combine = np.concatenate([et_index, np.array(data.n_edges_per_type, dtype=int).reshape(-1, 1), last_record], axis=1)
df = pd.DataFrame(combine, columns=['side_effect', 'n_instance', 'auprc', 'auroc', 'ap'])
df.astype({'side_effect': 'int32'})
df.to_csv(out_dir + name + '-record.csv', index=False)

print('The trained model and the result record have been saved!')

with open(out_dir + name + '.txt', 'w') as f:
    f.write(str(out.test_out[EPOCH_NUM-1]))

print('The trained model and the result record have been saved!')
