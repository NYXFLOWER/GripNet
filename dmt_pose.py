from src.layers import *
from src.decoder import multiRelaInnerProductDecoder
from torch_geometric.data import Data
import sys, time, os
import pandas as pd


# ###################################
# data processing
# ###################################
# load data
data = torch.load('./data/pose_comb_all.pt')

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
embed_dim = 32
learning_rate = 0.01


# model and initialization
class Model(Module):
    def forward(self, *input):
        pass

    def __init__(self, dmt):
        super(Model, self).__init__()
        self.embedding = Parameter(torch.Tensor(data.n_node, embed_dim))
        self.embedding.data.normal_()
        self.dmt = dmt


model = Model(
    multiRelaInnerProductDecoder(embed_dim, data.n_edge_type)
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

    pos_index = data.train_idx
    neg_index = typed_negative_sampling(data.train_idx, data.n_node, data.train_range).to(device)

    tmp_index = typed_negative_sampling(data.train_idx, data.n_drug,
                                        data.train_range[:-2]).to(device)
    neg_index = torch.cat([tmp_index, neg_index[:, tmp_index.shape[1]:]], dim=1)

    pos_score = model.dmt(z, pos_index, data.train_et)
    neg_score = model.dmt(z, neg_index, data.train_et)
    # pos_score = checkpoint(model.dmt, z, pos_index, data.train_et)
    # neg_score = checkpoint(model.dmt, z, neg_index, data.train_et)

    pos_loss = -torch.log(pos_score + EPS).mean()
    neg_loss = -torch.log(1 - neg_score + EPS).mean()
    loss = pos_loss + neg_loss

    loss.backward()

    optimizer.step()

    record = np.zeros((3, data.n_edge_type))  # auprc, auroc, ap

    # model.eval()
    # neg_index = typed_negative_sampling(data.train_idx, data.n_drug, data.train_range[:-2]).to(device)
    # neg_score = model.dmt(z, neg_index, data.train_et[:neg_index.shape[1]])

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
EPOCH_NUM = int(sys.argv[-1])
out_dir = './out/pose_dmt_all/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

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
name = '-{}-{}'.format(embed_dim, learning_rate)

if device == 'cuda':
    data = data.to('cpu')
    model = model.to('cpu')
    out = out.to('cpu')

# save model and record
torch.save(model.state_dict(), out_dir + str(EPOCH_NUM) + name + '-model.pt')
torch.save(out, out_dir + str(EPOCH_NUM) + name + '-record.pt')

# save record to csv
last_record = out.test_record[EPOCH_NUM-1].T
et_index = np.array(range(data.test_range.shape[0]), dtype=int).reshape(-1, 1)
combine = np.concatenate([et_index, np.array(data.n_edges_per_type, dtype=int).reshape(-1, 1), last_record], axis=1)
df = pd.DataFrame(combine, columns=['side_effect', 'n_instance', 'auprc', 'auroc', 'ap'])
df.astype({'side_effect': 'int32'})
df.to_csv(out_dir + str(EPOCH_NUM) + name + '-record.csv', index=False)

print('The trained model and the result record have been saved!')


