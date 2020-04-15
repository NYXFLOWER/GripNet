from src.layers import *
from src.decoder import multiClassInnerProductDecoder
from torch_geometric.data import Data
import sys, time, os
import pandas as pd



# ###################################
# data processing
# ###################################
# load data
ddd = int(sys.argv[-2])
data = torch.load('datasets-auta/auta-{}.pt'.format(ddd))

# output path
out_dir = './out/auta-{}/'.format(ddd)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# node feature vector initialization
data.a_feat = sparse_id(data.n_a_node)
data.p_feat = sparse_id(data.n_p_node)
data.aa_edge_weight = torch.ones(data.n_aa_edge)
data.pp_edge_weight = torch.ones(data.n_pp_edge)
data.n_node_per_type = [(i[1] - i[0]).data.tolist() for i in data.test_range]


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
class Model(Module):
    def forward(self, *input):
        pass

    def __init__(self, pp, pa, aa, mcip):
        super(Model, self).__init__()
        self.pp = pp
        self.pa = pa
        self.aa = aa
        self.mcip = mcip


# hyper-parameter setting
pp_nhids_gcn = [64, 32, 32]
pa_gcn = 32
pa_out = 32
aa_nhids_gcn = [pa_gcn + pa_out, 16, 16]
learning_rate = 0.01

# model init
model = Model(
    homoGraph(data.n_p_node, pp_nhids_gcn, start_graph=True),
    interGraph(sum(pp_nhids_gcn), pa_gcn, data.n_a_node, target_feat_dim=pa_out),
    homoGraph(pa_out, aa_nhids_gcn),
    multiClassInnerProductDecoder(sum(aa_nhids_gcn), data.n_a_type)
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

    z = model.pp(data.p_feat, data.pp_edge_idx, edge_weight=data.pp_edge_weight)
    z = model.pa(z, data.pa_edge_idx)
    z = model.aa(z, data.aa_edge_idx, edge_weight=data.aa_edge_weight)

    score, pred = model.mcip(z, data.train_node_idx, data.train_node_class)

    loss = -torch.log(score + EPS).mean()
    loss.backward()
    optimizer.step()

    micro, macro = micro_macro(data.train_node_class, pred)

    accuracy = np.zeros(shape=data.n_a_type)  # auprc, auroc, ap
    for i in range(data.train_range.shape[0]):
        [start, end] = data.train_range[i]
        s = score[start: end]
        t = torch.ones(size=s.shape)
        accuracy[i] = acc(t, s)

    out.train_record[epoch] = accuracy
    out.train_out[epoch] = np.array([accuracy.mean(), micro, macro])

    print('{:3d}   loss:{:0.4f}   accuracy:{:0.4f}   micro:{:0.4f}   macro:{:0.4f}'
          .format(epoch, loss.tolist(), accuracy.mean(), micro, macro))

    return z, loss


def test(z):
    model.eval()

    score, pred = model.mcip(z, data.test_node_idx, data.test_node_class)

    micro, macro = micro_macro(data.test_node_class, pred)

    accuracy = np.zeros(shape=data.n_a_type)  # auprc, auroc, ap
    for i in range(data.test_range.shape[0]):
        [start, end] = data.test_range[i]
        s = score[start: end]
        t = torch.ones(size=s.shape)
        accuracy[i] = acc(t, s)

    return accuracy, micro, macro


# if __name__ == '__main__':
# hhh
EPOCH_NUM = int(sys.argv[-1])
print('model training ...')

# train and test
for epoch in range(EPOCH_NUM):
    time_begin = time.time()

    z, loss = train(epoch)

    accuracy, micro, macro = test(z)


    print('{:3d}   loss:{:0.4f}   accuracy:{:0.4f}   micro:{:0.4f}   macro:{:0.4f}    time:{:0.1f}\n'
          .format(epoch, loss.tolist(), accuracy.mean(), micro, macro, (time.time() - time_begin)))

    out.test_record[epoch] = accuracy
    out.test_out[epoch] = np.array([accuracy.mean(), micro, macro])

# model name
name = '-{}-{}-{}-{}-{}'.format(pp_nhids_gcn, pa_gcn, pa_out, pa_out, aa_nhids_gcn, learning_rate)

if device == 'cuda':
    data = data.to('cpu')
    model = model.to('cpu')
    out = out.to('cpu')

# save model and record
torch.save(model.state_dict(), out_dir + str(EPOCH_NUM) + name + '-model.pt')
torch.save(out, out_dir + str(EPOCH_NUM) + name + '-record.pt')

# save record to csv
df = pd.DataFrame(columns=['author label', 'n_author', 'accuracy'])
df['author label'] = np.array(range(data.n_a_type), dtype=np.int)
df['n_author'] = np.array(data.n_node_per_type, dtype=np.int)
df['accuracy'] = out.test_record[EPOCH_NUM-1]

# df.astype({'side_effect': 'int32'})
df.to_csv(out_dir + str(EPOCH_NUM) + name + '-record.csv', index=False)
#
print('The trained model and the result record have been saved!')
