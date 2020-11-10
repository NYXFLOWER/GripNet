from pytorch_memlab import profile
import torch
import torch.nn.functional as F
from sklearn import metrics
from utils import multiClassInnerProductDecoder, sparse_id
from torch_geometric.nn import AGNNConv
import numpy as np
import argparse
import time

torch.manual_seed(1111)
np.random.seed(1111)

parser = argparse.ArgumentParser(description='agnn node classification')
parser.add_argument('-i', '--input', default='../data/AuTa_data_0.pt', type=str, help='input file path')
parser.add_argument('-o', '--output', default='../result/auta/auta_gcn.txt', type=str, help='output file path')
parser.add_argument('-l', '--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('-e', '--epoch', default=100, type=int, help='epoch')
parser.add_argument('--embedding', type=int, default=32, help='output layer embedding')
# parser.add_argument('--seed', type=int, default=3, help='seed value')
args = parser.parse_args()

data = torch.load(args.input)
output_result = args.output
data.num_nodes = data.edge_index.max().tolist() + 1
# torch.manual_seed(args.seed)
embedding = args.embedding
data.x = sparse_id(data.num_nodes)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = torch.nn.Parameter(torch.Tensor(data.num_nodes, embedding))
        self.embedding.data.normal_()
        self.conv1 = AGNNConv(requires_grad=False)
        self.conv2 = AGNNConv(requires_grad=True)
        self.mclp = multiClassInnerProductDecoder(embedding, data.num_classes)

    def forward(self, x, edge_index):
        x = torch.matmul(x, self.embedding)
        x = F.relu(x, inplace=True)
        x = self.conv1(x, edge_index)
        x = F.relu(self.conv2(x, edge_index), inplace=True)
        x = self.mclp(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def micro_macro(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    micro, macro = metrics.f1_score(y, pred, average='micro'), metrics.f1_score(y, pred, average='macro')
    return micro, macro


@profile
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_idx], data.train_y)
    loss.backward()
    optimizer.step()
    pred = out[data.train_idx].max(1)[1]
    micro, macro = micro_macro(data.train_y, pred)
    return micro, macro, loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[data.test_idx].max(1)[1]
    micro, macro = micro_macro(data.test_y, pred)
    return micro, macro


f = open(output_result, 'w')

for epoch in range(1, args.epoch+1):
    time_begin = time.time()

    train_micro, train_macro, train_loss = train()
    test_micro, test_macro = test()
    line = 'Epoch{:3d} | Train loss: {:0.4f} | Train Micro: {:0.4f} | Train Macro: {:0.4f}\n' \
           '         | Test Micro: {:0.4f} | Test Macro: {:0.4f}       Time:{:0.2f}'.format(
        epoch, train_loss, train_micro, train_macro, test_micro, test_macro,
        (time.time() - time_begin))
    if epoch == 100:
        print(line)
    f.write(line + '\n')
f.close()
