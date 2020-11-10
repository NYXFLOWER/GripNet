import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model_sparse import GTN
from matplotlib import pyplot as plt
import pdb
from torch_geometric.utils import dense_to_sparse, f1_score, accuracy
from torch_geometric.data import Data
import torch_sparse
import pickle
#from mem import mem_report
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import argparse
import time



file = './gripNet_baselines/data/auta.pt'


def get_datas(file):
    raw_datas = torch.load(file)
    n_nodes = raw_datas.num_nodes

    # node_features
    i = torch.LongTensor([[j, j] for j in range(n_nodes)])
    v = torch.FloatTensor([1.0 for j in range(n_nodes)])
    node_features = torch.sparse.FloatTensor(i.t(), v, torch.Size([n_nodes, n_nodes]))

    # A
    A = []
    n_type = raw_datas.edge_type.max() + 1
    for i in range(n_type):
        idx = (raw_datas.edge_type == i)
        edge_idx = raw_datas.edge_index[:, idx]
        l = edge_idx.shape[1]
        tmp = torch.ones(l, dtype=torch.float32)
        A.append((edge_idx, tmp))

    # labels
    labels = []
    tmp = torch.cat((raw_datas.train_idx.view(-1, 1), raw_datas.train_y.view(-1, 1)), dim=1)
    labels.append(tmp)
    tmp = torch.cat((raw_datas.train_idx.view(-1, 1), raw_datas.train_y.view(-1, 1)), dim=1)
    labels.append(tmp) # fake vali set
    tmp = torch.cat((raw_datas.test_idx.view(-1, 1), raw_datas.test_y.view(-1, 1)), dim=1)
    labels.append(tmp)

    return node_features, A, labels



epochs = 40
node_dim = 64
num_channels = 3
lr = 0.005
weight_decay = 0.001
num_layers = 3
norm = 'true'
adaptive_lr = 'false'
# dataset = 'IMDB'


node_features, A, labels = get_datas(file)

# num_nodes = edges[0].shape[0]
# A = []
#
# for i,edge in enumerate(edges):
#     edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(torch.cuda.LongTensor)
#     value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
#     A.append((edge_tmp,value_tmp))
# edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.cuda.LongTensor)
# value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
# A.append((edge_tmp,value_tmp))



node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.cuda.LongTensor)
train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.cuda.LongTensor)

valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.cuda.LongTensor)
valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.cuda.LongTensor)
test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.cuda.LongTensor)
test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.cuda.LongTensor)


num_classes = torch.max(train_target).item()+1

train_losses = []
train_f1s = []
val_losses = []
test_losses = []
val_f1s = []
test_f1s = []
final_f1 = 0

for cnt in range(5):
    best_val_loss = 10000
    best_test_loss = 10000
    best_train_loss = 10000
    best_train_f1 = 0
    best_val_f1 = 0
    best_test_f1 = 0
    model = GTN(num_edge=len(A),
                    num_channels=num_channels,
                    w_in = node_features.shape[1],
                    w_out = node_dim,
                    num_class=num_classes,
                    num_nodes = node_features.shape[0],
                    num_layers= num_layers)
    model.cuda()
    if adaptive_lr == 'false':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    else:
        optimizer = torch.optim.Adam([{'params':model.gcn.parameters()},
                                    {'params':model.linear1.parameters()},
                                    {'params':model.linear2.parameters()},
                                    {"params":model.layers.parameters(), "lr":0.01}
                                    ], lr=0.01, weight_decay=0.001)
    loss = nn.CrossEntropyLoss()
    Ws = []
    for i in range(20):
        time_begin = time.time()
        print('Epoch: ',i+1)
        for param_group in optimizer.param_groups:
            if param_group['lr'] > 0.005:
                param_group['lr'] = param_group['lr'] * 0.9
        model.train()
        model.zero_grad()
        loss, y_train, _ = model(A, node_features, train_node, train_target)
        loss.backward()
        optimizer.step()
        train_f1 = torch.mean(f1_score(torch.argmax(y_train,dim=1), train_target, num_classes=3)).cpu().numpy()
        print('Train - Loss: {}, Macro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1))
        model.eval()
        # Valid
        with torch.no_grad():
            val_loss, y_valid,_ = model.forward(A, node_features, valid_node, valid_target)
            val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=3)).cpu().numpy()
            print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
            test_loss, y_test,W = model.forward(A, node_features, test_node, test_target)
            test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=3)).cpu().numpy()
            test_acc = accuracy(torch.argmax(y_test,dim=1), test_target)
            print('Test - Loss: {}, Macro_F1: {}, Acc: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1, test_acc))
            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1
        torch.cuda.empty_cache()
        print('time: {:0.2f}'.format(time.time() - time_begin))
    print('---------------Best Results--------------------')
    print('Train - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_train_f1))
    print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
    print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))

