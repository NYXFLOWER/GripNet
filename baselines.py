from src.layers import *
from src.decoder import multiRelaInnerProductDecoder
from torch_geometric.data import Data
import sys, time, os
import pandas as pd
import torch.nn as nn

torch.manual_seed(1111)
np.random.seed(1111)

print()
print("========================================================")
print("run: {} === PoSE-{} === {} === embedding dim: {}".format(int(sys.argv[-4]), int(sys.argv[-2]), sys.argv[-3], int(sys.argv[-1])))
print("========================================================")
EPOCH_NUM = 100
# ###################################
# data processing
# ###################################
# load data
ddd = int(sys.argv[-2])
data = torch.load('./datasets-pose/pose-{}-combl.pt'.format(ddd))
# root = os.path.abspath(os.getcwd())
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
embed_dim = int(sys.argv[-1])
learning_rate = 0.01


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False,
                 double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        if model_name == 'RotatE':
            double_entity_embedding = True

        if model_name == 'ComplEx':
            double_entity_embedding = True
            double_relation_embedding = True

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(
            torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(
            torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (
                not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (
                not double_entity_embedding or not double_relation_embedding):
            raise ValueError(
                'ComplEx should use --double_entity_embedding and --double_relation_embedding')

        # self.evaluator = evaluator

    def forward(self, sample, et, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        head = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=sample[0]
        ).unsqueeze(1)

        relation = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=et
        ).unsqueeze(1)

        tail = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=sample[1]
        ).unsqueeze(1)

        # if mode == 'single':

        # else:
        #     head_part, tail_part = sample
        #     batch_size, negative_sample_size = tail_part.size(
        #         0), tail_part.size(1)
        #
        #     head = torch.index_select(
        #         self.entity_embedding,
        #         dim=0,
        #         index=head_part[:, 0]
        #     ).unsqueeze(1)
        #
        #     relation = torch.index_select(
        #         self.relation_embedding,
        #         dim=0,
        #         index=head_part[:, 1]
        #     ).unsqueeze(1)
        #
        #     tail = torch.index_select(
        #         self.entity_embedding,
        #         dim=0,
        #         index=tail_part.view(-1)
        #     ).view(batch_size, negative_sample_size, -1)


        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return F.logsigmoid(score).squeeze(dim = 1)

    def TransE(self, head, relation, tail):
        score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail):
        score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score


model = KGEModel(str(sys.argv[-3]), data.n_drug+data.n_gene, data.n_edge_type,
                 hidden_dim=embed_dim, gamma=12).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ###################################
# Train and Test
# ###################################
@profile
def train(epoch):
    model.train()
    optimizer.zero_grad()

    pos_index = data.train_idx
    neg_index = typed_negative_sampling(data.train_idx, data.n_node, data.train_range).to(device)

    neg_score = model(neg_index, data.train_et)
    pos_score = model(pos_index, data.train_et)

    pos_loss = -(pos_score + EPS).mean()
    neg_loss = -(1 - neg_score + EPS).mean()
    loss = pos_loss + neg_loss

    loss.backward()

    optimizer.step()

    record = np.zeros((3, data.n_edge_type))  # auprc, auroc, ap

    model.eval()
    neg_index = typed_negative_sampling(data.train_idx, data.n_drug, data.train_range[:-2]).to(device)
    neg_score = model(neg_index, data.train_et[: data.train_range[-2][0]])

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

    return loss


test_neg_index = typed_negative_sampling(data.test_idx, data.n_drug, data.test_range[:-2]).to(device)


def test():
    model.eval()

    record = np.zeros((3, data.n_edge_type))

    pos_score = model(data.test_idx, data.test_et)
    neg_score = model(test_neg_index, data.test_et[: data.test_range[-2][0]])
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


# if __name__ == '__0main__':
# hhh


print('model training ...')

# train and test
for epoch in range(EPOCH_NUM):
    time_begin = time.time()

    loss = train(epoch)

    record_te = test()
    [auprc, auroc, ap] = record_te.mean(axis=1)

    print(
        '{:3d}   loss:{:0.4f}   auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}    time:{:0.1f}\n'
        .format(epoch, loss.tolist(), auprc, auroc, ap,
                (time.time() - time_begin)))

    out.test_record[epoch] = record_te
    out.test_out[epoch] = [auprc, auroc, ap]

# model name
name = '{}-{}-{}'.format(int(sys.argv[-4]), str(sys.argv[-3]), embed_dim)

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