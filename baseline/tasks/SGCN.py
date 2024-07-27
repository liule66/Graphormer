from typing import Tuple
import matplotlib.pyplot as plt
from torch import Tensor
import torch.nn as nn
import torch
import numpy as np
from torch_geometric.nn import SignedGCN
from torch_geometric import seed_everything

import sys
sys.path.append("..")
from ShareMethod import DataLoad

# cuda / mps / cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class MySignedGCN(SignedGCN):

    def __init__(self, x, in_channels: int, hidden_channels: int, num_layers: int, lamb: float = 5, bias: bool = True):
        super().__init__(in_channels, hidden_channels, num_layers, lamb, bias)

        # dimension reduce embeddings
        self.linear_DR = nn.Linear(x.shape[1], in_channels).to(device)
        self.x = x

    def dimension_reduction(self):
        return self.linear_DR(self.x)
    
    def test(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tuple[float, float, float]:
        """Evaluates node embeddings :obj:`z` on positive and negative test
        edges by computing AUC and F1 scores.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]

        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones((pos_p.size(0))),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        acc = accuracy_score(y, pred)
        auc = roc_auc_score(y, pred)
        f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0
        micro_f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
        macro_f1 = f1_score(y, pred, average='macro') if pred.sum() > 0 else 0

        return acc, auc, f1, micro_f1, macro_f1


def train(model, optimizer, x, train_pos_edge_index, train_neg_edge_index):
    model.train()
    optimizer.zero_grad()
    z = model(x, train_pos_edge_index, train_neg_edge_index)
    loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, x, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index)
    
    return model.test(z, test_pos_edge_index, test_neg_edge_index)

""" Picture
plt.xlabel('epoch')
plt.ylabel('loss')
epochs = []
tmp = []
for epoch in range(101):
    loss = train()
    auc, f1, acc = test()
    epochs.append(epoch)
    tmp.append([loss, auc, f1, acc])
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, ACC: {acc:.4f}')

res = np.array(tmp).T
plt.plot(epochs, res[0, :], color=(57/255, 197/255, 187/255), label='loss')
# plt.plot(epochs, res[1, :], color=(255/255, 165/255, 0/255), label='auc')
# plt.plot(epochs, res[2, :], color=(153/255, 211/255, 0/255), label='f1')
plt.plot(epochs, res[3, :], color=(255/255, 192/255, 203/255), label='acc')
plt.grid()
plt.legend()
plt.show()
"""

percent_list = [30, 50, 60, 70, 80, 100]
seed_list = [114, 514, 1919, 810, 721]

for percent in percent_list:

    print(f"{percent} Start!")

    res = []

    for times in range(5):

        seed = seed_list[times]

        torch.random.manual_seed(seed)
        seed_everything(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        dataloader = DataLoad(percent, times+1)
        train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index = dataloader.load_data_format()
        # x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index)
        node_nums = torch.max(train_pos_edge_index).item()
        x = dataloader.create_feature(node_nums)

        # Build and train model
        model = MySignedGCN(x, 32, 32, num_layers=2, lamb=5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        best_acc = 0
        best_auc = 0
        best_f1 = 0
        best_epoch = 0

        for epoch in range(100):
            x = model.dimension_reduction()
            loss = train(model, optimizer, x, train_pos_edge_index, train_neg_edge_index)
            # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        acc, auc, f1, micro_f1, macro_f1 = test(model, x, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index)

        res.append((acc, auc, f1, micro_f1, macro_f1))
        print(res[times])

    res = np.array(res)
    print(res.mean(axis=0))
    print(res.var(axis=0))
    print()
