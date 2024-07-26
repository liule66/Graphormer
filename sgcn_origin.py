import os.path as osp
import torch
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.nn import SignedGCN
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

name = 'BitcoinOTC-1'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
dataset = BitcoinOTC(path, edge_window_size=1)
print(dataset[1])
# dataset = BitcoinOTC(root='Data/BitcoinOTC')
# Generate dataset.
pos_edge_indices, neg_edge_indices = [], []
for data in dataset:
    pos_edge_indices.append(data.edge_index[:, data.edge_attr > 0])
    neg_edge_indices.append(data.edge_index[:, data.edge_attr < 0])

pos_edge_index = torch.cat(pos_edge_indices, dim=1).to(device)
neg_edge_index = torch.cat(neg_edge_indices, dim=1).to(device)

# Build and train model.
model = SignedGCN(64, 64, num_layers=2, lamb=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_pos_edge_index, test_pos_edge_index = model.split_edges(pos_edge_index)
train_neg_edge_index, test_neg_edge_index = model.split_edges(neg_edge_index)

x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index)

def train():
    model.train()
    optimizer.zero_grad()
    z = model(x, train_pos_edge_index, train_neg_edge_index)
    loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index)
    return model.test(z, test_pos_edge_index, test_neg_edge_index)

embedding_dir = 'embedding'
for epoch in range(101):
    loss = train()
    auc, f1 = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, '
          f'F1: {f1:.4f}')

model.eval()
with torch.no_grad():
    final_embeddings = model(x, train_pos_edge_index, train_neg_edge_index)

embedding_save_path = osp.join(embedding_dir, f'soc-sign-bitcoinotc.csv_embeddings.pt')
torch.save(final_embeddings, embedding_save_path)
print(f'Embeddings for soc-sign-bitcoinotc.csv saved to {embedding_save_path}')