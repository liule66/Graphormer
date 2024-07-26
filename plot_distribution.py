import os.path as osp
import pandas as pd
from matplotlib import pyplot as plt
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.nn import SignedGCN
import torch
import os
# Define the list of datasets
datasets = ['Epinions.txt', 'Slashdot.txt',
          'WikiElec.txt', 'WikiRfa.txt']


def load_data(file_path):
    if file_path.endswith('.txt'):
        data = pd.read_csv(file_path, sep='\t', header=None)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format")
    return data

def process_edges(data):
    edge_index = torch.tensor(data.iloc[:, :2].values.T, dtype=torch.long)
    edge_attr = torch.tensor(data.iloc[:, 2].values, dtype=torch.float)
    return edge_index, edge_attr

def select_dataset(datasets):
    print("Select a dataset from the following list:")
    for i, dataset in enumerate(datasets):
        print(f"{i + 1}. {dataset}")

    choice = int(input("Enter the number of the dataset you want to use: ")) - 1
    if 0 <= choice < len(datasets):
        return datasets[choice]
    else:
        print("Invalid choice. Using the first dataset by default.")
        return datasets[0]

script_dir = osp.abspath(osp.dirname(__file__))
dataset_dir = osp.join(script_dir, 'dataset')

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

name = select_dataset(datasets)
path = osp.join(dataset_dir, name)
data = load_data(path)
edge_index, edge_attr = process_edges(data)

# Generate dataset.
pos_edge_indices, neg_edge_indices = [], []
pos_edge_indices.append(edge_index[:, edge_attr > 0])
neg_edge_indices.append(edge_index[:, edge_attr < 0])

pos_edge_index = torch.cat(pos_edge_indices, dim=1).to(device)
neg_edge_index = torch.cat(neg_edge_indices, dim=1).to(device)

# Build and train model.
model = SignedGCN(64, 64, num_layers=2, lamb=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_pos_edge_index, test_pos_edge_index = model.split_edges(pos_edge_index)
train_neg_edge_index, test_neg_edge_index = model.split_edges(neg_edge_index)
embedding_save_path = f'embedding/{name}_embeddings.pt'
z = torch.load(embedding_save_path)

Dist_pos = torch.norm(z[test_pos_edge_index[0]] - z[test_pos_edge_index[1]], dim=1)
Dist_neg = torch.norm(z[test_neg_edge_index[0]] - z[test_neg_edge_index[1]], dim=1)

os.makedirs('distribution', exist_ok=True)

# 绘制正负边的距离分布
plt.figure(figsize=(10, 6))

# 正边距离分布
plt.hist(Dist_pos.cpu().numpy(), bins=50, alpha=0.5, label='Positive edges')

# 负边距离分布
plt.hist(Dist_neg.cpu().numpy(), bins=50, alpha=0.5, label='Negative edges')

# 添加标签和标题
plt.xlabel('Distance')
plt.ylabel('Number of edges')
plt.title(f'Distance Distribution for {name}')
plt.legend()

# 保存图像
plt.savefig(f'distribution/{name}_distance_distribution_test.png')

# 显示图像
plt.show()
