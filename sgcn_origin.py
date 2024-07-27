import os
import os.path as osp
import torch
from sympy.physics.control.control_plots import plt
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.nn import SignedGCN
import dataset_loader

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def main():
    name = 'BitcoinOTC-1'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = dataset_loader.BitcoinOTC(path, edge_window_size=1)
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
            auc, f1 = model.test(z, test_pos_edge_index, test_neg_edge_index)
            preds = model.predict(z, test_pos_edge_index, test_neg_edge_index)
            labels = torch.cat([torch.ones(test_pos_edge_index.size(1)), torch.zeros(test_neg_edge_index.size(1))])
            acc = (preds == labels.to(device)).sum().item() / labels.size(0)
        return auc, f1, acc

    embedding_dir = 'embedding'
    os.makedirs(embedding_dir, exist_ok=True)

    for epoch in range(101):
        loss = train()
        auc, f1, acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}')

    model.eval()
    with torch.no_grad():
        final_embeddings = model(x, train_pos_edge_index, train_neg_edge_index)

    embedding_save_path = osp.join(embedding_dir, f'soc-sign-bitcoinotc.csv_embeddings.pt')
    torch.save(final_embeddings, embedding_save_path)
    print(f'Embeddings for soc-sign-bitcoinotc.csv saved to {embedding_save_path}')

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

    return auc, f1, acc

seed_list = [1145, 14, 191, 9810, 721]

if __name__ == '__main__':
    res = []
    for seed in seed_list:
        torch.manual_seed(seed)

        auc, f1, acc = main()

        res.append((auc, f1, acc))

    # compute avg var
    res = np.array(res)
    print("final res")
    print(res.mean(axis=0))
    print(res.var(axis=0))
    print()
