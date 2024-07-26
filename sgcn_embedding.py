import os
import os.path as osp
import torch
from torch_geometric.nn import SignedGCN
import pandas as pd

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

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


def main():
    dataset_dir = 'dataset'
    embedding_dir = 'embedding'
    os.makedirs(embedding_dir, exist_ok=True)

    # dataset_files = [
    #     'Epinions.txt', 'Slashdot.txt', 'soc-sign-bitcoinalpha.csv','soc-sign-bitcoinotc.csv',
    #      'WikiElec.txt', 'WikiRfa.txt'
    # ]
    dataset_files = [

        'WikiRfa.txt'
    ]

    for dataset_file in dataset_files:
        file_path = osp.join(dataset_dir, dataset_file)
        data = load_data(file_path)

        edge_index, edge_attr = process_edges(data)

        pos_edge_index = edge_index[:, edge_attr > 0]
        neg_edge_index = edge_index[:, edge_attr < 0]



        pos_edge_index = pos_edge_index.to(device)
        neg_edge_index = neg_edge_index.to(device)

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

        for epoch in range(101):
            loss = train()
            auc, f1 = test()
            print(f'Dataset: {dataset_file}, Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}')

        model.eval()
        with torch.no_grad():
            final_embeddings = model(x, train_pos_edge_index, train_neg_edge_index)

        embedding_save_path = osp.join(embedding_dir, f'{osp.splitext(dataset_file)[0]}_embeddings.pt')
        torch.save(final_embeddings, embedding_save_path)
        print(f'Embeddings for {dataset_file} saved to {embedding_save_path}')


if __name__ == '__main__':
    main()

#Dataset: Epinions.txt, Epoch: 100, Loss: 0.6277, AUC: 0.8260, F1: 0.9361
#Dataset: Slashdot.txt, Epoch: 100, Loss: 0.8408, AUC: 0.7592, F1: 0.8305
#Dataset: soc-sign-bitcoinalpha.csv, Epoch: 100, Loss: 0.4340, AUC: 0.8284, F1: 0.9432
#Dataset: WikiElec.txt, Epoch: 100, Loss: 0.7351, AUC: 0.7970, F1: 0.8529
#Dataset: WikiRfa.txt, Epoch: 100, Loss: 0.7679, AUC: 0.7848, F1: 0.8588