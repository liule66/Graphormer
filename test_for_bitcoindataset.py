import os.path as osp
import time
from math import fabs
import pandas as pd
from torch_geometric.nn import SignedGCN
import torch
from torch import Tensor
import os
import dataset_loader
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
# Define the list of datasets
datasets = ['soc-sign-bitcoinalpha.csv', 'soc-sign-bitcoinotc.csv']

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
train_pos_edge_index, test_pos_edge_index = model.split_edges(pos_edge_index)
train_neg_edge_index, test_neg_edge_index = model.split_edges(neg_edge_index)

x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index)

def discriminate(z: Tensor, pos_neg_edge_index: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tensor:
    """Given node embeddings :obj:z, classifies the link relation
    between node pairs :obj:edge_index to be either positive,
    negative or non-existent.

    Args:
        x (torch.Tensor): The input node features.
        edge_index (torch.Tensor): The edge indices.
    """
    k = 23
    device = z.device  # Ensure device is set to the same as z
    #data_dir = "/home/zyji/PycharmProjects/Graphormr/Dist"
    data_dir = "Dist"
    os.makedirs(data_dir, exist_ok=True)

    # Define file paths
    diffusion_matrix_path = os.path.join(data_dir, "diffusion_matrix.pt")
    k_pos_neighbors_path = os.path.join(data_dir, "k_pos_neighbors_tensor.pt")
    k_neg_neighbors_path = os.path.join(data_dir, "k_neg_neighbors_tensor.pt")
    k_avg_pos_distance_path = os.path.join(data_dir, "k_avg_pos_distance.pt")
    k_avg_neg_distance_path = os.path.join(data_dir, "k_avg_neg_distance.pt")

    z = z.to(device)
    pos_edge_index = pos_edge_index.to(device)
    neg_edge_index = neg_edge_index.to(device)
    num_nodes = z.size(0)

    # Check if files exist and load data
    if os.path.exists(k_pos_neighbors_path) and os.path.exists(k_neg_neighbors_path) and os.path.exists(k_avg_pos_distance_path) and os.path.exists(k_avg_neg_distance_path):
        k_pos_neighbors_tensor = [t.to(device) for t in torch.load(k_pos_neighbors_path)]
        k_neg_neighbors_tensor = [t.to(device) for t in torch.load(k_neg_neighbors_path)]
        k_avg_pos_distance = torch.load(k_avg_pos_distance_path).to(device)
        k_avg_neg_distance = torch.load(k_avg_neg_distance_path).to(device)

    else:
        # diffusion_matrix = torch.load(diffusion_matrix_path).to(device)
        # print("diffusion_matrix shape", diffusion_matrix.shape)
        print(z.size(0))
        # Initialize neighbors
        pos_neighbors = [[] for _ in range(z.size(0))]
        neg_neighbors = [[] for _ in range(z.size(0))]

        # Find neighbors from pos_edge_index and neg_edge_index
        for edge in pos_edge_index.t():
            pos_neighbors[edge[0]].append(edge[1].item())
            pos_neighbors[edge[1]].append(edge[0].item())

        for edge in neg_edge_index.t():
            neg_neighbors[edge[0]].append(edge[1].item())
            neg_neighbors[edge[1]].append(edge[0].item())

        # Supplement with diffusion matrix if necessary
        # for i in range(num_nodes):
        #     if len(pos_neighbors[i]) < k or len(neg_neighbors[i]) < k:
        #         for j in range(num_nodes):
        #             if i == j:
        #                 continue
        #             if diffusion_matrix[i, j] > 0 and j not in pos_neighbors[i]:
        #                 pos_neighbors[i].append(j)
        #             elif diffusion_matrix[i, j] < 0 and j not in neg_neighbors[i]:
        #                 neg_neighbors[i].append(j)

        # Convert lists to tensors
        pos_neighbors_tensor = [torch.tensor(neighbors, dtype=torch.long, device=device) for neighbors in pos_neighbors]
        neg_neighbors_tensor = [torch.tensor(neighbors, dtype=torch.long, device=device) for neighbors in neg_neighbors]

        # New k-nearest and k-farthest neighbors calculation
        k_pos_neighbors_tensor = [torch.tensor([], dtype=torch.long, device=device) for _ in range(z.size(0))]
        k_neg_neighbors_tensor = [torch.tensor([], dtype=torch.long, device=device) for _ in range(z.size(0))]
        k_avg_pos_distance = torch.zeros(z.size(0), device=device)
        k_avg_neg_distance = torch.zeros(z.size(0), device=device)

        for i in range(z.size(0)):
            if len(pos_neighbors_tensor[i]) > 0:
                pos_distances = torch.norm(z[i] - z[pos_neighbors_tensor[i]], dim=1)
                if len(pos_distances) > k:
                    k_pos_neighbors_tensor[i] = pos_neighbors_tensor[i][pos_distances.topk(k, largest=False).indices]
                    k_avg_pos_distance[i] = pos_distances.topk(k, largest=False).values.median().item()
                else:
                    k_pos_neighbors_tensor[i] = pos_neighbors_tensor[i]
                    k_avg_pos_distance[i] = pos_distances.median().item()
            else:
                k_avg_pos_distance[i] = float('inf')

            if len(neg_neighbors_tensor[i]) > 0:
                neg_distances = torch.norm(z[i] - z[neg_neighbors_tensor[i]], dim=1)
                if len(neg_distances) > k:
                    k_neg_neighbors_tensor[i] = neg_neighbors_tensor[i][neg_distances.topk(k, largest=True).indices]
                    k_avg_neg_distance[i] = neg_distances.topk(k, largest=True).values.median().item()
                else:
                    k_neg_neighbors_tensor[i] = neg_neighbors_tensor[i]
                    k_avg_neg_distance[i] = neg_distances.median().item()
            else:
                k_avg_neg_distance[i] = float('inf')

        # Save new k-nearest and k-farthest neighbors data
        torch.save(k_pos_neighbors_tensor, k_pos_neighbors_path)
        torch.save(k_neg_neighbors_tensor, k_neg_neighbors_path)
        torch.save(k_avg_pos_distance, k_avg_pos_distance_path)
        torch.save(k_avg_neg_distance, k_avg_neg_distance_path)

    # Calculate distances between pairs in pos_neg_edge_index
    Dist_ij = torch.norm(z[pos_neg_edge_index[0]] - z[pos_neg_edge_index[1]], dim=1)

    # print("Dist_ij", Dist_ij)
    # print("k_pos_neighbors_tensor", k_pos_neighbors_tensor)
    # print("k_neg_neighbors_tensor", k_neg_neighbors_tensor)
    # print("k_avg_pos_distance", k_avg_pos_distance)
    # print("k_avg_neg_distance", k_avg_neg_distance)
    # print("k_avg_pos_distance shape", k_avg_pos_distance.shape)
    # print("k_avg_neg_distance shape", k_avg_neg_distance.shape)
    # print("k_pos_neighbors_tensor shape",k_avg_pos_distance.shape)
    # print("k_neg_neighbors_tensor shape",k_avg_neg_distance.shape)
    # print("test_pos_edge_index", test_pos_edge_index)
    # print("test_neg_edge_index", test_neg_edge_index)
    # print("z shape", z.size(0))

    # Initialize logits for softmax
    logits = torch.zeros((Dist_ij.size(0), 3), device=device)
    # a=0.025
    # b=0.09
    #parameter for 6th dataset
    a=1
    b=0.5
    for i in range(Dist_ij.size(0)):
        node = pos_neg_edge_index[1, i].item()
        dist = Dist_ij[i]
        dist_1=fabs(dist - k_avg_pos_distance[node])-a
        dist_2=fabs(dist - k_avg_neg_distance[node])+b

        if dist_1< dist_2:
            logits[i] = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)  # Positive edge

        else:
            logits[i] = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)  # Negative edge

    return torch.log_softmax(logits, dim=1)


#embedding_save_path = '/home/zyji/PycharmProjects/Graphormr/embedding/embeddings.pt'
embedding_save_path = f'embedding/{name}_embeddings.pt'


def explainer(pos_neg_edge_index: Tensor, is_positive: bool) -> float:
    """
    Explains the predictions by comparing the expected neighbors with the ground truth neighbors.

    Args:
        pos_neg_edge_index (Tensor): The edge indices to explain.
        is_positive (bool): Flag indicating whether the edges are positive or negative.

    Returns:
        float: The explanation accuracy.
    """
    z = torch.load(embedding_save_path)
    device = pos_neg_edge_index.device
    num_nodes = pos_neg_edge_index.max().item() + 1

    # Initialize ground truth and expected neighbors for each node
    ground_truth = [set() for _ in range(num_nodes)]
    exp_nei = [set() for _ in range(num_nodes)]

    # Load precomputed neighbor tensors
    # k_pos_neighbors_tensor = torch.load("/home/zyji/PycharmProjects/Graphormr/Dist/k_pos_neighbors_tensor.pt")
    # k_neg_neighbors_tensor = torch.load("/home/zyji/PycharmProjects/Graphormr/Dist/k_neg_neighbors_tensor.pt")
    k_pos_neighbors_tensor = torch.load("Dist/k_pos_neighbors_tensor.pt")
    k_neg_neighbors_tensor = torch.load("Dist/k_neg_neighbors_tensor.pt")

    # Predict the edge types
    predictions = discriminate(z, pos_neg_edge_index, pos_edge_index, neg_edge_index)[:, :2].max(dim=1)[1]

    for i, edge in enumerate(pos_neg_edge_index.t()):
        node = edge[1].item()
        if is_positive:
            ground_truth[node].update(k_pos_neighbors_tensor[node].tolist())
            if predictions[i] == 0:  # Predicted as positive
                exp_nei[node].update(k_pos_neighbors_tensor[node].tolist())
            else:  # Predicted as negative or no edge
                exp_nei[node].update(k_neg_neighbors_tensor[node].tolist())
        else:
            ground_truth[node].update(k_neg_neighbors_tensor[node].tolist())
            if predictions[i] == 1:  # Predicted as negative
                exp_nei[node].update(k_neg_neighbors_tensor[node].tolist())
            else:  # Predicted as positive or no edge
                exp_nei[node].update(k_pos_neighbors_tensor[node].tolist())

    # Calculate explanation accuracy for each node and take the mean
    accuracies = []
    for i in range(num_nodes):
        if len(ground_truth[i]) > 0:
            correct_predictions = ground_truth[i].intersection(exp_nei[i])
            accuracy = len(correct_predictions) / len(ground_truth[i])
            accuracies.append(accuracy)

    # If there are no valid ground truth sets, return 0.0
    if len(accuracies) == 0:
        return 0.0
    else:
        explanation_accuracy = sum(accuracies) / len(accuracies)

    return explanation_accuracy

from typing import Tuple
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def test() -> Tuple[float, float, float, float]:
    """Evaluates node embeddings :obj:z on positive and negative test
    edges by computing AUC, F1 scores, and Accuracy.

    Args:
        z (torch.Tensor): The node embeddings.
        pos_edge_index (torch.Tensor): The positive edge indices.
        neg_edge_index (torch.Tensor): The negative edge indices.
    """
    with torch.no_grad():
        z = torch.load(embedding_save_path)
        pos_p = discriminate(z, test_pos_edge_index, train_pos_edge_index, train_neg_edge_index)[:, :2].max(dim=1)[1]
        neg_p = discriminate(z, test_neg_edge_index, train_pos_edge_index, train_neg_edge_index)[:, :2].max(dim=1)[1]
        exp_pos_p = explainer(test_pos_edge_index, is_positive=True)
        exp_neg_p = explainer(test_neg_edge_index, is_positive=False)
        exp_total = (exp_pos_p + exp_neg_p) / 2

    pred = (1 - torch.cat([pos_p, neg_p])).cpu()
    y = torch.cat([pred.new_ones((pos_p.size(0))), pred.new_zeros(neg_p.size(0))])
    pred, y = pred.numpy(), y.numpy()

    auc = roc_auc_score(y, pred)
    f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0
    acc = accuracy_score(y, pred.round())

    # Calculate the number of edges that are misclassified
    false_pos = ((1 - pos_p) == 0).sum().item()  # Positive edges predicted as negative
    false_neg = (neg_p == 0).sum().item()       # Negative edges predicted as positive

    # Plotting the distributions
    import matplotlib.pyplot as plt
    import os
    os.makedirs('distribution', exist_ok=True)

    plt.figure()
    plt.bar(['False Positive', 'False Negative'], [false_pos, false_neg], color=['red', 'blue'])
    plt.xlabel('Misclassification Type')
    plt.ylabel('Number of Edges')
    plt.title('Misclassified Edges Distribution')
    plt.savefig(f'distribution/{name}_misclassified_edges.png')
    plt.close()

    return auc, f1, acc, exp_total

for epoch in range(1):
    epoch_time_start = time.time()
    auc, f1, acc, exp_total = test()
    epoch_time_end = time.time()
    epoch_time = epoch_time_end - epoch_time_start
    print(f'Epoch: {epoch:03d}, Time: {epoch_time:.2f}s, AUC: {auc:.4f}, '
          f'F1: {f1:.4f}, ACC: {acc:.4f}, exp_total: {exp_total:.4f}')
