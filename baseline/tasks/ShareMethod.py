import torch
from torch import Tensor
import numpy as np
import pandas as pd
import os

# cuda / mps / cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class DataLoad:

    def __init__(self, percent, times) -> None:
        self.percent = percent
        self.times = times


    def load_data_format(self):
        """
        load data in original files
        return train pos/neg; test pos/neg : [[start][end]]  size=(2, m)
        """
    
        # **using "../data/cotton-data/" except "data/" here** : run `python xxx.py` need `cd` the dir
        train_data = torch.tensor(np.loadtxt(f"../../data/cotton-data/{self.percent}-0-{self.times}_training.txt", delimiter="\t", dtype=int))
        test_data = torch.tensor(np.loadtxt(f"../../data/cotton-data/{self.percent}-0-{self.times}_test.txt", delimiter="\t", dtype=int))

        # train
        train_data_idx = train_data[:, :2].T  # ori_data: (src, dst, val), only read the index
        train_dataset = train_data[:, 2]

        train_pos_edge_index = train_data_idx[:, train_dataset > 0].to(device)
        train_neg_edge_index = train_data_idx[:, train_dataset < 0].to(device)

        # test
        test_data_idx = test_data[:, :2].T
        test_dataset = test_data[:, 2]

        test_pos_edge_index = test_data_idx[:, test_dataset > 0].to(device)
        test_neg_edge_index = test_data_idx[:, test_dataset < 0].to(device)

        return train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index


    def create_feature(self, node_num):
        # return torch.tensor(np.loadtxt(f"../../data/cotton-data/{self.percent}_feature.txt", delimiter="\t"), dtype=torch.float).to(device)
        sim_matrix = np.loadtxt(f"../../data/cotton-data/{self.percent}-{self.times}_feature.txt", delimiter="\t")

        """
        # PCA
        pca = PCA(args.feature_dim, random_state=seed)
        x = torch.tensor(pca.fit_transform(sim_matrix)).to(torch.float).to(device)
        """

        """
        # uniform random
        x = torch.randn((node_num+1, args.feature_dim)).to(device)
        """

        # linear
        x = torch.tensor(sim_matrix).to(torch.float).to(device)

        return x

    def load_gene_simMat(self):
        """
        load data in original adjacency files ( g-g-{percent}.csv )
        return adjmat: np array
        """
        return np.loadtxt(f"../data/cotton-data/g-g-{self.percent}.csv", delimiter=",")
    # return np.loadtxt(f"../../data/cotton-data/g-g-{self.percent}.csv", delimiter=",")


    def load_data(self):
        """return directly"""
        
        train_data = torch.tensor(np.loadtxt(f"../../data/cotton-data/{self.percent}-0-{self.times}_training.txt", delimiter="\t", dtype=int)).to(device)
        test_data = torch.tensor(np.loadtxt(f"../../data/cotton-data/{self.percent}-0-{self.times}_test.txt", delimiter="\t", dtype=int)).to(device)

        # train
        train_data_idx = train_data[:, :2].T  # ori_data: (src, dst, val), only read the index
        train_dataset = train_data[:, 2]

        # test
        test_data_idx = test_data[:, :2].T
        test_dataset = test_data[:, 2]

        return train_data_idx, train_dataset, test_data_idx, test_dataset


    def load_diffusion_data(self):
        """load the diffusion training data and split the pos and neg"""

        diffusion_graph = torch.load(f"../../data/cotton-data/{self.percent}-0-{self.times}-d_training").to(device)
        
        data_index = diffusion_graph[:, :2].T
        data_value = diffusion_graph[:, 2]

        """
        diff_pos_edge_index = data_index[:, data_value > 0]
        diff_neg_edge_index = data_index[:, data_value < 0]

        return diff_pos_edge_index, diff_neg_edge_index
        """
        
        return data_index, data_value


class Triad:
    """save (src_reidx, dst_reidx, [re]sign) size=(m, 3)"""
    
    def __init__(self, percent, times, seed = 114514, p_value = 0.05, day = 0) -> None:
        """
        percent: 30 50 70 80 100
        p_value: relevance select constant
        """

        print(f"Triad percent: {percent}; day: {day}; times: {times} Start!")

        # seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # file name
        self.percent = percent
        self.times = times
        self.seed = seed
        self.p_value = p_value
        self.day = day
        self.file_list = [
            f"../data/cotton-data/ori-data/{percent}%/correlation_results_FE_0913.csv",
            f"../data/cotton-data/ori-data/{percent}%/correlation_results_FL_0913.csv",
            f"../data/cotton-data/ori-data/{percent}%/correlation_results_FS_0913.csv",
            f"../data/cotton-data/ori-data/{percent}%/correlation_results_FU_0913.csv"
        ]

        # load file
        self.cor_data = torch.tensor([pd.read_csv(file).fillna(p_value)[f"CF{day/4}"] for file in self.file_list]).T.to(device)
        self.p_data = torch.tensor([pd.read_csv(file).fillna(p_value)[f"P{day/4}"] for file in self.file_list]).T.to(device)


    def getCor(self):
        return self.cor_data


    def select(self) -> Tensor:
        """
        relevance -> p_data < p_value
        return triad tensor(start, end, cor_data) size=(m, 3)
        """
        idx = torch.where(self.p_data < self.p_value)  # ph idx 0 1 2 3 not +70199 yet
        data = self.cor_data[idx]

        data[data < 0] = -1
        data[data > 0] = 1

        dataset = torch.cat((idx[0].reshape(-1, 1), idx[1].reshape(-1, 1), data.reshape(-1, 1)), dim=1)  # shape (m, 3)

        return dataset


    def split_data(self):
        """
        1. 80% train 20% test
        2. test use "100% data" format
        """
        dataset = self.select()

        # shuffle
        shuffle_idx = torch.randperm(dataset.shape[0])
        dataset = dataset[shuffle_idx]
        
        # split
        edge_nums = dataset.shape[0]
        train_data = dataset[: int(edge_nums*0.8)].long()
        train_data[:, 1] += 70199  # reindex ph idx  0 1 2 3 -> 70199 70200 70201 70202
        test_data = dataset[int(edge_nums*0.8):].long()

        # format test data
        if self.percent != 100:

            all_cor_data = Triad(100, self.times).getCor()

            # the gene & ph idx order are the same
            all_cor_data_selected = all_cor_data[(test_data[:, 0], test_data[:, 1])]
            all_cor_data_selected[all_cor_data_selected < 0] = -1
            all_cor_data_selected[all_cor_data_selected > 0] = 1

            # format
            test_data = torch.cat((test_data[:, :2], all_cor_data_selected.reshape(-1, 1).long()), dim=1)

        test_data[:, 1] += 70199  # reindex ph idx  0 1 2 3 -> 70199 70200 70201 70202

        return train_data, test_data


    def gene_id_reidx(self):
        """
        convert a disjointed node id to a consecutive node idx
        save:
        1. reidx train_data, test_data
        2. reidx feature
        """

        train_data, test_data = self.split_data()

        # create reidx dict through training set
        node_id_involved = torch.unique(train_data[:, :2])
        all_node_id_involved = torch.unique(torch.concat((train_data, test_data), dim=0)[:, :2])
        
        # gene id: train reidx ; train reidx: gene id
        id_train_dict = {gene_id.item(): reidx for reidx, gene_id in enumerate(node_id_involved)}
        # train_id_dict = {reidx: gene_id.item() for reidx, gene_id in enumerate(node_id_involved)}

        # gene id: all reidx
        id_all_dict = {gene_id.item(): reidx for reidx, gene_id in enumerate(all_node_id_involved)}

        # change the test set to contain only nodes related to the training set
        # list comprehension is faster than torch.isin(test_data[:, 0], node_id_involved) :D
        mask = [idx for idx, src in enumerate(test_data[:, 0]) if id_train_dict.get(src.item())]
        test_data = test_data[mask]

        # reidx the training & test set
        for row in range(train_data.shape[0]):
            train_data[row, 0] = id_train_dict.get(train_data[row, 0].item())
            train_data[row, 1] = id_train_dict.get(train_data[row, 1].item())

        for row in range(test_data.shape[0]):
            test_data[row, 0] = id_train_dict.get(test_data[row, 0].item())
            test_data[row, 1] = id_train_dict.get(test_data[row, 1].item())

        # load similarity adjacency matrix
        if not os.path.exists(f"../data/cotton-data/g-g-{self.percent}.csv"):
            print(f"percent {self.percent}'s similarity adjacency matrix not exists, start generating...")
            sim_adjmat = SGNNMD_generator(self.percent, self.seed, self.p_value, self.day).similarityAdj()
        else:
            sim_adjmat = DataLoad(self.percent, self.times).load_gene_simMat()

        sim_adjmat = torch.tensor(sim_adjmat)

        # add the phenotype feature (one-hot)
        node_num = sim_adjmat.shape[0]
        x = torch.eye(node_num+4)
        x[: node_num, : node_num] = sim_adjmat[: node_num+1]

        # select the node appear in the training set
        train_reidx_in_all = [id_all_dict.get(gene_id.item()) for gene_id in node_id_involved]
        x = x[train_reidx_in_all]

        # save data as .txt
        np.savetxt(f"../data/cotton-data/{self.percent}-{self.times}_feature.txt", x.numpy(), delimiter="\t", fmt="%.2f")

        # save data as .txt
        np.savetxt(f"../data/cotton-data/{self.percent}-{self.day}-{self.times}_training.txt", train_data.cpu().numpy(), delimiter="\t", fmt="%d")
        np.savetxt(f"../data/cotton-data/{self.percent}-{self.day}-{self.times}_test.txt", test_data.cpu().numpy(), delimiter="\t", fmt="%d")

        return f"percent: {self.percent}; day: {self.day}; times: {self.times} Done!"


class SGNNMD_generator:
    """
    1. up-down adj matrix
    2. similarity triad -> sim adj matrix ( gene )
    3. ph sim adj matrix ( one-hot )
    """

    def __init__(self, percent, seed = 114514, p_value = 0.05, day = 0) -> None:
        """
        percent: 30 50 70 80 100
        p_value: relevance select constant
        """

        print(f"SGNNMD data generator Start! percent: {percent}")

        # seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # file name
        self.percent = percent
        self.file_list = [
            f"../data/cotton-data/ori-data/{percent}%/correlation_results_FE_0913.csv",
            f"../data/cotton-data/ori-data/{percent}%/correlation_results_FL_0913.csv",
            f"../data/cotton-data/ori-data/{percent}%/correlation_results_FS_0913.csv",
            f"../data/cotton-data/ori-data/{percent}%/correlation_results_FU_0913.csv"
        ]

        # gene id -> gene idx dict ( start with Ghir_A01G000010: 0 )
        gene_name = pd.read_csv(self.file_list[0])["Gene"]
        self.gene_dict = {name: idx for idx, name in enumerate(gene_name)}  # len=n

        # cor data
        selected_cor_data = Triad(percent, None).select()
        self.data_idx = selected_cor_data[:, :2].T  # size=(2, m) (start, end)
        self.dataset = selected_cor_data[:, 2]  # size=(m) (value)

        self.data_idx_unique = torch.unique(self.data_idx[0])  # all relevance start idx in non-decrease order
        self.gene_nums = self.data_idx_unique.shape[0]


    def binary_select(self, arr, target):
        """Binary search, return index of target"""
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = low + (high - low) / 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1


    def upDownAdj(self):

        print("up down adjacency matrix...")

        up_down_adjmat = torch.zeros((self.gene_nums, 4))

        for col_idx in range(self.data_idx.shape[1]):

            src = self.binary_select(self.data_idx_unique, self.data_idx[0, col_idx])
            dst = self.data_idx[1, col_idx].int().item()

            up_down_adjmat[src, dst] = 1 if self.dataset[col_idx] > 0 else -1

        return up_down_adjmat


    def similarityAdj(self):

        print("similarity adjacency matrix...")

        gene_sim_data = pd.read_table("../data/cotton-data/ori-data/out_file.", header=None, usecols=(0, 1, 2))
        triad_data = torch.tensor([(self.gene_dict.get(each_triad[1]), self.gene_dict.get(each_triad[2]), each_triad[3]) for each_triad in gene_sim_data.itertuples()]).T.to(device)  # size=(3, n)

        sim_adjmat = torch.zeros((self.gene_nums, self.gene_nums))

        for rele_start_gene_reidx, rele_start_gene_idx in enumerate(self.data_idx_unique):

            # reidx: idx in m ; idx: idx in n
            col_idx = torch.where(triad_data[0] == rele_start_gene_idx)[0]  # col idx in n
            all_end_idx = triad_data[1, col_idx]  # edge: idx -> all_end_idx

            for i, rele_end_gene_idx in enumerate(all_end_idx):

                # i: i-th in col_idx; rele_end_gene_idx: idx in n
                rele_end_gene_reidx = self.binary_select(self.data_idx_unique, rele_end_gene_idx)  # reidx in m

                if rele_end_gene_reidx != None:
                    # == None -> not relevance
                    sim_adjmat[rele_start_gene_reidx, rele_end_gene_reidx] = triad_data[2, col_idx[i]] / 100
                    sim_adjmat[rele_end_gene_reidx, rele_start_gene_reidx] = triad_data[2, col_idx[i]] / 100

        return sim_adjmat


    def generate(self):

        """
        up_down_adjmat = self.upDownAdj()
        np.savetxt(f"../data/cotton-data/g-p-{self.percent}.csv", up_down_adjmat.numpy(), fmt="%d", delimiter=",")
        """

        sim_adjmat = self.similarityAdj()
        np.savetxt(f"../data/cotton-data/g-g-{self.percent}.csv", sim_adjmat.numpy(), fmt="%2f", delimiter=",")

        return f"SGNNMD data generator Done! percent: {self.percent}"


percent_list = [30, 50, 60, 70, 80, 100]
seed_list = [114, 514, 1919, 810, 721]

if __name__ == "__main__":
    """
    # adj mat
    for percent in percent_list:
        print(SGNNMD_generator(percent, 0).generate())
    """

    # triad
    for percent in percent_list:
        for times in range(5):
            print(Triad(percent, times+1, seed=seed_list[times]).gene_id_reidx())

