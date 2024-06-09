import collections
import os
import glob
import json
import pandas as pd
import torch
import pickle
import random
import numpy as np
import os.path as osp
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, vstack, save_npz
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from Configures import data_args

os.environ["OMP_NUM_THREADS"] = '1'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
random.seed(42)


def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data


def split(data, batch):
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graphs.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices


def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f'_{name}.txt')
    return np.genfromtxt(file_path, dtype=np.int64)


def get_id_2_gene(species_data_path, species, tissue, filetype,id):
    data_path = species_data_path
    data_files = data_path.glob(f'human_brain{id}_data.{filetype}')
    gene = None
    for file in data_files:
        if filetype == 'csv':
            data = pd.read_csv(file, dtype='str', header=0).values[:, 0]
        else:
            data = pd.read_csv(file, compression='gzip', header=0).values[:, 0]  # data是['A1BG'  ...  'ZUP1']这样的基因名称

        if gene is None:
            gene = set(data)
        else:
            gene = gene | set(data)

    id2gene = list(gene)
    id2gene.sort()

    return id2gene


def get_id_2_label_and_label_statistics(species_data_path, species, tissue, id=0):
    if id == 0:
        data_path = species_data_path
        cell_files = data_path.glob(f'{species}_{tissue}*_celltype.csv')
    else:
        data_path = species_data_path
        cell_files = data_path.glob(f'human_brain{id}_celltype.csv')

    cell_types = set()
    cell_type_list = list()

    for file in cell_files:
        df = pd.read_csv(file, dtype=str, header=0)
        df['Cell_type'] = df['Cell_type'].map(str.strip)  # 删除df中Cell_type列中的每个字符串的前导和尾随空格
        cell_types = set(df.values[:, 2]) | cell_types
        cell_type_list.extend(df.values[:, 2].tolist())
    id2label = list(cell_types)
    label_statistics = dict(collections.Counter(cell_type_list))
    return id2label, label_statistics


def save_statistics(statistics_path, id2label, id2gene, tissue):
    gene_path = statistics_path / f'{tissue}_genes.txt'
    label_path = statistics_path / f'{tissue}_label.txt'
    with open(gene_path, 'w', encoding='utf-8') as f:
        for gene in id2gene:
            f.write(gene + '\r\n')
    with open(label_path, 'w', encoding='utf-8') as f:
        for label in id2label:
            f.write(label + '\r\n')


def label_classification(all_labels, num_labels):
    label_classes = [[] for _ in range(num_labels)]
    for idx, label in enumerate(all_labels):
        label_classes[int(label)].append(idx)
    return label_classes


def make_single_graph(data, pearson_threshold, label):  # pearson
    # data : (num_cells x genes)
    data = data.T
    adj = torch.corrcoef(data)
    torch.diagonal(adj, 0).zero_()
    split = torch.abs(adj) > pearson_threshold
    adj[~split] = 0
    edges = torch.nonzero(adj)
    return Data(x=data.to(torch.float), edge_index=edges.long().T, y=label)


def make_graph(data, label_classes, n, pearson_threshold):
    graphs = []
    for i in range(len(label_classes)):
        L = len(label_classes[i])
        for j in range(0, L, n):
            slices = label_classes[i][j:j + n]
            if (len(slices) != n):
                continue
            graphs.append(make_single_graph(data[slices, :], pearson_threshold, i))
            if L - j < 2 * n:
                break

    return graphs


def get_dataset(dataset_dir, dataset_name, mode, id, task=None):
    if dataset_name == 'BrainCellDataset':
        return BrainCellDataset(dataset_dir, mode, id)
    else:
        raise NotImplementedError


class BrainCellDataset(InMemoryDataset):
    def __init__(self, root, mode, id, transform=None, pre_transform=None):
        self.root = root
        self.mode = mode  # train / predict
        self.id = id
        super(BrainCellDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices['x']) - 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'BrainCellDataset', self.mode)

    @property
    def raw_file_names(self):
        return [' ']

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'BrainCellDataset', 'pretrained', 'graphs')

    @property
    def processed_file_names(self):
        return [f'{data_args.species}_brain{self.id}_{data_args.num_cells}.pt']

    def process(self):
        species = data_args.species
        tissue = data_args.tissue

        species_data_path = Path(f'./datasets/BrainCellDataset/train/')  # 训练用单细胞表达数据路径
        graph_path = Path(f'./datasets/BrainCellDataset/pretrained/graphs')  # 训练时保存的图路径

        if not species_data_path.exists():
            raise NotImplementedError

        if not graph_path.exists():
            graph_path.mkdir(parents=True)

        # 生成基因统计文件
        id2gene = get_id_2_gene(species_data_path, species, tissue, filetype=data_args.filetype,id=self.id)
        # 生成细胞标签统计文件
        id2label, label_statistics = get_id_2_label_and_label_statistics(species_data_path, species, tissue, self.id)
        id2label = sorted(id2label)
        print(id2label)

        # 准备统一的基因
        gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
        num_genes = len(id2gene)  # 基因的数量

        # 准备统一的细胞标签
        num_labels = len(id2label)  # 标签的数量
        label2id = {label: idx for idx, label in enumerate(id2label)}  # 将类型列表转换为字典 label2id，字典的键是类型名称，值是类型在列表中的索引

        print(f"The built graph contains {num_genes} genes with {num_labels} labels supported.")

        all_labels = []
        matrices = []
        num_cells = 0

        data_path = species_data_path
        data_files = data_path.glob(f'{data_args.species}_brain{self.id}_data.{data_args.filetype}')

        for data_file in data_files:
            number = ''.join(list(filter(str.isdigit, data_file.name)))
            type_file = species_data_path / f'{data_args.species}_brain{self.id}_celltype.csv'
            cell2type = pd.read_csv(type_file, index_col=0)
            cell2type.columns = ['cell', 'type']
            cell2type['type'] = cell2type['type'].map(str.strip)  # 去除字符串两端的空白字符
            cell2type['id'] = cell2type['type'].map(label2id)  # 将细胞类型映射为数字

            filter_cell = np.where(pd.isnull(cell2type['id']) == False)[0]
            cell2type = cell2type.iloc[filter_cell]

            assert not cell2type['id'].isnull().any(), 'something wrong about celltype file.'
            all_labels += cell2type['id'].tolist()

            if data_args.filetype == 'csv':
                df = pd.read_csv(data_file, index_col=0)  # (gene, cell)
            elif data_args.filetype == 'gz':
                df = pd.read_csv(data_file, compression='gzip', index_col=0)
            else:
                print(f'Not supported type for {data_path}. Please verify your data file')

            df = df.transpose(copy=True)
            df = df.iloc[filter_cell]

            df = df.rename(columns=gene2id)
            col = [c for c in df.columns if c in gene2id.values()]
            df = df[col]
            print(
                f'{data_args.species}_{tissue}{number}_data.{data_args.filetype} -> Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')

            arr = df.to_numpy()
            row_idx, col_idx = np.nonzero(arr > data_args.threshold)
            non_zeros = arr[(row_idx, col_idx)]
            gene_idx = df.columns[col_idx].astype(int).tolist()
            info_shape = (len(df), num_genes)
            info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
            matrices.append(info)
            num_cells += len(df)

        assert len(all_labels) == num_cells

        # 2. create features
        sparse_feat = vstack(matrices).toarray()
        assert sparse_feat.shape[0] == num_cells

        # 归一化
        cell_feat = torch.from_numpy(sparse_feat)  # cells x genes
        cell_feat = cell_feat / (torch.sum(cell_feat, dim=1, keepdims=True) + 1e-6)

        label_classes = label_classification(all_labels, num_labels)

        graphs = make_graph(cell_feat, label_classes, data_args.num_cells, data_args.pearson_threshold)
        random.shuffle(graphs)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=5):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(seed))

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=False)
    dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloader
