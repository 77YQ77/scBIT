import os
import argparse
import random
import time
import torch
import torch.nn.functional as F
import shutil
import numpy as np
import threading
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from models import GnnNets, GnnNets_NC
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args
from my_mcts import mcts
from tqdm import tqdm
from multiprocessing import Pool, Manager
import matplotlib.pyplot as plt
from pathlib import Path

random.seed(42)

def Project(id, dataset_name):
    print('start loading data====================')
    dataset = get_dataset(data_args.dataset_dir, dataset_name, 'train', id, task=data_args.task)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    dataloader = get_dataloader(dataset, train_args.batch_size, data_split_ratio=data_args.data_split_ratio)

    print('start projecting==================')
    gnnNets = GnnNets(input_dim, output_dim, model_args)
    if id == 0:
        ckpt_dir = f"./checkpoint/{data_args.species}_{data_args.tissue}/"
    else:
        ckpt_dir = f"./checkpoint/human_X{id}/"

    checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best.pth'))
    gnnNets.update_state_dict(checkpoint['net'])

    gnnNets.to_device()

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]
    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index / 2 :.4f}")

    data_size = len(dataset)
    print(f'The total num of dataset is {data_size}')

    data_indices = dataloader['train'].dataset.indices
    random.shuffle(data_indices)

    gnnNets.eval()
    for i in range(0, len(data_indices)):
        data = dataset[data_indices[i]]
        prototype_nums = gnnNets.model.prototype_shape[0]
        j = random.randint(0, prototype_nums - 1)
        ok_flag = mcts(data, gnnNets, gnnNets.model.prototype_vectors[j], id)
        if ok_flag == True:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--clst', type=float, default=0.2,
                        help='cluster')
    parser.add_argument('--sep', type=float, default=0.05,
                        help='separation')
    parser.add_argument('--id', type=int, default=1,
                        help='brain dataset clips id')
    parser.add_argument('--dataset', type=str, default='BrainCellDataset',
                        help='dataset')
    args = parser.parse_args()

    path=Path(f'./datasets/adj-emb pairs/{args.id}_adj.pth')
    if path.exists():
        exit()
    st_time = time.time()
    Project(args.id, args.dataset)
    ed_time = time.time()
    x=ed_time-st_time
    print(x)
