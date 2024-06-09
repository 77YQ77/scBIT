import os
import argparse
import torch
import pandas as pd
import numpy as np
from models import GnnNets
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args
from tqdm import tqdm

def get_embedding(id):
    print('start loading data====================')
    dataset = get_dataset(data_args.dataset_dir, 'BrainCellDataset', 'train', id=id, task=data_args.task)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    dataloader = get_dataloader(dataset, train_args.batch_size, data_split_ratio=[1, 0, 0])

    gnnNets = GnnNets(input_dim, output_dim, model_args)

    ckpt_dir = f'./checkpoint/human_X{id}'
    checkpoint = torch.load(os.path.join(ckpt_dir, 'gin_best.pth'))
    gnnNets.update_state_dict(checkpoint['net'])
    gnnNets.eval()
    gnnNets.to_device()

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]
    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index / 2 :.4f}")

    label_name = ['Astrocyte', 'L2/3 IT', 'L4 IT', 'L5', 'L5/6 NP', 'L6', 'Lamp5', 'Lamp5 Lhx6', 'Microglia-PVM', 'OPC',
                  'Oligodendrocyte', 'Pvalb', 'Sncg', 'Sst', 'Vip']

    embeddings = []
    labels = []
    for batch in tqdm(dataloader['train']):
        batch = batch.to('cuda:0')
        _, _, _, graph_emb, _, _ = gnnNets(batch)
        embeddings.append(graph_emb.detach().cpu())
        y=batch.y.detach().cpu().tolist()
        labels.extend(y)

    indexes=[]
    for label in labels:
        indexes.append(label_name[label])
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.numpy()
    pd.DataFrame(embeddings,index=indexes).to_csv(f'./datasets/embeddings/{id}.csv')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', type=int, default=0,help='brain dataset clips id')
    args = parser.parse_args()

    get_embedding(args.id)