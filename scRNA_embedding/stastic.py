import torch
import os
import argparse
import pandas as pd
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract states from saved models')
    parser.add_argument('--id', type=int, default=1, help='brain dataset clips id')
    args = parser.parse_args()

    ckpt_dir = r'./checkpoint'
    model = torch.load(os.path.join(ckpt_dir, f'human_X{args.id}/gin_best.pth'))

    # print(model['acc'])

    st_acc = f'human_X{args.id},{model["acc"]}\n'
    with open('./acc.csv', 'a') as f:
        f.write(st_acc)

    x = ['Astrocyte', 'L2/3 IT', 'L4 IT', 'L5', 'L5/6 NP', 'L6', 'Lamp5', 'Lamp5 Lhx6', 'Microglia-PVM', 'OPC',
         'Oligodendrocyte', 'Pvalb', 'Sncg', 'Sst', 'Vip']


    for idx, vec in enumerate(model['net']['model.prototype_vectors']):
        l = x[idx // 6]

        tissue_label = f'human_X{args.id},{l},'
        # print(st_vec)
        vec_list = vec.numpy().tolist()

        with open('./prototype_vectors.csv', 'a',newline='') as f:
            f.write(tissue_label)
            writer = csv.writer(f)
            writer.writerow(vec_list)

