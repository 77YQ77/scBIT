import torch
import pandas as pd
from distillation_model import ProjectModel
import numpy as np

data = pd.read_csv('../prototype_vectors.csv', header=None)

model = ProjectModel(2865).cuda()
checkpoint = torch.load('../checkpoint/distillation_best.pth')
model.load_state_dict(checkpoint['net'])
model.eval()

batch_size=90

matrix_name = list(data.iloc[:, 0])
cell_type = list(data.iloc[:, 1])
data = data.iloc[:, 2:].to_numpy(dtype=np.float32)


for i in range(0, data.shape[0]//batch_size):
    # 检查90个的名字是否一样
    names = matrix_name[i * batch_size:(i + 1) * batch_size]
    assert len(set(names)) == 1

    batch = data[i * batch_size:(i + 1) * batch_size,:]
    batch=torch.from_numpy(batch).to('cuda:0')

    graphs=model(batch)
    # todo:存图
