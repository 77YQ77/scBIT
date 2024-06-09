import os
import shutil

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ProjectDataset(Dataset):
    def __init__(self):
        self.data_path = '../datasets/adj-emb pairs/'
        self.id = 0
        self.adjs = None
        self.embs = None

    def __getitem__(self, idx):
        id = idx // 32 + 1
        if id != self.id:
            self.id = id
            self.adjs = torch.load(os.path.join(self.data_path, f'{id}_adj.pth')).to_dense()

            self.embs = torch.load(os.path.join(self.data_path, f'{id}_emb.pth'))
        return self.adjs[idx % 32], self.embs[idx % 32]

    def __len__(self):
        return 32 * 171


class ProjectModel_vae(nn.Module):
    def __init__(self, num_node, latent_dim):
        super(ProjectModel_vae, self).__init__()
        self.transform = nn.Linear(128, num_node)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        self.latent_dim = latent_dim

        # 均值和方差线性层
        self.fc_mu = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(32 * 7 * 7, latent_dim)

    def encode(self, x):
        x = self.encoder(x)
        print(x.shape)
        exit()
        x = x.view(x.size(0), -1)
        print(x.shape)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        x = self.decoder(z)
        return x

    def forward(self, embs):
        embs = self.transform(embs)
        emb2d = torch.bmm(embs.reshape(embs.shape[0], -1, 1), embs.reshape(embs.shape[0], 1, -1))
        emb2d = emb2d.unsqueeze(1)
        mu, logvar = self.encode(emb2d)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class ProjectModel(nn.Module):
    def __init__(self, num_node):
        super(ProjectModel, self).__init__()
        self.transform = nn.Linear(128, num_node,bias=False)

    def forward(self, embs):
        embs = self.transform(embs)
        emb2d = torch.bmm(embs.reshape(embs.shape[0], -1, 1), embs.reshape(embs.shape[0], 1, -1))
        return emb2d


class SparseGeneratorLoss(nn.Module):
    def __init__(self, l1_weight):
        super(SparseGeneratorLoss, self).__init__()
        self.l1_weight = l1_weight
        self.criterion = nn.L1Loss()

    def forward(self, generated_matrix, target_matrix):
        l1_loss = torch.norm(generated_matrix, p=1)
        reconstruction_loss = self.criterion(generated_matrix, target_matrix)
        total_loss = reconstruction_loss + self.l1_weight * l1_loss
        return total_loss


def save_best(ckpt_dir, gnnNets, model_name, is_best):
    print('saving....')
    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
    }
    pth_name = f"{model_name}_latest.pth"
    best_pth_name = f'{model_name}_best.pth'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
    gnnNets.to('cuda:0')

if __name__=='__main__':
    data = ProjectDataset()
    train_loader = DataLoader(data, batch_size=32)
    criterion = SparseGeneratorLoss(0.00)

    input_dim = 784
    hidden_dim = 256
    latent_dim = 32
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = ProjectModel(2865).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    num_epochs = 50
    pre_loss = 0
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (adjs, embs) in enumerate(train_loader):
            # print(batch_idx)
            adjs = adjs.to(device)
            embs = embs.to(device)

            ret_data = model(embs)
            loss = criterion(ret_data, adjs)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if train_loss < pre_loss or epoch % 10 == 0:
            save_best('../checkpoint', model, 'distillation', train_loss < pre_loss)
        pre_loss = train_loss

        print(f'Epoch {epoch}, Loss: {train_loss.item() / len(train_loader)}')
