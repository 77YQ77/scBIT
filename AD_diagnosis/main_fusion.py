import warnings
warnings.filterwarnings("ignore")
import scipy.io as scio
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sklearn.metrics as metrics
from model import encoder_regression,encoder,Transform_fusion
from sklearn.metrics import confusion_matrix,roc_auc_score
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
from sklearn.model_selection import StratifiedKFold
import random
from contrastive_loss import SupConLoss
criterion_mse=nn.MSELoss()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
clu_criterion = SupConLoss()

class extra_data(Dataset):
    def  __init__(self,data,label, patient_label, transforms=None):
        self.D_client0 = data
        self.D_label = label
        self.cell_info = patient_label
        self.transforms = transforms
    def __getitem__(self, index):
        img_0 = self.D_client0[index]
        label = self.D_label[index]
        cell_info_0 = self.cell_info[index]
        return (img_0, label,cell_info_0)
    def __len__(self):
        return len(self.D_client0)
    
class client_datas(Dataset):
    def  __init__(self,data,label, transforms=None):
        self.D_client0 = data
        self.D_label = label
        self.transforms = transforms
    def __getitem__(self, index):
        img_0 = self.D_client0[index]
        label = self.D_label[index]
        return (img_0, label)
    def __len__(self):
        return len(self.D_client0)

class client_data(Dataset):
    def  __init__(self,data,label, age_cell_info, gene_cell_info, sex_cell_info, group_cell_info, transforms=None):
        self.D_client0 = data
        self.D_label = label
        self.age_cell_info = age_cell_info
        self.gene_cell_info = gene_cell_info
        self.sex_cell_info = sex_cell_info
        self.group_cell_info = group_cell_info
        self.transforms = transforms
    def __getitem__(self, index):
        img_0 = self.D_client0[index]
        label = self.D_label[index]
        age_cell_info_0 = self.age_cell_info[index]
        sex_cell_info_0 = self.sex_cell_info[index]
        group_cell_info_0 = self.group_cell_info[index]
        gene_cell_info_0 = self.gene_cell_info[index]
        return (img_0, label,age_cell_info_0,sex_cell_info_0,group_cell_info_0,gene_cell_info_0)
    def __len__(self):
        return len(self.D_client0)


class client_data_aug(Dataset):
    def  __init__(self,data, aug_data,label, pheno, transforms=None):
        self.D_client0 = data
        self.D_client1 = aug_data
        self.D_label = label
        self.pheno = pheno
        self.transforms = transforms
    def __getitem__(self, index):
        img_0 = self.D_client0[index]
        img_1 = self.D_client1[index]
        label = self.D_label[index]
        pheno_0 = self.pheno[index]
        return (img_0, img_1, label,pheno_0)
    def __len__(self):
        return len(self.D_client1)

def train_encoder_regression(num,encoder_model, cell_tensor, data_loader, criterion_mse , optimizer):
    for i in range(num):
        for _, (data,aug_data,_, label) in enumerate (data_loader):
            data=data.type(torch.FloatTensor).to(device)
            labels=label.type(torch.FloatTensor).to(device)
            similar = encoder_model(data,cell_tensor).to(device)
            sim = similar.mean(3).mean(2)
            loss = criterion_mse(sim,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return encoder_model

def train_encoder_contrastive(num,encoder_model, cell_tensor, data_loader, clu_criterion , optimizer):
    for i in range(num):
        for _, (data,aug_data,_, label) in enumerate (data_loader):
            data=data.type(torch.FloatTensor).to(device)
            aug_data=aug_data.type(torch.FloatTensor).to(device)
            label_t = label.view(data.shape[0]*171)
            labels=label_t.type(torch.FloatTensor).to(device)
            images = torch.cat([data, aug_data], dim=0)
            labels_ = torch.cat([labels, labels], dim=0)
            images = images.cuda()
            labels_=labels_.cuda()
            _,__,similar = encoder_model(images,cell_tensor)
            bsz = label_t.shape[0]
            f1, f2 = torch.split(similar, [bsz, bsz], dim=0) # 64*171
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = clu_criterion(features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return encoder_model


def generate_augdata(data):
    a = np.random.randn(data.shape[0], data.shape[1], data.shape[2]) / 5
    aug_data=data+a
    aug_data=torch.tensor(aug_data)
    return aug_data

def generate_agu_tensor(X_train_ASD_s1,y_train_ASD_s1,pheno_ASD_s1):
    X_train_ASD_s1 = X_train_ASD_s1.type(torch.FloatTensor)
    y_train_ASD_s1 = y_train_ASD_s1.type(torch.FloatTensor)
    pheno_ASD_s1 = pheno_ASD_s1.type(torch.FloatTensor)
    data_tensor_s1 = generate_augdata(X_train_ASD_s1)
    feature_tensor_s1 = torch.Tensor(X_train_ASD_s1)
    label_tensor_s1 = torch.Tensor(y_train_ASD_s1)
    dataset_tensor = client_data_aug(feature_tensor_s1, data_tensor_s1, label_tensor_s1,pheno_ASD_s1)
    return dataset_tensor

def test(testLoader, model, device):
    model.to(device)
    model.eval()
    with torch.no_grad(): # when in test stage, no grad
        pred_list = []
        label_list = []
#         for (imgs,_, labels) in testLoader:
        for _, (imgs,labels,age_similarit, gene_similarit, sex_similarit, group_similarit) in enumerate (testLoader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            age_similarit = age_similarit.type(torch.FloatTensor).to(device)
            gene_similarit = gene_similarit.type(torch.FloatTensor).to(device)
            sex_similarit = sex_similarit.type(torch.FloatTensor).to(device)
            group_similarit = group_similarit.type(torch.FloatTensor).to(device)
            out  = model(imgs,age_similarit,sex_similarit,gene_similarit,group_similarit)
            _,pre= torch.max(out.data, 1)
#             _,lab= torch.max(labels.data, 1)
            for i in range(len(pre)):
                pred_list.append(pre[i].cpu().item())
                label_list.append(labels[i].cpu().item())
    confusion = confusion_matrix(pred_list, label_list)
    TP_num = confusion[1, 1]
    TN_num = confusion[0, 0]
    FP_num = confusion[0, 1]
    FN_num = confusion[1, 0]

    if float(TN_num + FP_num) == 0:
        SPE = 0
    else:
        SPE = TN_num / float(TN_num + FP_num)
    if float(TP_num + FN_num) == 0:
        REC = 0
    else:
        REC = TP_num / float(TP_num + FN_num)
    AUC = roc_auc_score(label_list, pred_list)
    return  metrics.accuracy_score(pred_list, label_list),SPE, REC, AUC


def data_loader_build(age_encoder_model, sex_encoder_model, gene_encoder_model, group_encoder_model,cell_tensor, data_loader):
    for idx, (data, patient_label) in enumerate (data_loader): # data,aug_data,_, label
        data=data.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            age_similar = age_encoder_model(data,cell_tensor)
            _, sex_similar,_ = sex_encoder_model(data,cell_tensor)
            gene_similar = gene_encoder_model(data,cell_tensor)
            _,group_similar,_ = group_encoder_model(data,cell_tensor)
            age_similar = age_similar.cpu()
            sex_similar = sex_similar.cpu()
            gene_similar = gene_similar.cpu()
            group_similar = group_similar.cpu()
            group_sim = group_similar.view(-1,171,116,90)
            group_sim = group_sim.mean(3).mean(2)
            sex_sim = sex_similar.view(-1,171,116,90)
            sex_sim = sex_sim.mean(3).mean(2)

            age_sim = age_similar.view(-1,171,116,90)
            age_sim = age_sim.mean(3).mean(2)
            gene_sim = gene_similar.view(-1,171,116,90)
            gene_sim = gene_sim.mean(3).mean(2)

            if idx ==0:
                age_similarit = age_sim.cpu()
                gene_similarit = gene_sim.cpu()
                sex_similarit = sex_sim.cpu()
                group_similarit = group_sim.cpu()
                da = data.cpu()
                pa = patient_label.cpu()
            else:
                age_similar = age_encoder_model(data,cell_tensor)
                _, sex_similar,_ = sex_encoder_model(data,cell_tensor)
                gene_similar = gene_encoder_model(data,cell_tensor)
                _,group_similar,_ = group_encoder_model(data,cell_tensor)
                group_sim = group_similar.view(-1,171,116,90).cpu()
                group_sim = group_sim.mean(3).mean(2)
                sex_sim = sex_similar.view(-1,171,116,90).cpu()
                sex_sim = sex_sim.mean(3).mean(2)
                age_sim = age_similar.view(-1,171,116,90).cpu()
                age_sim = age_sim.mean(3).mean(2)
                gene_sim = gene_similar.view(-1,171,116,90).cpu()
                gene_sim = gene_sim.mean(3).mean(2)
                age_similarit = torch.cat((age_similarit,age_sim),dim=0)
                gene_similarit = torch.cat((gene_similarit,gene_sim),dim=0)
                sex_similarit = torch.cat((sex_similarit,sex_sim),dim=0)
                group_similarit = torch.cat((group_similarit,group_sim),dim=0)
                da = torch.cat((da,data.cpu()),dim=0)
                pa = torch.cat((pa,patient_label.cpu()),dim=0)


    train_tensor = client_data(da, pa, age_similarit, gene_similarit, sex_similarit, group_similarit)
    train_loader = DataLoader(train_tensor,batch_size=128,shuffle=True)
    return train_loader

def test_data_loader_build(age_encoder_model, sex_encoder_model, gene_encoder_model, group_encoder_model, cell_tensor, data_loader):
    for idx, (data, patient_label) in enumerate (data_loader): # data,aug_data,_, label
        data=data.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            age_similar = age_encoder_model(data,cell_tensor)
            _, sex_similar,_ = sex_encoder_model(data,cell_tensor)
            gene_similar = gene_encoder_model(data,cell_tensor)
            _,group_similar,_ = group_encoder_model(data,cell_tensor)
            age_similar = age_similar.cpu()
            sex_similar = sex_similar.cpu()
            gene_similar = gene_similar.cpu()
            group_similar = group_similar.cpu()
            group_sim = group_similar.view(-1,171,116,90)
            group_sim = group_sim.mean(3).mean(2)
            sex_sim = sex_similar.view(-1,171,116,90)
            sex_sim = sex_sim.mean(3).mean(2)
            age_sim = age_similar.view(-1,171,116,90)
            age_sim = age_sim.mean(3).mean(2)
            gene_sim = gene_similar.view(-1,171,116,90)
            gene_sim = gene_sim.mean(3).mean(2)

            if idx ==0:
                age_similarit = age_sim.cpu()
                gene_similarit = gene_sim.cpu()
                sex_similarit = sex_sim.cpu()
                group_similarit = group_sim.cpu()
                da = data.cpu()
                pa = patient_label.cpu()
            else:
                age_similar = age_encoder_model(data,cell_tensor)
                _, sex_similar,_ = sex_encoder_model(data,cell_tensor)
                gene_similar = gene_encoder_model(data,cell_tensor)
                _,group_similar,_ = group_encoder_model(data,cell_tensor)
                group_sim = group_similar.view(-1,171,116,90).cpu()
                group_sim = group_sim.mean(3).mean(2)
                sex_sim = sex_similar.view(-1,171,116,90).cpu()
                sex_sim = sex_sim.mean(3).mean(2)

                age_sim = age_similar.view(-1,171,116,90).cpu()
                age_sim = age_sim.mean(3).mean(2)
                gene_sim = gene_similar.view(-1,171,116,90).cpu()
                gene_sim = gene_sim.mean(3).mean(2)

                age_similarit = torch.cat((age_similarit,age_sim),dim=0)
                gene_similarit = torch.cat((gene_similarit,gene_sim),dim=0)
                sex_similarit = torch.cat((sex_similarit,sex_sim),dim=0)
                group_similarit = torch.cat((group_similarit,group_sim),dim=0)
                da = torch.cat((da,data.cpu()),dim=0)
                pa = torch.cat((pa,patient_label.cpu()),dim=0)


    train_tensor = client_data(da, pa, age_similarit, gene_similarit, sex_similarit, group_similarit)
    train_loader = DataLoader(train_tensor,batch_size=128,shuffle=True)
    return train_loader

def normalization(data):
    data_list=[]
    for i in data:
        normalized_data = (i - np.min(i)) / (np.max(i) - np.min(i)) 
        data_list.append(normalized_data)
    data_array = np.array(data_list)
    return data_array

def main():

    similarity_age = np.load('data/age_absolute_difference_matrix.npy',allow_pickle=True)
    similarity_gene = np.load('data/gene_cosine_difference_matrix.npy',allow_pickle=True)
    similarity_sex = np.load('data/sex_absolute_difference_matrix.npy',allow_pickle=True)
    similarity_group = np.load('data/group_absolute_similarity_classification.npy',allow_pickle=True)
    age_similarity = normalization(similarity_age)
    gene_similarity = normalization(similarity_gene)
    sex_similarity = similarity_sex
    group_similarity = similarity_group
    print('age_similarity',age_similarity.shape)
    print('gene_similarity',gene_similarity.shape)
    print('sex_similarity',sex_similarity.shape)
    print('group_similarity',group_similarity.shape)
    
    fmri_data = np.load('data/fmri_data_norm.npy',allow_pickle=True)
    print('fmri_data',fmri_data.shape)
    single_cell = np.load('data/single_cell.npy',allow_pickle=True)
    print('single_cell',single_cell.shape)
    patient_label = np.load('data/patient_label.npy',allow_pickle=True)
    patient_label = torch.tensor(patient_label)
    print('patient_label',patient_label.shape)
    fmri_data = torch.tensor(fmri_data)
    fmri_data = fmri_data.to(torch.float32).to(device)
    ACC_list_avg = []
    SPE_list_avg = []
    REC_list_avg = []
    AUC_list_avg = []
    for i_count in range(5):
        ACC_list = []
        SPE_list = []
        REC_list = []
        AUC_list = []
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=3 ** (i_count + 1))
        for train_index, test_index in kf.split(fmri_data, patient_label):
            x_train = fmri_data[train_index]
            y_train = patient_label[train_index]
            age_sim_train = age_similarity[train_index]
            gene_sim_train = gene_similarity[train_index]
            sex_sim_train = sex_similarity[train_index]
            group_sim_train = group_similarity[train_index]
            x_test = fmri_data[test_index]
            y_test = patient_label[test_index]

            index = [i for i in range(x_train.shape[0])]
            random.shuffle(index)
            x_train = x_train[index]
            y_train = y_train[index]
            age_sim_train = age_sim_train[index]
            gene_sim_train = gene_sim_train[index]
            sex_sim_train = sex_sim_train[index]
            group_sim_train = group_sim_train[index]
            age_sim_train = torch.tensor(age_sim_train)
            gene_sim_train = torch.tensor(gene_sim_train)
            sex_sim_train = torch.tensor(sex_sim_train)
            group_sim_train = torch.tensor(group_sim_train)
            cell_tensor = torch.tensor(single_cell).to(torch.float32)
            cell_tensor = cell_tensor.permute(0,2,1)
            sex_dataset = generate_agu_tensor(x_train, y_train,sex_sim_train)
            sex_data_loader = DataLoader(sex_dataset,batch_size=64,shuffle=True)
            sex_encoder_model = encoder(input_size=116, hidden_size=128, time_seq = fmri_data.shape[1] ,num_head=4).to(device)
            sex_encoder_model = nn.DataParallel(sex_encoder_model)
            cell_over = torch.cat([cell_tensor,cell_tensor,cell_tensor,cell_tensor],dim=0).to(device)
            sex_optimizer = optim.Adam(sex_encoder_model.parameters(), lr=0.001)
            sex_encoder_model = train_encoder_contrastive(10,sex_encoder_model, cell_over, sex_data_loader, clu_criterion , sex_optimizer)
            del sex_data_loader,sex_optimizer
            torch.cuda.empty_cache()

            group_dataset = generate_agu_tensor(x_train, y_train,group_sim_train)
            group_data_loader = DataLoader(group_dataset,batch_size=64,shuffle=True)
            group_encoder_model = encoder(input_size=116, hidden_size=128, time_seq = fmri_data.shape[1] ,num_head=4).to(device)
            group_encoder_model = nn.DataParallel(group_encoder_model)
            group_optimizer = optim.Adam(group_encoder_model.parameters(), lr=0.001)
            group_encoder_model = train_encoder_contrastive(10,group_encoder_model, cell_over, group_data_loader, clu_criterion , group_optimizer)
            del group_data_loader,group_optimizer
            torch.cuda.empty_cache()

            age_dataset = generate_agu_tensor(x_train, y_train,age_sim_train)
            age_data_loader = DataLoader(age_dataset,batch_size=64,shuffle=True)
            age_encoder_model = encoder_regression(input_size=116, hidden_size=128, time_seq = fmri_data.shape[1] ,num_head=4).to(device)
            age_encoder_model = nn.DataParallel(age_encoder_model)
            age_optimizer = optim.Adam(age_encoder_model.parameters(), lr=0.001)
            age_encoder_model = train_encoder_regression(10,age_encoder_model, cell_over, age_data_loader, criterion_mse , age_optimizer)
            del age_data_loader,age_optimizer
            torch.cuda.empty_cache()

            gene_dataset = generate_agu_tensor(x_train, y_train,gene_sim_train)
            gene_data_loader = DataLoader(gene_dataset,batch_size=64,shuffle=True)
            gene_encoder_model = encoder_regression(input_size=116, hidden_size=128, time_seq = fmri_data.shape[1] ,num_head=4).to(device)
            gene_encoder_model = nn.DataParallel(gene_encoder_model)
            gene_optimizer = optim.Adam(gene_encoder_model.parameters(), lr=0.001)
            gene_encoder_model = train_encoder_regression(10,gene_encoder_model, cell_over, gene_data_loader, criterion_mse , gene_optimizer)
            del gene_data_loader,gene_optimizer
            torch.cuda.empty_cache()

            dataset = client_datas(x_train, y_train)
            data_loader = DataLoader(dataset,batch_size=8,shuffle=True)
            train_loader = data_loader_build(age_encoder_model, sex_encoder_model, gene_encoder_model, group_encoder_model, cell_over, data_loader)

            test_dataset = client_datas(x_test, y_test) 
            testda_loader = DataLoader(test_dataset,batch_size=8,shuffle=True)
            test_loader = test_data_loader_build(age_encoder_model, sex_encoder_model, gene_encoder_model, group_encoder_model, cell_over, testda_loader)
            del age_encoder_model, sex_encoder_model, gene_encoder_model, group_encoder_model,cell_over
            torch.cuda.empty_cache()

            model = Transform_fusion(input_size=116, hidden_size=128, time_seq = 140,num_head=4, num_classes=2).to(device)
            model = nn.DataParallel(model)
            model.train()

            criterion_1 = nn.CrossEntropyLoss()
            model_optimizer = optim.Adam(model.parameters(), lr=0.0005)

            for j in range(126):
                for _, (data,labels, age_similarit, gene_similarit, sex_similarit, group_similarit) in enumerate (train_loader):
                    data=data.type(torch.FloatTensor).to(device)
                    labels=labels.type(torch.LongTensor).to(device)
                    age_similarit = age_similarit.type(torch.FloatTensor).to(device)
                    gene_similarit = gene_similarit.type(torch.FloatTensor).to(device)
                    sex_similarit = sex_similarit.type(torch.FloatTensor).to(device)
                    group_similarit = group_similarit.type(torch.FloatTensor).to(device)
                    out = model(data, age_similarit,sex_similarit,gene_similarit,group_similarit)
                    loss2 = criterion_1(out,labels)
                    model_optimizer.zero_grad()
                    loss2.backward()
                    model_optimizer.step()

            ACC,SPE, REC, AUC = test(test_loader, model, device)
            tqdm.write(f'training_round {j:01d}: test_ACC={ACC:.4f}, ' 
                                                f'test_SPE= {SPE:.4f}, '
                                                f'test_REC= {REC:.4f}, '
                                                f'test_AUC= {AUC:.4f}')
            ACC_list.append(ACC)
            SPE_list.append(SPE)
            REC_list.append(REC)
            AUC_list.append(AUC)

        ACC_array = np.mean(np.array(ACC_list))
        SPE_array = np.mean(np.array(SPE_list))
        REC_array = np.mean(np.array(REC_list))
        AUC_array = np.mean(np.array(AUC_list))
        tqdm.write(f'CV_number {i_count:01d}: cv_ACC={ACC_array:.4f}, ' 
                                                    f'cv_SPE= {SPE_array:.4f}, '
                                                    f'cv_REC= {REC_array:.4f}, '
                                                    f'cv_AUC= {AUC_array:.4f}')
        ACC_list_avg.append(ACC_array)
        SPE_list_avg.append(SPE_array)
        REC_list_avg.append(REC_array)
        AUC_list_avg.append(AUC_array)
    ACC_array_avg = np.mean(np.array(ACC_list_avg))
    SPE_array_avg = np.mean(np.array(SPE_list_avg))
    REC_array_avg = np.mean(np.array(REC_list_avg))
    AUC_array_avg = np.mean(np.array(AUC_list_avg))
    tqdm.write(f'cv_ACC_avg {ACC_array_avg:.4f}: cv_SPE_avg={SPE_array_avg:.4f}, ' 
                                                f'cv_REN_avg= {REC_array_avg:.4f}, '
                                                f'cv_AUC_avg= {AUC_array_avg:.4f}')



if __name__ =='__main__':
    main()
