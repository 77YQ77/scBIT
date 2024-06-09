
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class Transform(nn.Module):
    def __init__(self, input_size, hidden_size, num_head, time_seq,num_classes):
        super(Transform, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.input_size = input_size
        self.time_seq = time_seq
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.fc = nn.Linear(self.time_seq, self.hidden_size)  
        self.fc1 = nn.Linear(self.hidden_size, 90)
        self.fc2 = nn.Linear(90, num_classes)
        self.avg_1d = nn.AvgPool1d(90)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, info):
        out = self.encoder_layer1(x)
        out1 = self.encoder_layer2(out)
        outs = out1.permute(0,2,1)
        ou = self.fc(outs)
        out_1 = ou.transpose(2, 1)
        pool = self.avg_1d(out_1)
        pool = pool.squeeze(2)
        out_2 = self.fc1(pool)
        out_2 = self.dropout(out_2)
        out_3 = self.fc2(out_2)
        return out_3


class Transform_eval(nn.Module):
    def __init__(self, input_size, hidden_size, num_head, time_seq, num_classes):
        super(Transform_eval, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.input_size = input_size
        self.time_seq = time_seq
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.fc = nn.Linear(self.time_seq, self.hidden_size)  
        self.fc1 = nn.Linear(self.hidden_size, 9)
        self.fc2 = nn.Linear(9+171, 90)
        self.fc3 = nn.Linear(90, num_classes)
        self.avg_1d = nn.AvgPool1d(90)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, info):
        out = self.encoder_layer1(x)
        out1 = self.encoder_layer2(out)
        outs = out1.permute(0,2,1)
        ou = self.fc(outs)
        out_1 = ou.transpose(2, 1)
        pool = self.avg_1d(out_1)
        pool = pool.squeeze(2)
        out_2 = self.fc1(pool)
        out_2 = self.dropout(out_2)
        out_2 = torch.cat([out_2, info], dim=1)
        out_3 = self.fc2(out_2)
        out_4 = self.fc3(out_3)
        return out_4

class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, time_seq, num_head,):
        super(encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.input_size = input_size
        self.time_seq = time_seq
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.fc = nn.Linear(time_seq, self.hidden_size)  

 
    def forward(self, x, sim):
        out = self.encoder_layer1(x)
        out1 = self.encoder_layer2(out)
        outs = out1.permute(0,2,1)
        fea = self.fc(outs)
        feas = fea.permute(0,2,1) # 650*128*116
        fea = feas.unsqueeze(1).unsqueeze(4)
        sim = sim.unsqueeze(0).unsqueeze(3).to(torch.float32)
        similarity = torch.nn.functional.cosine_similarity(fea, sim, dim=2)
        similar = similarity.view(-1,116*90)
        similar = F.normalize(similar, dim=1)
        return fea, similarity, similar
    

class encoder_regression(nn.Module):
    def __init__(self, input_size, hidden_size, time_seq, num_head,):
        super(encoder_regression, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.input_size = input_size
        self.time_seq = time_seq
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.fc = nn.Linear(time_seq, self.hidden_size)  

 
    def forward(self, x, sim):
        out = self.encoder_layer1(x)
        out1 = self.encoder_layer2(out)
        outs = out1.permute(0,2,1)
        fea = self.fc(outs)
        feas = fea.permute(0,2,1) 
        fea = feas.unsqueeze(1).unsqueeze(4)
        sim = sim.unsqueeze(0).unsqueeze(3).to(torch.float32)
        similarity = torch.nn.functional.cosine_similarity(fea, sim, dim=2)
        return  similarity
    

class Gate_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_head, time_seq):
        super(Gate_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.input_size = input_size
        self.time_seq = time_seq
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.fc = nn.Linear(self.time_seq, self.hidden_size)  
        self.fc1 = nn.Linear(self.hidden_size, 4)
        self.avg_1d = nn.AvgPool1d(90)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.encoder_layer1(x)
        outs = out.permute(0,2,1)
        ou = self.fc(outs)
        out_1 = ou.transpose(2, 1)
        pool = self.avg_1d(out_1)
        pool = pool.squeeze(2)
        out_2 = self.fc1(pool)
        return out_2


class Transform_fusion(nn.Module):
    def __init__(self, input_size, hidden_size, num_head, time_seq, num_classes):
        super(Transform_fusion, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.input_size = input_size
        self.time_seq = time_seq
        self.experts_out = 171
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.w_gates = Gate_model(input_size=116, hidden_size=256, time_seq =140 ,num_head=4)
        self.fc = nn.Linear(self.time_seq, self.hidden_size)  
        self.fc1 = nn.Linear(self.hidden_size, 9)
        self.fc2 = nn.Linear(9+171, 90)
        self.fc3 = nn.Linear(90, num_classes)
        self.avg_1d = nn.AvgPool1d(90)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, age_info,sex_info,gene_info,group_info):
        experts_o_tensor = torch.stack([age_info,sex_info,gene_info,group_info])
        gates_o = self.softmax(self.w_gates(x))
        tower_input = gates_o.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor
        info = torch.sum(tower_input, dim=0)
        out = self.encoder_layer1(x)
        out1 = self.encoder_layer2(out)
        outs = out1.permute(0,2,1)
        ou = self.fc(outs)
        out_1 = ou.transpose(2, 1)
        pool = self.avg_1d(out_1)
        pool = pool.squeeze(2)
        out_2 = self.fc1(pool)
        out_2 = self.dropout(out_2)
        out_2 = torch.cat([out_2, info], dim=1)
        out_3 = self.fc2(out_2)
        out_4 = self.fc3(out_3)
        return out_4
    
class Transform_fusion_multiclass(nn.Module):
    def __init__(self, input_size, hidden_size, num_head, time_seq, num_classes):
        super(Transform_fusion_multiclass, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.input_size = input_size
        self.time_seq = time_seq
        self.experts_out = 171
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_head)
        self.w_gates = Gate_model(input_size=116, hidden_size=256, time_seq =140 ,num_head=4)
        self.fc = nn.Linear(self.time_seq, self.hidden_size)  
        self.fc1 = nn.Linear(self.hidden_size, 9)
        self.fc2 = nn.Linear(9+171, 90)
        self.fc3 = nn.Linear(90, num_classes)
        self.avg_1d = nn.AvgPool1d(90)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, age_info,sex_info,gene_info,group_info):
        experts_o_tensor = torch.stack([age_info,sex_info,gene_info,group_info])
        gates_o = self.softmax(self.w_gates(x))
        tower_input = gates_o.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor
        info = torch.sum(tower_input, dim=0)
        out = self.encoder_layer1(x)
        out1 = self.encoder_layer2(out)
        outs = out1.permute(0,2,1)
        ou = self.fc(outs)
        out_1 = ou.transpose(2, 1)
        pool = self.avg_1d(out_1)
        pool = pool.squeeze(2)
        out_2 = self.fc1(pool)
        out_2 = self.dropout(out_2)
        out_2 = torch.cat([out_2, info], dim=1)
        out_3 = self.fc2(out_2)
        out_4 = self.fc3(out_3)
        return out_4