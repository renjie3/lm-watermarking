import numpy as np
import torch
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet18


class CLModel(nn.Module):
    def __init__(self, encoder_dim=2048, feat_dim=2):
        super(CLModel, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(2560, 4096, bias=False),  # FIXME
                               nn.BatchNorm1d(4096),
                               nn.ReLU(),
                               )

        self.fc2 = nn.Sequential(nn.Linear(4096, 2048, bias=False), 
                               nn.BatchNorm1d(2048),
                               nn.ReLU(),
                               )

        self.fc3 = nn.Sequential(nn.Linear(2048, 512, bias=False), 
                               nn.BatchNorm1d(512),
                               nn.ReLU(),
                               )
        
        self.fc4 = nn.Sequential(nn.Linear(512, 128, bias=False), 
                               nn.BatchNorm1d(128),
                               nn.ReLU(),
                               )
                               
        self.g = nn.Linear(128, feat_dim, bias=True)

    def forward(self, x):
        # print(x.shape)
        # input("check")
        x = self.fc1(x)
        # print(x.shape)
        # input("check")
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        # x = self.fc5(x)
        x = self.g(x)
        return F.normalize(x, dim=-1)
        # return feature, F.normalize(out, dim=-1)

def contrastive_train_batch(net, pos_1, pos_2, train_optimizer, temperature, pytorch_aug=False):
    net.eval()
    total_loss, total_num = 0.0, 0
    
    out_1 = net(pos_1)
    out_2 = net(pos_2)

    for i in range(10):
        print(out_1[i])
        print(out_2[i])
        input("check")
    
    out = torch.cat([out_1, out_2], dim=0)
    
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
    
    sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    total_loss = loss.item()

    return total_loss * pos_1.shape[0], pos_1.shape[0]

def infoNCE_loss(out_1, out_2, temperature):

    with torch.no_grad():
    
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # print(out)
        # input("check")
        # print(out.shape)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * out_1.shape[0], device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * out_1.shape[0], -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        total_loss = loss.item()

    return total_loss * out_1.shape[0], out_1.shape[0]