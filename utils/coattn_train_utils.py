import numpy as np
import torch
import pickle 
from utils.utils import *
import random
from collections import OrderedDict

from argparse import Namespace
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def samping(patches, labels, p=0.5):
    temp = []
    for i in range(4):
        indexes = torch.nonzero(labels==i)
        length = indexes.size(0)
        num_selected_nodes = int(p * length)
        perm = torch.randperm(length)#.cuda()
        selected_nodes = perm[0: num_selected_nodes]
        
        temp.append(patches[selected_nodes])
    temp = torch.cat(temp, dim=0)
    perm = torch.randperm(temp.size(0)) #.cuda()
    temp = temp[perm]

    return temp

def train_loop_survival_coattn(epoch, model, loader, optimizer, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, scheduler=None, omic_mask=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss_const_l1, train_loss_const_distri, train_loss = 0., 0., 0., 0. 
    loss_fn_recons = nn.MSELoss()

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):

        length = data_WSI.size(0)

        data_WSI = data_WSI.to(device)
        if len(data_WSI.size()) == 1:
            continue

        loss_const_distri, loss_const_l1 = 0.0, 0.0

        data_omics = [data_omic1.to(device), data_omic2.to(device), data_omic3.to(device), data_omic4.to(device), data_omic5.to(device), data_omic6.to(device)]

        label = label.type(torch.LongTensor).to(device)

        c = c.type(torch.FloatTensor).to(device)

        hazards, S, _, _, re_omics, sim  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        loss_survival = loss_fn(hazards=hazards, S=S, Y=label, c=c) 

        
        for i in range(6):
            loss_const_distri += sce_loss(re_omics[i], data_omics[i][omic_mask[i]])
            loss_const_l1 += loss_fn_recons(re_omics[i], data_omics[i][omic_mask[i]])

        loss = loss_survival + 0.3 * (loss_const_distri + loss_const_l1)

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_survival.item()
        train_loss_const_l1 += loss_const_l1.item()
        train_loss_const_distri += loss_const_distri.item()
       

        if (batch_idx + 1) % 200 == 0:
            print('batch {}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size:'.format(batch_idx, label.item(), float(event_time), float(risk)))
        loss = loss / gc 
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= (len(loader))
    train_loss_const_l1 /= (len(loader))
    train_loss_const_distri /= (len(loader))

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss_const_l1: {:.4f}, train_loss_const_distri: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss_const_l1, train_loss_const_distri, c_index))
      


def validate_survival_coattn(cur, epoch, model, loader, loss_fn=None, omic_mask=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    val_loss_surv, val_loss_const_l1, val_loss_const_distri = 0., 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    loss_fn_recons = nn.MSELoss()

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):
        length = data_WSI.size(0)
        
        if len(data_WSI.size()) == 1:
            continue
    
        data_WSI = data_WSI.to(device)

        data_omics = [data_omic1.to(device), data_omic2.to(device), data_omic3.to(device), data_omic4.to(device), data_omic5.to(device), data_omic6.to(device)]

        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        loss_const_distri, loss_const_l1 = 0.0, 0.0

        with torch.no_grad():
            hazards, S, _, _, re_omics, sim  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

            loss_survival = loss_fn(hazards=hazards, S=S, Y=label, c=c) 
            
            for i in range(6):
                loss_const_distri += sce_loss(re_omics[i], data_omics[i][omic_mask[i]])
                loss_const_l1 += loss_fn_recons(re_omics[i], data_omics[i][omic_mask[i]])

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_survival.item()
        val_loss_const_l1 += loss_const_l1.item()
        val_loss_const_distri += loss_const_distri.item()

    val_loss_surv /= len(loader)
    val_loss_const_l1 /= len(loader)
    val_loss_const_distri /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, val_loss_surv: {:.4f}, val_loss_const_l1: {:.4f}, val_loss_const_distri: {:.4f}, test_c_index: {:.4f}'.format(epoch, val_loss_surv, val_loss_const_l1, val_loss_const_distri, c_index))

    return c_index
