from argparse import Namespace
from collections import OrderedDict
import os
import pickle 

from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored
from scipy.stats import ttest_ind
import torch
from torch.optim import lr_scheduler

from datasets.dataset_generic import save_splits
from models.model_coattn import G_HANet_Surv
from utils.utils import *

from utils.coattn_train_utils import *
from utils.cluster_train_utils import *
from statsmodels.stats.multitest import multipletests


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def train(datasets: tuple, cur: int, args: Namespace):
    print('\nTraining Fold {}!'.format(cur))
    writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError
 
    if args.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')
    


    print('\nInit Loaders...', end=' ')
   
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')
   
    c_index_best = 0

    print('Doing differential analysis')
    omic_1_high_risk, omic_2_high_risk, omic_3_high_risk, omic_4_high_risk, omic_5_high_risk, omic_6_high_risk = [], [], [], [], [], []
    omic_1_low_risk, omic_2_low_risk, omic_3_low_risk, omic_4_low_risk, omic_5_low_risk, omic_6_low_risk = [], [], [], [], [], []
    for (_, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in train_loader:
        if c == 0:
            if label >= 2:
                omic_1_low_risk.append(data_omic1)
                omic_2_low_risk.append(data_omic2)
                omic_3_low_risk.append(data_omic3)
                omic_4_low_risk.append(data_omic4)
                omic_5_low_risk.append(data_omic5)
                omic_6_low_risk.append(data_omic6)
            else:
                omic_1_high_risk.append(data_omic1)
                omic_2_high_risk.append(data_omic2)
                omic_3_high_risk.append(data_omic3)
                omic_4_high_risk.append(data_omic4)
                omic_5_high_risk.append(data_omic5)
                omic_6_high_risk.append(data_omic6)
        elif label >= 2:
            omic_1_low_risk.append(data_omic1)
            omic_2_low_risk.append(data_omic2)
            omic_3_low_risk.append(data_omic3)
            omic_4_low_risk.append(data_omic4)
            omic_5_low_risk.append(data_omic5)
            omic_6_low_risk.append(data_omic6)

    omic_1_low_risk = np.stack(omic_1_low_risk)
    omic_1_high_risk = np.stack(omic_1_high_risk)
    omic_2_low_risk = np.stack(omic_2_low_risk)
    omic_2_high_risk = np.stack(omic_2_high_risk)
    omic_3_low_risk = np.stack(omic_3_low_risk)
    omic_3_high_risk = np.stack(omic_3_high_risk)
    omic_4_low_risk = np.stack(omic_4_low_risk)
    omic_4_high_risk = np.stack(omic_4_high_risk)
    omic_5_low_risk = np.stack(omic_5_low_risk)
    omic_5_high_risk = np.stack(omic_5_high_risk)
    omic_6_low_risk = np.stack(omic_6_low_risk)
    omic_6_high_risk = np.stack(omic_6_high_risk)

    omic_size = []
    omic_mask = []

    p_values = []
    for i in range(omic_1_low_risk.shape[1]):
        t_stat, p_val = ttest_ind(omic_1_low_risk[:, i], omic_1_high_risk[:, i], equal_var=False)
        p_values.append(p_val)

    p_values = np.array(p_values)
    omic_size.append((p_values < 0.05).sum())
    omic_mask.append(p_values < 0.05)
    
    p_values = []
    for i in range(omic_2_low_risk.shape[1]):
        t_stat, p_val = ttest_ind(omic_2_low_risk[:, i], omic_2_high_risk[:, i], equal_var=False)
        p_values.append(p_val)

    p_values = np.array(p_values)
    omic_size.append((p_values < 0.05).sum())
    omic_mask.append(p_values < 0.05)

    p_values = []
    for i in range(omic_3_low_risk.shape[1]):
        t_stat, p_val = ttest_ind(omic_3_low_risk[:, i], omic_3_high_risk[:, i], equal_var=False)
        p_values.append(p_val)

    p_values = np.array(p_values)
    omic_size.append((p_values < 0.05).sum())
    omic_mask.append(p_values < 0.05)

    p_values = []
    for i in range(omic_4_low_risk.shape[1]):
        t_stat, p_val = ttest_ind(omic_4_low_risk[:, i], omic_4_high_risk[:, i], equal_var=False)
        p_values.append(p_val)

    p_values = np.array(p_values)
    omic_size.append((p_values < 0.05).sum())
    omic_mask.append(p_values < 0.05)


    p_values = []
    for i in range(omic_5_low_risk.shape[1]):
        t_stat, p_val = ttest_ind(omic_5_low_risk[:, i], omic_5_high_risk[:, i], equal_var=False)
        p_values.append(p_val)

    p_values = np.array(p_values)
    omic_size.append((p_values < 0.05).sum())
    omic_mask.append(p_values < 0.05)

    p_values = []
    for i in range(omic_6_low_risk.shape[1]):
        t_stat, p_val = ttest_ind(omic_6_low_risk[:, i], omic_6_high_risk[:, i], equal_var=False)
        p_values.append(p_val)

    p_values = np.array(p_values)
    omic_size.append((p_values < 0.05).sum())
    omic_mask.append(p_values < 0.05)
    
    print('Number of high-risk samples {}; number of low-risk samples {}'.format(omic_1_high_risk.shape[0], omic_1_low_risk.shape[0]))
    print('Omic sizes for different datasets {}'.format(omic_size))

    print('\nInit Model...', end=' ')

    model_dict = {'omic_sizes': omic_size, 'n_classes': args.n_classes, 'num_tokens': args.num_tokens, 'ratio': args.ratio}
    model = G_HANet_Surv(**model_dict).to(torch.device('cuda'))

    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('Done!')

    # breakpoint()

    for epoch in range(args.max_epochs):
        train_loop_survival_coattn(epoch, model, train_loader, optimizer, loss_fn, reg_fn, args.lambda_reg, args.gc, scheduler, omic_mask)
        c_index = validate_survival_coattn(cur, epoch, model, val_loader, loss_fn, omic_mask)
        if c_index > c_index_best:
            c_index_best = c_index
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
            # model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))

    # c_index_best = 0
           
    return c_index_best

