import os
import sys
import json
import copy
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import argparse
import timm
from tqdm import tqdm

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

sys.path.append('..')
from src.utils import *
from src.model import *
from src.losses import *

from src.libauc.losses import AUCMLoss, AUCM_MultiLabel, CompositionalAUCLoss, CompositionalAUC_MultiLabel
from src.libauc.optimizers import PESG, PDSCA

# setup seed
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

SEED = 0
seed_everything(SEED)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

data_path = "../data"
label_file = "CAS_1-1054_score.csv"
landmark_file = "1-1054_landmark_xy.csv"

symptoms_list = ['Red_lid', 'Red_conj','Swl_crncl', 'Swl_lid', 'Swl_conj']
symptoms_mode_dict = dict(zip(symptoms_list, [7,1,2,7,1]))

# load label info
# labels = pd.read_excel(os.path.join(data_path, label_file))
labels = pd.read_csv(label_file)
labels = labels.set_index('p_num')
labels = labels.replace(0.5,1)
    
labels_eye, labels_patient = load_labels(data_path, label_file)
landmarks_left, landmarks_right = load_landmark(data_path, landmark_file)


def model_run(setting, base, loss, mode, symptom_idx, symptom_name):
    seed_everything(symptom_idx)
    symptom = symptom_name
    
    # make directory
    if not os.path.exists(os.path.join('results')):
        os.mkdir(os.path.join('results'))
    if not os.path.exists(os.path.join('results', setting)):
        os.mkdir(os.path.join('results', setting))
    if not os.path.exists(os.path.join('results', setting, base)):
        os.mkdir(os.path.join('results', setting, base))
    if not os.path.exists(os.path.join('results', setting, base, symptom)):
        os.mkdir(os.path.join('results', setting, base, symptom))

    # setup label info
    symptoms_left = ['_'.join([symptom, 'left', 'true']) for symptom in symptoms_list]
    symptoms_right = ['_'.join([symptom, 'right', 'true']) for symptom in symptoms_list]

    # load info
    labels_left = labels_eye.apply(lambda s: s[symptoms_left].astype(int).tolist(), axis=1).to_dict()
    labels_right = labels_eye.apply(lambda s: s[symptoms_right].astype(int).tolist(), axis=1).to_dict()

    # read splits
    with open('pre_split_random.pkl', 'rb') as f:
        splits_list = pickle.load(f)
        
    # road img
    try:
        with open('img_dict/img_dict_{}.pkl'.format(mode), 'rb') as f:
            img_dict = pickle.load(f)
            
    except:
        img_dict = {}
        for pid in tqdm(labels.index):
            try:
                img_dict[str(pid)+'left'] = get_img_cropped(data_path, pid, landmarks_left, mode=mode, bound=0.1) # left 
                img_dict[str(pid)+'right'] = get_img_cropped(data_path, pid, landmarks_right, mode=mode, bound=0.1) # right
            except:
                continue

        with open('img_dict/img_dict_{}.pkl'.format(mode), 'wb') as f:
            pickle.dump(img_dict, f)    
    
    print('total img: {}/{}'.format(len(img_dict), len(labels.index)*2))
    
    # run model training
    for k in range(30):
        # check out
        out_path = os.path.join('results', setting, base, str(symptom), '_'.join([str(mode), str(k), str(domain), str(loss)]))
        if os.path.exists(out_path):
            print(out_path)
            continue
        else:
            pass
        
        # load cv
        test_split = splits_list[k]
        train_idx = [idx for idx in labels.dropna().index if idx not in test_split]
        
        # k folds
        pid_train = train_idx
        pid_test = test_split
        
        # get class weight
        ll_left = labels_eye.apply(lambda s: s[symptoms_left].astype(int).to_numpy(), axis=1).loc[pid_train].to_numpy() 
        ll_right = labels_eye.apply(lambda s: s[symptoms_right].astype(int).to_numpy(), axis=1).loc[pid_train].to_numpy()
        ll_eye = np.vstack([np.vstack(ll_left), np.vstack(ll_right)])
        ll_patiet = np.round((np.vstack(ll_left) + np.vstack(ll_right))/2)
        
        # class weight between 0 to 1
        alpha = ll_eye.sum(axis=0)/len(ll_eye)
        alpha = torch.FloatTensor(alpha).flatten().to(device)
        
        gamma_neg, gamma_pos = [], []
        for i in range(5):
            gamma_neg.append(1/alpha[i].item())
            gamma_pos.append(0)
        
        # relative ratio
        pos_weight = (ll_eye==0).sum(axis=0)/ll_eye.sum(axis=0)
        pos_weight = torch.FloatTensor(pos_weight).flatten().to(device)        

        ## 
        results_all = []
        pid_train_k, pid_valid_k = train_test_split(pid_train, test_size=100/920)
        pid_test_k = pid_test

        img_train = [str(pid)+'_left' for pid in pid_train_k] + [str(pid)+'_right' for pid in pid_train_k]
        img_valid = [str(pid)+'_left' for pid in pid_valid_k] + [str(pid)+'_right' for pid in pid_valid_k]
        img_test = [str(pid)+'_left' for pid in pid_test_k] + [str(pid)+'_right' for pid in pid_test_k]
        
        # Load image
        train_transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.RandomCrop((224, 224)),
                            transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        valid_transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        train_dataset = img_Dataset(data_path, img_dict, img_train, landmarks_left, landmarks_right,
                                    labels_left, labels_right, mode, train_transform)
        valid_dataset = img_Dataset(data_path, img_dict, img_valid, landmarks_left, landmarks_right,
                                    labels_left, labels_right, mode, valid_transform)
        test_dataset = img_Dataset(data_path, img_dict, img_test, landmarks_left, landmarks_right,
                                    labels_left, labels_right, mode, valid_transform)

        BATCH_SIZE = 64

        train_batch = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=16)
        valid_batch = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=16)
        test_batch = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=16)

        # Define model & load pretrained parameters
        model = Prediction_model(setting, base).to(device)
        model.model.load_state_dict(torch.load(os.path.join('pretraining', base, '_'.join([str(mode), str(k), '.pth']))), strict=False)

        # Set fc parameters trainable
        for param in model.fc.parameters():
            param.grad = None
            param.requires_grad = True
            
        optimizer = MultipleOptimizer(optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5))

        if loss == 'CE':
            criterion = [nn.BCEWithLogitsLoss(reduction='mean'), 
                            nn.BCEWithLogitsLoss(reduction='mean')] # BCE loss
        elif loss == 'WCE':
            criterion = [nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean'), 
                            nn.BCEWithLogitsLoss(pos_weight=pos_weight[symptom_idx], reduction='mean')] # Weighted BCE loss
        elif loss == 'Focal':
            criterion = [Focal_Multilabel(alpha=alpha, num_classes=5, reduction='mean'),
                            FocalLoss(alpha=alpha[symptom_idx], reduction='mean')] # Focal loss
        elif loss == 'ASL':
            criterion = [ASL_Multilabel(gamma_neg=gamma_neg, gamma_pos=gamma_pos, num_classes=5),
                            AsymmetricLossOptimized(gamma_neg=gamma_neg[symptom_idx], gamma_pos=gamma_pos[symptom_idx])] # ASL loss
        elif loss == 'AUCM':
            criterion = [AUCM_MultiLabel(num_classes=5, imratio=alpha.tolist()), 
                            AUCMLoss(imratio=alpha[symptom_idx].item())] # ACUM loss
            optimizer = MultipleOptimizer(PESG(model, criterion[0], lr=1e-3, weight_decay=1e-5, device=device), 
                                            PESG(model, criterion[1], lr=1e-3, weight_decay=1e-5, device=device))
        elif loss == 'CAUCM':
            criterion = [CompositionalAUC_MultiLabel(num_classes=5, imratio=alpha.tolist()), 
                            CompositionalAUCLoss(imratio=alpha[symptom_idx].item())] # CAUCM loss
            optimizer = MultipleOptimizer(PDSCA(model, criterion[0], lr=1e-3, weight_decay=1e-5, device=device), 
                                            PDSCA(model, criterion[1], lr=1e-3, weight_decay=1e-5, device=device))
            
        if loss in ['CE', 'WCE', 'Focal', 'ASL']:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizers[0], T_max=10, eta_min=1e-5)
        elif loss in ['AUCM', 'CAUCM']:
            scheduler = [optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizers[0], T_max=10, eta_min=1e-5),
                            optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizers[1], T_max=10, eta_min=1e-5)]
        
        if setting == 'adaptive':
            # lamda = nn.Parameter(torch.tensor([[0.5]]), requires_grad=True).to(device)
            lamda = nn.Parameter(torch.tensor(0.5))
            lamda = lamda.requires_grad_(True)
        elif setting == 'combined':
            lamda = 0.5

        if setting == 'adaptive':
            try:
                optimizer = MultipleOptimizer(*optimizer.optimizers, optim.Adam([lamda], lr=1e-3, weight_decay=1e-5))
            except:
                optimizer = MultipleOptimizer(optimizer, optim.Adam([lamda], lr=1e-3, weight_decay=1e-5))                    
        else:
            pass
            
        best_loss = np.inf
        prev_val_loss = np.inf
        best_metric = 0
        prev_val_metric = 0
        counter = 0

        loss_list = []
        for e in tqdm(range(100)):
            if (setting == 'adaptive') or (setting == 'combined'):
                train_loss, train_metrics = train_multi(model, optimizer, criterion, train_batch, symptom_idx, loss, lamda, device)
                valid_loss, valid_metrics = evaluate_multi(model, criterion, valid_batch, symptom_idx, loss, lamda, device)
                test_loss, test_metrics = evaluate_multi(model, criterion, test_batch, symptom_idx, loss, lamda, device)
            elif setting == 'multi':
                train_loss, train_metrics = train_multi(model, optimizer, criterion, train_batch, symptom_idx, loss, None, device)
                valid_loss, valid_metrics = evaluate_multi(model, criterion, valid_batch, symptom_idx, loss, None, device)
                test_loss, test_metrics = evaluate_multi(model, criterion, test_batch, symptom_idx, loss, None, device)
            elif setting == 'binary':
                train_loss, train_metrics = train_binary(model, optimizer, criterion, train_batch, symptom_idx, loss, device)
                valid_loss, valid_metrics = evaluate_binary(model, criterion, valid_batch, symptom_idx, loss, device)
                test_loss, test_metrics = evaluate_binary(model, criterion, test_batch, symptom_idx, loss, device)

            # save loss and lamda
            if setting == 'adaptive':
                loss_list.append([train_loss, valid_loss, test_loss, train_metrics, valid_metrics, test_metrics, lamda.clip(0.0,1.0).item()])
            else:
                loss_list.append([train_loss, valid_loss, test_loss, train_metrics, valid_metrics, test_metrics])
                
            if (e+1) % 10 == 0:
                if setting == 'adaptive':
                    print(f"lambda: {lamda.clip(0.0,1.0).item():.3f}")
                else:
                    pass
                
                print("[Epoch: %03d] train loss : %3.3f | %3.3f | %3.3f | %3.3f | %3.3f" % (e+1, train_loss, train_metrics[0], train_metrics[3], train_metrics[7], train_metrics[8]))
                print("[Epoch: %03d] valid loss : %3.3f | %3.3f | %3.3f | %3.3f | %3.3f" % (e+1, valid_loss, valid_metrics[0], valid_metrics[3], valid_metrics[7], valid_metrics[8]))
                print("[Epoch: %03d] test  loss : %3.3f | %3.3f | %3.3f | %3.3f | %3.3f" % (e+1, test_loss, test_metrics[0], test_metrics[3], test_metrics[7], test_metrics[8]))
                
            valid_metric = valid_metrics[8] # auc
            
            ## best weight by auc
            # if (valid_loss < best_loss): 
            if (valid_metric > best_metric): 
                best_loss = valid_loss
                best_metric = valid_metric
                best_model_wts = copy.deepcopy(model.state_dict())

            ## early stop with loss
            if (valid_loss < prev_val_loss):
                counter = 0
            else:
                counter += 1

            if (e > 20) & (counter >= 5):
                break

            prev_val_loss = valid_loss
            prev_val_metric = valid_metric
    
            # lr scheduler
            if loss in ['AUCM', 'CAUCM']:
                for s in scheduler:
                    s.step()
            else:
                scheduler.step()

            
        best_model = copy.deepcopy(model)
        best_model.load_state_dict(best_model_wts)

        ## get all performance metrics
        best_model.eval()
        total_loss = 0
        trues_list, preds_list, scores_list = [], [], []
        with torch.no_grad():
            for batch in test_batch:
                x = batch['img'].to(device)
                y = batch['label'].float().to(device)

                logit = best_model(x)

                scores = torch.sigmoid(logit)
                preds = scores.round()
                trues = y.round()

                if setting in ['adaptive', 'combined', 'multi']:
                    trues_list.append(trues.cpu().detach().numpy().reshape(-1,5))
                    preds_list.append(preds.cpu().detach().numpy().reshape(-1,5))
                    scores_list.append(scores.cpu().detach().numpy().reshape(-1,5))
                elif setting == 'binary':
                    trues_list.append(trues[:,symptom_idx].unsqueeze(1).cpu().detach().numpy().reshape(-1,1))
                    preds_list.append(preds.cpu().detach().numpy().reshape(-1,1))
                    scores_list.append(scores.cpu().detach().numpy().reshape(-1,1))

        trues_all = np.vstack(trues_list)
        preds_all = np.vstack(preds_list)
        scores_all = np.vstack(scores_list)
        
        out = [k, trues_all, preds_all, scores_all, loss_list]
        results_all.append(out)
        
        print(setting, base, loss, str(symptom), str(mode), str(k))

    # save out
    out_path = os.path.join('results', setting, base, str(symptom), '_'.join([str(mode), str(k), str(domain), str(loss)]))
    with open(out_path, 'wb') as f:
        pickle.dump(results_all, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--setting', type=str, default='combined')
    parser.add_argument('--base', type=str, default='ViT')
    parser.add_argument('--loss', type=str, default='CE')
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--symptom', type=int, default=0)

    parser.add_argument('--domain', action='store_true')
    parser.add_argument('--no-domain', dest='domain', action='store_false')
    parser.set_defaults(domain=True)
    
    # set args
    args = parser.parse_args()
    setting = args.setting
    base = args.base
    loss = args.loss
    mode = args.mode
    symptom_idx = args.symptom
    symptom_name = symptoms_list[symptom_idx]
    domain = bool(args.domain)

    if setting not in ['adaptive', 'combined', 'multi', 'binary']: # check setting
        raise ValueError() 

    if base not in ['VGG16', 'ResNet18', 'ResNet50', 'SE-Net50', 'SK-Net50', 'ResNeSt50', 'EfficientNet', 'ViT', 'Swin']: # check base
        raise ValueError()

    if loss not in ['CE', 'WCE', 'Focal', 'ASL', 'AUCM', 'CAUCM']: # check base
        raise ValueError()
        
    if mode not in range(8): # check input mode
        raise ValueError()

    if symptom_name not in symptoms_list: # check symptom name
        raise ValueError()

    if domain: # using domain knowledge, else use input mode
        mode = symptoms_mode_dict[symptom_name]
        
    # print settings
    print('====================')
    print('- symptom: {}'.format(symptom_name))
    print('- setting: {}'.format(setting))
    print('- base: {}'.format(base))
    print('- loss: {}'.format(loss))
    print('- mode: {}'.format(mode))
    print('- domain: {}'.format(domain))
    print('====================')

    model_run(setting, base, loss, mode, symptom_idx, symptom_name)
