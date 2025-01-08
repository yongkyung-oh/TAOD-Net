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

# from src.libauc.losses import AUCMLoss, AUCM_MultiLabel
# from src.libauc.optimizers import PESG

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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def model_run(base, mode):
    # make directory
    if not os.path.exists(os.path.join('img_dict')):
        os.mkdir(os.path.join('img_dict'))

    if not os.path.exists(os.path.join('pretraining')):
        os.mkdir(os.path.join('pretraining'))
    if not os.path.exists(os.path.join('pretraining', base)):
        os.mkdir(os.path.join('pretraining', base))

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
        out_path = os.path.join('pretraining', base, '_'.join([str(mode), str(k)]))
        if os.path.exists(out_path):
            print(out_path)
            continue
        else:
            pass
        
        # load cv
        test_split = splits_list[k]
        train_idx = [idx for idx in labels.dropna().index if idx not in test_split]
        pid_train, _ = train_test_split(pid_train, test_size=100/920)
        
        # k folds
        pid_train = train_idx
        pid_test = test_split
        
        # get class weight
        ll_left = labels_eye.apply(lambda s: s[symptoms_left].astype(int).to_numpy(), axis=1).loc[pid_train].to_numpy() 
        ll_right = labels_eye.apply(lambda s: s[symptoms_right].astype(int).to_numpy(), axis=1).loc[pid_train].to_numpy()
        ll_eye = np.vstack([np.vstack(ll_left), np.vstack(ll_right)])
        ll_patiet = np.round((np.vstack(ll_left) + np.vstack(ll_right))/2)
        
        pos_weight = torch.FloatTensor([1,1,1,1,1]).flatten().to(device)
            
        # train model
        img_train = [str(pid)+'_left' for pid in pid_train] + [str(pid)+'_right' for pid in pid_train]
        img_test = [str(pid)+'_left' for pid in pid_test] + [str(pid)+'_right' for pid in pid_test]

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
        test_dataset = img_Dataset(data_path, img_dict, img_test, landmarks_left, landmarks_right,
                                   labels_left, labels_right, mode, valid_transform)

        BATCH_SIZE = 64

        train_batch = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=16)
        test_batch = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=16)

        # Define model
        model = Prediction_model('multi', base).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = [nn.BCEWithLogitsLoss(reduction='mean'), nn.BCEWithLogitsLoss(reduction='mean')] # BCE loss
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        
        best_loss = np.inf
        prev_val_loss = np.inf
        counter = 0

        for e in tqdm(range(100)):
            train_loss, train_metrics = train_multi(model, optimizer, criterion, train_batch, 0, 'CE', None, device)
            test_loss, test_metrics = evaluate_multi(model, criterion, test_batch, 0, 'CE', None, device)

            if (e+1) % 10 == 0:
                print("[Epoch: %03d] train loss : %3.3f | %3.3f | %3.3f | %3.3f | %3.3f" % (e+1, train_loss, train_metrics[0], train_metrics[3], train_metrics[7], train_metrics[8]))
                print("[Epoch: %03d] test  loss : %3.3f | %3.3f | %3.3f | %3.3f | %3.3f" % (e+1, test_loss, test_metrics[0], test_metrics[3], test_metrics[7], test_metrics[8]))

            scheduler.step()

        torch.save(model.state_dict(), os.path.join('pretraining', base, '_'.join([str(mode), str(k), '.pth'])))

        ## get all performance metrics
        model.eval()
        total_loss = 0
        trues_list, preds_list, scores_list = [], [], []
        with torch.no_grad():
            for batch in test_batch:
                x = batch['img'].to(device)
                y = batch['label'].float().to(device)

                logit = model(x)

                scores = torch.sigmoid(logit)
                preds = scores.round()
                trues = y.round()

                trues_list.append(trues.cpu().detach().numpy().reshape(-1,5))
                preds_list.append(preds.cpu().detach().numpy().reshape(-1,5))
                scores_list.append(scores.cpu().detach().numpy().reshape(-1,5))

        trues_all = np.vstack(trues_list)
        preds_all = np.vstack(preds_list)
        scores_all = np.vstack(scores_list)

        out = [k, trues_all, preds_all, scores_all]
        print(base, str(mode), str(k))

        # save out
        out_path = os.path.join('pretraining', base, '_'.join([str(mode), str(k)]))
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--base', type=str, default='ViT')
    parser.add_argument('--mode', type=int, default=0)
    
    # set args
    args = parser.parse_args()
    base = args.base
    mode = args.mode

    if base not in ['VGG16', 'ResNet18', 'ResNet50', 'SE-Net50', 'SK-Net50', 'ResNeSt50', 'EfficientNet', 'ViT', 'Swin']: # check base
        raise ValueError()

    if mode not in range(8): # check input mode
        raise ValueError()
        
    # print settings
    print('====================')
    print('- base: {}'.format(base))
    print('- mode: {}'.format(mode))
    print('====================')

    model_run(base, mode)
