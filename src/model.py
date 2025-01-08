import numpy as np

import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import timm
from tqdm import tqdm

from src.utils import *


# Define dataset
class img_Dataset(Dataset):
    def __init__(self, data_path, img_dict, img_set, landmarks_left, landmarks_right, 
                 labels_left, labels_right, mode, image_transform):
        self.data_path = data_path
        self.img_dict = img_dict
        self.img_set = img_set
        self.landmarks_left = landmarks_left
        self.landmarks_right = landmarks_right
        self.labels_left = labels_left
        self.labels_right = labels_right
        self.mode = mode
        self.image_transform = image_transform
        
    def __len__(self):
        return len(self.img_set)
    
    def __getitem__(self, idx):
        pid = int(self.img_set[idx].split('_')[0])
        loc = self.img_set[idx].split('_')[1]
        
        if loc=='left':
            img_c = self.img_dict[str(pid)+'left']
            img_c = self.image_transform(Image.fromarray(img_c))
            label = self.labels_left[pid]
        elif loc=='right':
            img_c = self.img_dict[str(pid)+'right']
            img_c = cv2.flip(img_c, 1) # vertical flip
            img_c = self.image_transform(Image.fromarray(img_c))
            label = self.labels_right[pid]
        else:
            pass
        
        sample = {'label': torch.as_tensor(label), 'img': torch.FloatTensor(img_c)}
        return sample
    
    
# Define model
class Prediction_model(nn.Module):
    def __init__(self, setting='multi', base='ViT', hidden=256, dropout=0.5):
        super().__init__()

        # set model_dict
        model_dict = {
            'VGG16': 'vgg16',
            'ResNet18': 'resnet18',
            'ResNet50': 'resnet50',
            'SE-Net50': 'legacy_seresnet50',
            'SK-Net50': 'skresnext50_32x4d',
            'ResNeSt50': 'resnest50d',
            'EfficientNet': 'efficientnet_b0',
            'ViT': 'vit_small_patch16_224',
            'Swin': 'swin_small_patch4_window7_224',
        }

        model_hidden_dict = {
            'VGG16': 1000,
            'ResNet18': 512,
            'ResNet50': 2048,
            'SE-Net50': 1000,
            'SK-Net50': 2048,
            'ResNeSt50': 2048,
            'EfficientNet': 1000,
            'ViT': 384,
            'Swin': 768,
        }

        # set target_num
        if setting in ['adaptive', 'combined', 'multi']:
            target_num = 5
        elif setting=='binary':
            target_num = 1
           
        self.model = timm.create_model(model_dict[base], pretrained=True)
        if base in ['VGG16', 'ResNet18', 'ResNet50', 'SE-Net50', 'SK-Net50', 'ResNeSt50', 'EfficientNet']:
            self.model.fc = nn.Identity()
        elif base in ['ViT', 'Swin']:
            self.model.head = nn.Identity()
        
        self.bn = nn.BatchNorm2d(3) # batch-wise 
        self.fc = nn.Sequential(nn.Linear(model_hidden_dict[base], hidden), 
                                nn.BatchNorm1d(hidden), nn.LeakyReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden, target_num))
        ##
        init_weights(self.fc)
        self.fc[-1].weight.register_hook(lambda grad: 100 * grad)
        self.fc[-1].bias.register_hook(lambda grad: 100 * grad)
                
        # Set all parameters trainable
        for param in self.model.parameters():
            param.grad = None
            param.requires_grad = True

    def forward(self, x):
        x = self.bn(x.float())
        x = self.model(x)
        x = self.fc(x)
        return x

    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    
def train_binary(model, optimizer, criterion, train_iter, symptom_idx, loss, device):
    model.train()
    total_loss = 0
    trues_list, preds_list, scores_list = [], [], []
    for batch in train_iter:
        x = batch['img'].to(device)
        y = batch['label'].float().to(device)
        optimizer.zero_grad()

        logit = model(x)
        loss_val = criterion[1](logit, y[:,symptom_idx].unsqueeze(1))            
        loss_val.backward()
        optimizer.step()

        scores = torch.sigmoid(logit)
        preds = scores.round()
        trues = y[:,symptom_idx].unsqueeze(1).round()

        total_loss += loss_val.item()
        trues_list.append(trues.cpu().detach().numpy().reshape(-1,1))
        preds_list.append(preds.cpu().detach().numpy().reshape(-1,1))
        scores_list.append(scores.cpu().detach().numpy().reshape(-1,1))
        
    trues_all = np.vstack(trues_list)
    preds_all = np.vstack(preds_list)
    scores_all = np.vstack(scores_list)
    
    avg_loss = total_loss / len(train_iter)
    avg_metrics = get_metrics(trues_all, preds_all, scores_all)
    
    return avg_loss, avg_metrics 


def evaluate_binary(model, criterion, val_iter, symptom_idx, loss, device):
    model.eval()
    total_loss = 0
    trues_list, preds_list, scores_list = [], [], []
    with torch.no_grad():
        for batch in val_iter:
            x = batch['img'].to(device)
            y = batch['label'].float().to(device)

            logit = model(x)
            loss_val = criterion[1](logit, y[:,symptom_idx].unsqueeze(1))

            scores = torch.sigmoid(logit)
            preds = scores.round()
            trues = y[:,symptom_idx].unsqueeze(1).round()

            total_loss += loss_val.item()
            trues_list.append(trues.cpu().detach().numpy().reshape(-1,1))
            preds_list.append(preds.cpu().detach().numpy().reshape(-1,1))
            scores_list.append(scores.cpu().detach().numpy().reshape(-1,1))
            
    trues_all = np.vstack(trues_list)
    preds_all = np.vstack(preds_list)
    scores_all = np.vstack(scores_list)
    
    avg_loss = total_loss / len(val_iter)
    avg_metrics = get_metrics(trues_all, preds_all, scores_all)
    
    return avg_loss, avg_metrics 
    
    
def train_multi(model, optimizer, criterion, train_iter, symptom_idx, loss, lamda, device):
    model.train()
    total_loss = 0
    trues_list, preds_list, scores_list = [], [], []
    for batch in train_iter:
        x = batch['img'].to(device)
        y = batch['label'].float().to(device)
        optimizer.zero_grad()

        logit = model(x)
        loss_m = criterion[0](logit, y)
        loss_b = criterion[1](logit[:,symptom_idx], y[:,symptom_idx])                     
                                 
        if lamda:
            if type(lamda) == type(nn.Parameter()):
                lamda = lamda.clone().clip(0.0,1.0).to(device)
            else:
                pass
            loss_val = lamda * loss_m + (1-lamda) * loss_b
        else:
            loss_val = loss_m

        if type(lamda) == type(nn.Parameter().to(device)):
            loss_val.backward(retain_graph=True)
        else:
            loss_val.backward()
        optimizer.step()

        scores = torch.sigmoid(logit)
        preds = scores.round()
        trues = y.round()

        total_loss += loss_val.item()
        trues_list.append(trues.cpu().detach().numpy().reshape(-1,5))
        preds_list.append(preds.cpu().detach().numpy().reshape(-1,5))
        scores_list.append(scores.cpu().detach().numpy().reshape(-1,5))

    trues_all = np.vstack(trues_list)
    preds_all = np.vstack(preds_list)
    scores_all = np.vstack(scores_list)
    
    avg_loss = total_loss / len(train_iter)
    avg_metrics = get_metrics(trues_all[:,symptom_idx], preds_all[:,symptom_idx], scores_all[:,symptom_idx])
    
    return avg_loss, avg_metrics 


def evaluate_multi(model, criterion, val_iter, symptom_idx, loss, lamda, device):
    model.eval()
    total_loss = 0
    trues_list, preds_list, scores_list = [], [], []
    with torch.no_grad():
        for batch in val_iter:
            x = batch['img'].to(device)
            y = batch['label'].float().to(device)

            logit = model(x)
            loss_m = criterion[0](logit, y)
            loss_b = criterion[1](logit[:,symptom_idx], y[:,symptom_idx])                     

            if lamda:
                try:
                    lamda = lamda.clone().clip(0.0,1.0).to(device)
                except:
                    pass
                loss_val = lamda * loss_m + (1-lamda) * loss_b
            else:
                loss_val = loss_m

            scores = torch.sigmoid(logit)
            preds = scores.round()
            trues = y.round()

            total_loss += loss_val.item()
            trues_list.append(trues.cpu().detach().numpy().reshape(-1,5))
            preds_list.append(preds.cpu().detach().numpy().reshape(-1,5))
            scores_list.append(scores.cpu().detach().numpy().reshape(-1,5))
            
    trues_all = np.vstack(trues_list)
    preds_all = np.vstack(preds_list)
    scores_all = np.vstack(scores_list)
    
    avg_loss = total_loss / len(val_iter)
    avg_metrics = get_metrics(trues_all[:,symptom_idx], preds_all[:,symptom_idx], scores_all[:,symptom_idx])
    
    return avg_loss, avg_metrics 
    
