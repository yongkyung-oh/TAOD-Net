import os
import random

import numpy as np
import pandas as pd
import torch
import cv2

from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, mean_absolute_error


# check symptom information
symptoms_list = ['Red_lid', 'Red_conj','Swl_crncl', 'Swl_lid', 'Swl_conj']

# load label function
def load_labels(data_path, label_file):
    '''
    Load label file and preprocess for individual eye and patient
    '''

    # labels = pd.read_excel(os.path.join(data_path, label_file))
    labels = pd.read_csv(label_file)
    labels = labels.set_index('p_num')
    labels = labels.replace(0.5,1)
    
    # soft & hard labels + true labels
    labels_revised = []
    for symptom in symptoms_list:
        for loc in ['left', 'right']:
            c_names = [c for c in labels.columns if (symptom in c) & (loc in c) & ('MG' not in c)]
            soft_label = labels[c_names].mean(axis=1)
            hard_label = labels[c_names].mean(axis=1).round(0)

            c_names_MG = [c for c in labels.columns if (symptom in c) & (loc in c) & ('MG' in c)]
            true_label = labels[c_names_MG].squeeze()

            soft_label.name = '_'.join([symptom, loc, 'soft'])
            hard_label.name = '_'.join([symptom, loc, 'hard'])
            true_label.name = '_'.join([symptom, loc, 'true'])

            labels_revised.append(soft_label)
            labels_revised.append(hard_label)
            labels_revised.append(true_label)

    labels_revised = pd.concat(labels_revised, axis=1)
    labels_revised = labels_revised.dropna()

    # labels for patients 
    labels_patient = []
    for symptom in symptoms_list:
        patient_label = labels_revised[[c for c in labels_revised.columns if (symptom in c) & ('true' in c)]].sum(axis=1)
        patient_label = (patient_label>=1).astype(int)

        patient_label.name = '_'.join([symptom, 'patient'])
        labels_patient.append(patient_label)

    labels_patient = pd.concat(labels_patient, axis=1)
    
    return labels_revised, labels_patient


# load landmark function
def load_landmark(data_path, landmark_file):
    '''
    load facial landmark file and align it with left and right eye
    '''    
    
    # landmarks = pd.read_excel(os.path.join(data_path, landmark_file))
    landmarks = pd.read_csv(landmark_file)
    landmarks = landmarks.set_index('p_num')
    
    idx_left = [18, 19, 20, 21, 22, 37, 38, 39, 40, 41, 42]
    idx_right = [27, 26, 25, 24, 23, 46, 45, 44, 43, 48, 47]

    landmarks_left = landmarks[['x_'+str(i) for i in idx_left] + ['y_'+str(i) for i in idx_left]]
    landmarks_right = landmarks[['x_'+str(i) for i in idx_right] + ['y_'+str(i) for i in idx_right]]

    landmarks_left.columns = ['x_'+str(i) for i in np.arange(1,12)] + ['y_'+str(i) for i in np.arange(1,12)]
    landmarks_right.columns = ['x_'+str(i) for i in np.arange(1,12)] + ['y_'+str(i) for i in np.arange(1,12)]

    # get pupil center coords
    landmarks_left.loc[:, 'x_c'] = landmarks_left[['x_'+str(i) for i in [7,8,10,11]]].copy().mean(axis=1).round(0)
    landmarks_left.loc[:, 'y_c'] = landmarks_left[['y_'+str(i) for i in [7,8,10,11]]].copy().mean(axis=1).round(0)
    landmarks_right.loc[:, 'x_c'] = landmarks_right[['x_'+str(i) for i in [7,8,10,11]]].copy().mean(axis=1).round(0)
    landmarks_right.loc[:, 'y_c'] = landmarks_right[['y_'+str(i) for i in [7,8,10,11]]].copy().mean(axis=1).round(0)

    # additional landmark (inner)
    landmarks_left.loc[:, 'x_n'] = landmarks_left[['x_'+str(i) for i in [5,9]]].copy().mean(axis=1).round(0)
    landmarks_left.loc[:, 'y_n'] = landmarks_left[['y_'+str(i) for i in [9]]].copy().mean(axis=1).round(0)
    landmarks_right.loc[:, 'x_n'] = landmarks_right[['x_'+str(i) for i in [5,9]]].copy().mean(axis=1).round(0)
    landmarks_right.loc[:, 'y_n'] = landmarks_right[['y_'+str(i) for i in [9]]].copy().mean(axis=1).round(0)

    # additional landmark (outer)
    landmarks_left.loc[:, 'x_m'] = landmarks_left[['x_'+str(i) for i in [1,6]]].copy().mean(axis=1).round(0)
    landmarks_left.loc[:, 'y_m'] = landmarks_left[['y_'+str(i) for i in [6]]].copy().mean(axis=1).round(0)
    landmarks_right.loc[:, 'x_m'] = landmarks_right[['x_'+str(i) for i in [1,6]]].copy().mean(axis=1).round(0)
    landmarks_right.loc[:, 'y_m'] = landmarks_right[['y_'+str(i) for i in [6]]].copy().mean(axis=1).round(0)

    landmarks_left = landmarks_left.astype(int)
    landmarks_right = landmarks_right.astype(int)
    
    return landmarks_left, landmarks_right


# image crop function
def get_img_cropped(data_path, pid, landmarks, mode=0, bound=0.1, dim=(512,512)):
    '''
    define image crop function
    mode: 0(raw), 1(whole eye), 2(inner eye), 3(outer eye), 4(upper eye), 5(lower eye), 6(eyelid), 7(orbital)
    '''

    img = cv2.imread(os.path.join(data_path, str(pid)+'.jpg'))
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        # print(os.path.join(data_path, str(pid) + '.jpg'))
        pass

    if mode==0:   # raw image
        landmarks_idx=[1,2,3,4,5,6,7,8,9,10,11,'c']       
    elif mode==1: # whole eye
        landmarks_idx=[6,7,8,9,10,11,'c']
    elif mode==2: # inner eye 
        landmarks_idx=[8,9,10,'c']       
    elif mode==3: # outer eye 
        landmarks_idx=[6,7,11,'c']       
    elif mode==4: # upper eye 
        landmarks_idx=[6,7,8,'c']       
    elif mode==5: # lower eye 
        landmarks_idx=[6,10,11,'c']       
    elif mode==6: # eyelid 
        landmarks_idx=[1,5,7,8]
    elif mode==7: # orbital region
        landmarks_idx=[6,7,8,9,10,11,'c']

    coords = []
    for i in landmarks_idx:
        x = landmarks.loc[pid]['x_{}'.format(i)]
        y = landmarks.loc[pid]['y_{}'.format(i)]
        coords.append([x,y])
    coords = np.array(coords)
    
    d_x = abs(coords[:,1].max()-coords[:,1].min())
    d_y = abs(coords[:,0].max()-coords[:,0].min())
    
    # bounding box
    if mode == 0 or mode == 7: 
        c_x_1 = max(round(coords[:,1].min()-1.5*d_x), 0)
        c_x_2 = max(round(coords[:,1].max()+1.5*d_x), 0)
        c_y_1 = max(round(coords[:,0].min()-0.5*d_y), 0)
        c_y_2 = max(round(coords[:,0].max()+0.5*d_y), 0)
    else: 
        c_x_1 = max(round(coords[:,1].min()-bound*d_x), 0)
        c_x_2 = max(round(coords[:,1].max()+bound*d_x), 0)
        c_y_1 = max(round(coords[:,0].min()-bound*d_y), 0)
        c_y_2 = max(round(coords[:,0].max()+bound*d_y), 0)
    
    img_c = img[c_x_1:c_x_2, c_y_1:c_y_2]
    
    # remove black region
    mask = (img_c.mean(axis=2)!=0)
    img_c = img_c[np.mean(mask,axis=1)!=0, :, :]
    img_c = img_c[:, np.mean(mask,axis=0)!=0, :]
    
    # border thresholds
    img_c = img_c[5:-5, 5:-5]
    
    # resize
    img_c = cv2.resize(img_c, dim, interpolation=cv2.INTER_AREA)
    
    return img_c


# get classification metrics
def get_metrics(real_values, pred_values, score_values):
    CM = confusion_matrix(real_values,pred_values)
    TN = CM[0][0]
    FN = CM[1][0] 
    TP = CM[1][1]
    FP = CM[0][1]
    Population = TN+FN+TP+FP
    Accuracy   = round((TP+TN) / Population, 4)
    Precision  = round(TP / (TP+FP), 4)
    Recall     = round(TP / (TP+FN), 4) #Sensitivity
    # F1         = round( 2 * ((Precision*Recall)/(Precision+Recall)), 4)
    F1 = f1_score(real_values,pred_values, zero_division=0)
    FPR        = round(FP / (TN+FP), 4) #Fall-out
    sensitivity = Recall
    specificity = 1-FPR
    
    bac = balanced_accuracy_score(real_values, pred_values)
    wF1 = f1_score(real_values,pred_values, average='weighted', zero_division=0)
    auc = roc_auc_score(real_values, score_values)
    
    return  [Accuracy, Precision, Recall, F1, sensitivity, specificity, bac, wF1, auc]

