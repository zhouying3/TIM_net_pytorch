"""
@author: Jiaxin Ye, modified by Ying Zhou
@contact: jiaxin-ye@foxmail.com, 442049887@qq.com
"""
# -*- coding:UTF-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
from Model_zy import TIMNET_Model
import argparse
import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import copy
from Model_zy import smooth_labels
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import random

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--model_path', type=str, default='./Models/')
parser.add_argument('--result_path', type=str, default='./Results/')
parser.add_argument('--test_path', type=str, default='./Test_Models/EMODB_46')
parser.add_argument('--data', type=str, default='EMODB')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.93)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--random_seed', type=int, default=46)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--filter_size', type=int, default=39)
parser.add_argument('--dilation_size', type=int, default=8)# If you want to train model on IEMOCAP, you should modify this parameter to 10 due to the long duration of speech signals.
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--stack_size', type=int, default=1)
parser.add_argument('--split_fold', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
# from monai.utils import set_determinism
# set_determinism(seed=0)

if args.data=="IEMOCAP" and args.dilation_size!=10:
    args.dilation_size = 10
    
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

CLASS_LABELS_finetune = ("angry", "fear", "happy", "neutral","sad")
CASIA_CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")#CASIA
EMODB_CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")#EMODB
SAVEE_CLASS_LABELS = ("angry","disgust", "fear", "happy", "neutral", "sad", "surprise")#SAVEE
RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")#rav
IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad")#IEMOCAP
EMOVO_CLASS_LABELS = ("angry", "disgust", "fear", "happy","neutral","sad","surprise")#emovo
MEAD_CLASS_LABELS = ("angry", "contempt","disgust", "fear", "happy","neutral","sad","surprise")#mead

CLASS_LABELS_dict = {"CASIA": CASIA_CLASS_LABELS,
               "EMODB": EMODB_CLASS_LABELS,
               "EMOVO": EMOVO_CLASS_LABELS,
               "IEMOCAP": IEMOCAP_CLASS_LABELS,
               "RAVDE": RAVDE_CLASS_LABELS,
               "SAVEE": SAVEE_CLASS_LABELS,
               "MEAD":MEAD_CLASS_LABELS}

data = np.load("./MFCC/"+args.data+".npy",allow_pickle=True).item() # 39 MFCC features
x_source = data["x"]
y_source = data["y"]
CLASS_LABELS = CLASS_LABELS_dict[args.data]

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train(args,x,y):    
    filepath = args.model_path
    resultpath = args.model_path
    
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)

    i=1
    now = datetime.datetime.now()
    now_time = datetime.datetime.strftime(now,'%Y-%m-%d_%H-%M-%S')
    kfold = KFold(n_splits=args.split_fold, shuffle=True, random_state=args.random_seed)
    avg_accuracy = 0
    avg_loss = 0
    eva_matrix = []
    matrix = []
    for train, test in kfold.split(x, y):
        model = TIMNET_Model(args=args, input_shape=x_source.shape[1:], class_label=CLASS_LABELS)
        model = torch.nn.DataParallel(model).cuda()
        y_train = smooth_labels(copy.deepcopy(y[train]), 0.1)
        folder_address = filepath+args.data+"_"+str(args.random_seed)+"_"+now_time
        if not os.path.exists(folder_address):
            os.mkdir(folder_address)
        weight_path=folder_address+'/'+str(args.split_fold)+"-fold_weights_best_"+str(i)+".hdf5"
        
        max_acc = 0
        best_eva_list = []
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, betas=(args.beta1,args.beta2), eps=1e-8)
        training_data = MyDataset(x[train],y_train)
        training_loader = DataLoader(training_data,batch_size=args.batch_size,shuffle=True)
        # testing_data = MyDataset(x[test],y[test])
        # testing_loader = DataLoader(testing_data)
        for _ in range(args.epoch):
            for input, target in training_loader:
                optimizer.zero_grad()
                # input = torch.tensor(np.ones_like(input),dtype=torch.float32)
                input, target = input.cuda(), target.cuda()
                y_pred = model(input)
                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()

        i+=1
        model.eval()
        y_pred_best = model(torch.Tensor(x[test]).cuda()).cpu().detach().numpy()
        matrix.append(confusion_matrix(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1)))
        em = classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=CLASS_LABELS,output_dict=True)
        eva_matrix.append(em)
        print(classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=CLASS_LABELS))

    
    for item in eva_matrix:
        avg_accuracy += item["accuracy"]
    
    print("Average ACC:",avg_accuracy/args.split_fold)
    acc = avg_accuracy/args.split_fold
    filename = resultpath+args.data+'_'+str(args.split_fold)+'fold_'+str(round(acc*10000)/100)+"_"+str(args.random_seed)+"_"+now_time+'.xlsx'

    for i,item in enumerate(matrix):
        temp = {}
        temp[" "] = CLASS_LABELS
        for j,l in enumerate(item):
            temp[CLASS_LABELS[j]]=item[j]
        data1 = pd.DataFrame(temp)
        df = pd.DataFrame(eva_matrix[i]).transpose()
        if i==0:
            with pd.ExcelWriter(filename, mode='w') as writer:
                data1.to_excel(writer,sheet_name=str(i))
                df.to_excel(writer,sheet_name=str(i)+"_evaluate")
        else:
            with pd.ExcelWriter(filename, mode='a') as writer:
                data1.to_excel(writer,sheet_name=str(i))            
                df.to_excel(writer,sheet_name=str(i)+"_evaluate")

    
    matrix = []
    eva_matrix = []

# x_feats and y_labels are test datas for t-sne
train(args,x_source,y_source)  