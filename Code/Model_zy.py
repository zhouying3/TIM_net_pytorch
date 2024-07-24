"""
@author: Jiaxin Ye, modified by Ying Zhou
@contact: jiaxin-ye@foxmail.com, 442049887@qq.com
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


from sklearn.metrics import confusion_matrix
from Common_Model import Common_Model
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime
import pandas as pd
import copy

from TIMNET_zy import TIMNET


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

class WeightLayer(nn.Module):
    def __init__(self, input_shape):
        super(WeightLayer, self).__init__()
        self.kernel = torch.nn.Linear(input_shape,1, bias=False)
            
 
    def init_weights(self):
        self.kernel.weight.data.uniform_(0, 0.01)

    def forward(self, x):
        # tempx = x.transpose(1,2)
        x = self.kernel(x)
        x = torch.squeeze(x,axis=-1)
        return  x
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])
    
def softmax(x, axis=-1):
    ex = torch.exp(x - torch.max(x, axis=axis, keepdims=True))
    return ex/torch.sum(ex, axis=axis, keepdims=True)

class TIMNET_Model(nn.Module):
    def __init__(self, args, input_shape, class_label, **params):
        super(TIMNET_Model,self).__init__(**params)
        self.args = args
        self.data_shape = input_shape
        self.num_classes = len(class_label)
        self.class_label = class_label
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        print("TIMNET MODEL SHAPE:",input_shape)
    
        
        self.multi_decision = TIMNET(nb_filters=self.args.filter_size,
                                kernel_size=self.args.kernel_size, 
                                nb_stacks=self.args.stack_size,
                                dilations=self.args.dilation_size,
                                dropout_rate=self.args.dropout,
                                activation = self.args.activation,
                                return_sequences=True, 
                                name='TIMNET')

        self.decision = WeightLayer(self.args.dilation_size)
        self.predictions = torch.nn.Linear(self.data_shape[1],self.num_classes)
        self.softmax_fn = nn.Softmax(dim=-1)   
        
    def forward(self, x): 
        x = self.multi_decision(x)
        x = self.decision(x)
        x = self.softmax_fn(self.predictions(x))
        return x      
