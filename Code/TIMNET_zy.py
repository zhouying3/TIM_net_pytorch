"""
@author: Jiaxin Ye, modified by Ying Zhou
@contact: jiaxin-ye@foxmail.com, 442049887@qq.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # self.conv1 = weight_norm()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs,momentum=0.99,eps=0.001)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        self.conv2 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs,momentum=0.99,eps=0.001)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        # nn.init.xavier_normal_(self.conv1.weight)
        # nn.init.xavier_normal_(self.conv2.weight)
        # if self.downsample is not None:
        #     nn.init.xavier_normal_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)        
        res = x if self.downsample is None else self.downsample(x)
        # return self.sigmoid(out*res)
        out = self.sigmoid(out)
        return out*res


class ChannelAvgPool(nn.Module):
    def __init__(self):
        super(ChannelAvgPool, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1).unsqueeze(1)

class TIMNET(nn.Module):
    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=None,
                 activation = "relu",
                 dropout_rate=0.1,
                 return_sequences=True,
                 name='TIMNET'):
        super(TIMNET, self).__init__()
        self.name = name
        self.return_sequences = return_sequences
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        self.supports_masking = True
        self.mask_value=0.

        if not isinstance(nb_filters, int):
            raise Exception()
        # 这俩的功能是什么？
        self.forward_convd = nn.Conv1d(self.nb_filters, self.nb_filters,kernel_size=1, dilation=1) 
        self.backward_convd = nn.Conv1d(self.nb_filters, self.nb_filters,kernel_size=1, dilation=1)
        
        
        for s in range(self.nb_stacks):
            # for i in [2 ** i for i in range(self.dilations)]:
            # self.blocks_forward = nn.ModuleList([Temporal_Aware_Block(s, 2 ** i, activation, nb_filters, kernel_size, dropout_rate) for i in range(dilations)])
                
            # self.blocks_backward = nn.ModuleList([Temporal_Aware_Block(s, 2 ** i, activation, nb_filters, kernel_size, dropout_rate) for i in range(dilations)])
            self.blocks_forward = nn.ModuleList([TemporalBlock(self.nb_filters,self.nb_filters,kernel_size=kernel_size, stride=1, dilation=i, padding=(kernel_size-1) *i, dropout=dropout_rate) for i in [2 ** i for i in range(self.dilations)]] )

            self.blocks_backward = nn.ModuleList([TemporalBlock(self.nb_filters,self.nb_filters,kernel_size=kernel_size, stride=1, dilation=i, padding=(kernel_size-1) *i, dropout=dropout_rate) for i in [2 ** i for i in range(self.dilations)]] )


                
        self.pool = ChannelAvgPool()   

    def forward(self, inputs, mask=None):
        if self.dilations is None:
            self.dilations = 8
        forward_data = inputs.transpose(1,2)
        backward_data = torch.flip(inputs, dims=[1]).transpose(1,2)	
        
        
        # print("Input Shape=",inputs.shape)
        
        
        
        skip_out_forward = self.forward_convd(forward_data)
        skip_out_backward = self.backward_convd(backward_data)
        
        final_skip_connection = []
        for block_forward, block_backward in zip(self.blocks_forward, self.blocks_backward):
            x1 = block_forward(skip_out_forward)
            x2 = block_backward(skip_out_backward) # None,39,188
            x2 = self.pool(torch.concatenate([x1,x2],axis=2).transpose(1,2)).transpose(1,2)
            # None,39,1
            final_skip_connection.append(x2)          
            
        output_2 = final_skip_connection[0]
        for i,item in enumerate(final_skip_connection):
            if i==0:
                continue
            output_2 = torch.concatenate([output_2,item],axis=2)
        x = output_2        

        return x
