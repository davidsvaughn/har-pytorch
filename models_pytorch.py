'''
PYTORCH CNN MODELS fro SPEED PREDICTION
'''
import sys
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

from nnAudio import Spectrogram

####################################################################

def create_filters(d,n=1):
    x = np.arange(0, d, 1)
    wsin = np.empty((d,n,d), dtype=np.float32)
    wcos = np.empty((d,n,d), dtype=np.float32)
    for i in range(d):
        for j in range(n):
            wsin[i,j,:] = np.sin(2*np.pi*((i+1)/d)*x)
            wcos[i,j,:] = np.cos(2*np.pi*((i+1)/d)*x)
    return wsin,wcos


''' CNN MODELS '''

class CNN_1(torch.nn.Module):
    
    #input x is [25, 8, 30]
    def __init__(self, cfg, x):
        super(CNN_1, self).__init__()
        
        bs = cfg.batch_size
        win = cfg.window_size
        c = cfg.n_inputs
        d = 16
        
        wsin, wcos = create_filters(d, c)
        self.wsin_var = Variable(torch.from_numpy(wsin), requires_grad=cfg.fft_grad)
        self.wcos_var = Variable(torch.from_numpy(wcos), requires_grad=cfg.fft_grad)
        
        x = torch.conv1d(x, self.wsin_var, stride=2).pow(2)
        print(x.shape)
        
        self.conv2 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1).float()
        self.conv2_drop = torch.nn.Dropout(p=cfg.dropout)
        
        x = F.relu(self.conv2(x))
        print(x.shape)
        
        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=1).float()
        self.conv3_drop = torch.nn.Dropout(p=cfg.dropout)
        
        x = F.relu(self.conv3(x))
        print(x.shape)
        
        self.lin1 = torch.nn.Linear(64, 8)
        self.lin1_drop = torch.nn.Dropout(p=cfg.dropout)
        
        self.lin2 = torch.nn.Linear(8, 1)
        
    def forward(self, x):
        #bs = x.shape[0]
        
        x1 = torch.conv1d(x, self.wsin_var, stride=2)#.pow(2)
        x2 = torch.conv1d(x, self.wcos_var, stride=2)#.pow(2)
        x = x1 + x2
        x = F.relu(x)
        
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)
        
        x = F.relu(self.conv3(x))
        x = self.conv3_drop(x)
        
        x = x.view(-1, 64)
        
        x = F.relu(self.lin1(x))
        x = self.lin1_drop(x)
        
        x = self.lin2(x)
        x = F.relu(x)
        #x = F.tanh(x)
    
        return(x)

class CNN_2(torch.nn.Module):
    
    #input x is [25, 8, 30]
    def __init__(self, cfg, x):
        super(CNN_2, self).__init__()
        
        bs = cfg.batch_size
        win = cfg.window_size
        c = cfg.n_inputs
        d = 10
        f = 32
        
        wsin, wcos = create_filters(d, c)
        self.wsin_var = Variable(torch.from_numpy(wsin), requires_grad=cfg.fft_grad)
        self.wcos_var = Variable(torch.from_numpy(wcos), requires_grad=cfg.fft_grad)
        
        x = torch.conv1d(x, self.wsin_var, stride=2)
        print(x.shape)
        
        self.conv2 = torch.nn.Conv1d(in_channels=d, out_channels=f, kernel_size=5, stride=1).float()
        self.conv2_drop = torch.nn.Dropout(p=cfg.dropout)
        
        x = self.conv2(x)
        print(x.shape)
        
        self.conv3 = torch.nn.Conv1d(in_channels=f, out_channels=f*2, kernel_size=4, stride=1).float()
        self.conv3_drop = torch.nn.Dropout(p=cfg.dropout)
        f*=2
        
        x = self.conv3(x)
        print(x.shape)
        
        self.conv4= torch.nn.Conv1d(in_channels=f, out_channels=f*2, kernel_size=4, stride=1).float()
        self.conv4_drop = torch.nn.Dropout(p=cfg.dropout)
        f*=2
        self.f = f
        
        x = self.conv4(x)
        print(x.shape)
        
        x = x.view(-1, f)
        print(x.shape)
        
        n = 32
        
        self.lin1 = torch.nn.Linear(f, n)
        self.lin1_drop = torch.nn.Dropout(p=cfg.dropout)
        
        x = self.lin1(x)
        print(x.shape)
        
        self.lin2 = torch.nn.Linear(n, 1)
        
        x = self.lin2(x)
        print(x.shape)
        
    def forward(self, x):
        #bs = x.shape[0]
        
        x1 = torch.conv1d(x, self.wsin_var, stride=2)
        x2 = torch.conv1d(x, self.wcos_var, stride=2)
        x = x1 + x2
        x = F.relu(x)
        
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)
        
        x = F.relu(self.conv3(x))
        x = self.conv3_drop(x)
        
        x = F.relu(self.conv4(x))
        x = self.conv4_drop(x)
        
        x = x.view(-1, self.f)
        
        x = F.relu(self.lin1(x))
        x = self.lin1_drop(x)
        
        x = self.lin2(x)
        x = F.relu(x)
    
        return(x) 


#####################################################

'''
torch.Size([25, 8, 12])
torch.Size([25, 32, 7])
torch.Size([25, 64, 4])
torch.Size([25, 128, 1])
torch.Size([25, 128])
torch.Size([25, 32])
torch.Size([25, 1])

MODEL SHAPES
torch.Size([8, 8, 8])
torch.Size([8])
torch.Size([32, 8, 6])
torch.Size([32])
torch.Size([64, 32, 4])
torch.Size([64])
torch.Size([128, 64, 4])
torch.Size([128])
torch.Size([32, 128])
torch.Size([32])
torch.Size([1, 32])
torch.Size([1])

'''
    
class CNN_3(torch.nn.Module):
    
    #input x is [25, 8, 30]
    def __init__(self, cfg, x):
        super(CNN_3, self).__init__()
        
        bs = cfg.batch_size
        win = cfg.window_size
        c = cfg.n_inputs
        d = 8
        f = 32
        
        self.conv1 = torch.nn.Conv1d(in_channels=c, out_channels=d, kernel_size=d, stride=2).float()
        self.conv1_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv1(x)
        print(x.shape)
        
        self.conv2 = torch.nn.Conv1d(in_channels=d, out_channels=f, kernel_size=6, stride=1).float()
        self.conv2_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv2(x)
        print(x.shape)
        
        self.conv3 = torch.nn.Conv1d(in_channels=f, out_channels=f*2, kernel_size=4, stride=1).float()
        self.conv3_drop = torch.nn.Dropout(p=cfg.dropout)
        f*=2
        x = self.conv3(x)
        print(x.shape)
        
        self.conv4= torch.nn.Conv1d(in_channels=f, out_channels=f*2, kernel_size=4, stride=1).float()
        self.conv4_drop = torch.nn.Dropout(p=cfg.dropout)
        f*=2
        self.f = f
        x = self.conv4(x)
        print(x.shape)
        
        x = x.view(-1, f)
        print(x.shape)
        
        n = 32
        
        self.lin1 = torch.nn.Linear(f, n)
        self.lin1_drop = torch.nn.Dropout(p=cfg.dropout)
        
        x = self.lin1(x)
        print(x.shape)
        
        self.lin2 = torch.nn.Linear(n, 1)
        
        x = self.lin2(x)
        print(x.shape)
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.conv1_drop(x)
        
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)
        
        x = F.relu(self.conv3(x))
        x = self.conv3_drop(x)
        
        x = F.relu(self.conv4(x))
        x = self.conv4_drop(x)
        
        x = x.view(-1, self.f)
        
        x = F.relu(self.lin1(x))
        x = self.lin1_drop(x)
        
        x = self.lin2(x)
        x = F.relu(x)
    
        return(x)
    
class CNN_4(torch.nn.Module):
    
    #input x is [25, 8, 30]
    def __init__(self, cfg, x):
        super(CNN_4, self).__init__()
        
        bs = cfg.batch_size
        win = cfg.window_size
        c = cfg.n_inputs
        d = 8
        f = 32
        
        self.conv1 = torch.nn.Conv1d(in_channels=c, out_channels=f, kernel_size=d, stride=2).float()
        self.conv1_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv1(x)
        print(x.shape)
        
        f*=2
        self.conv2 = torch.nn.Conv1d(in_channels=f//2, out_channels=f, kernel_size=6, stride=1).float()
        self.conv2_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv2(x)
        print(x.shape)
        
        f*=2
        self.conv3 = torch.nn.Conv1d(in_channels=f//2, out_channels=f, kernel_size=4, stride=1).float()
        self.conv3_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv3(x)
        print(x.shape)
        
        f*=2
        self.conv4= torch.nn.Conv1d(in_channels=f//2, out_channels=f, kernel_size=4, stride=1).float()
        self.conv4_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv4(x)
        print(x.shape)
        
        self.f = f
        x = x.view(-1, f)
        print(x.shape)
        
        n = 32
        
        self.lin1 = torch.nn.Linear(f, n)
        self.lin1_drop = torch.nn.Dropout(p=cfg.dropout)
        
        x = self.lin1(x)
        print(x.shape)
        
        self.lin2 = torch.nn.Linear(n, 1)
        
        x = self.lin2(x)
        print(x.shape)
        
    def forward(self, x):
       
        x = F.relu(self.conv1(x))
        x = self.conv1_drop(x)
        
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)
        
        x = F.relu(self.conv3(x))
        x = self.conv3_drop(x)
        
        x = F.relu(self.conv4(x))
        x = self.conv4_drop(x)
        
        x = x.view(-1, self.f)
        
        x = F.relu(self.lin1(x))
        x = self.lin1_drop(x)
        
        x = self.lin2(x)
        x = F.relu(x)
    
        return(x) 
    
class CNN_5(torch.nn.Module):
    
    #input x is [25, 8, 30]
    def __init__(self, cfg, x):
        super(CNN_5, self).__init__()
        
        bs = cfg.batch_size
        win = cfg.window_size
        c = cfg.n_inputs
        d = 8
        f = 32
        
        self.conv1 = torch.nn.Conv1d(in_channels=c, out_channels=f, kernel_size=d, stride=2).float()
        self.conv1_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv1(x)
        print(x.shape)
        
        f*=2
        self.conv2 = torch.nn.Conv1d(in_channels=f//2, out_channels=f, kernel_size=6, stride=2).float()
        self.conv2_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv2(x)
        print(x.shape)
        
        f*=2
        self.conv3 = torch.nn.Conv1d(in_channels=f//2, out_channels=f, kernel_size=4, stride=1).float()
        self.conv3_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv3(x)
        print(x.shape)

#         f*=2
#         self.conv4= torch.nn.Conv1d(in_channels=f//2, out_channels=f, kernel_size=4, stride=1).float()
#         self.conv4_drop = torch.nn.Dropout(p=cfg.dropout)
#         x = self.conv4(x)
#         print(x.shape)
        
        self.f = f
        x = x.view(-1, f)
        print(x.shape)
        
        n = 32
        
        self.lin1 = torch.nn.Linear(f, n)
        self.lin1_drop = torch.nn.Dropout(p=cfg.dropout)
        
        x = self.lin1(x)
        print(x.shape)
        
        self.lin2 = torch.nn.Linear(n, 1)
        
        x = self.lin2(x)
        print(x.shape)
        
    def forward(self, x):
       
        x = F.relu(self.conv1(x))
        x = self.conv1_drop(x)
        
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)
        
        x = F.relu(self.conv3(x))
        x = self.conv3_drop(x)
        
#         x = F.relu(self.conv4(x))
#         x = self.conv4_drop(x)
        
        x = x.view(-1, self.f)
        
        x = F.relu(self.lin1(x))
        x = self.lin1_drop(x)
        
        x = self.lin2(x)
        x = F.relu(x)
    
        return(x) 

class CNN_6(torch.nn.Module):
    
    #input x is [25, 8, 20]
    def __init__(self, cfg, x):
        super(CNN_6, self).__init__()
        
        bs = cfg.batch_size
        win = cfg.window_size
        c = cfg.n_inputs
        f = 32
        
        self.conv1 = torch.nn.Conv1d(in_channels=c, out_channels=f, kernel_size=6, stride=2).float()
        self.conv1_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv1(x)
        print(x.shape)
        
        f*=2
        self.conv2 = torch.nn.Conv1d(in_channels=f//2, out_channels=f, kernel_size=5, stride=1).float()
        self.conv2_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv2(x)
        print(x.shape)
        
        f*=2
        self.conv3 = torch.nn.Conv1d(in_channels=f//2, out_channels=f, kernel_size=4, stride=1).float()
        self.conv3_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv3(x)
        print(x.shape)

#         f*=2
#         self.conv4= torch.nn.Conv1d(in_channels=f//2, out_channels=f, kernel_size=4, stride=1).float()
#         self.conv4_drop = torch.nn.Dropout(p=cfg.dropout)
#         x = self.conv4(x)
#         print(x.shape)
        
        self.f = f
        x = x.view(-1, f)
        print(x.shape)
        
        n = 8
        
        self.lin1 = torch.nn.Linear(f, n)
        self.lin1_drop = torch.nn.Dropout(p=cfg.dropout)
        
        x = self.lin1(x)
        print(x.shape)
        
        self.lin2 = torch.nn.Linear(n, 1)
        
        x = self.lin2(x)
        print(x.shape)
        
    def forward(self, x):
       
        x = F.relu(self.conv1(x))
        x = self.conv1_drop(x)
        
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)
        
        x = F.relu(self.conv3(x))
        x = self.conv3_drop(x)
        
#         x = F.relu(self.conv4(x))
#         x = self.conv4_drop(x)
        
        x = x.view(-1, self.f)
        
        x = F.relu(self.lin1(x))
        x = self.lin1_drop(x)
        
        x = self.lin2(x)
        x = F.relu(x)
    
        return(x) 
    
class CNN_7(torch.nn.Module):
    
    #input x is [25, 8, 20]
    def __init__(self, cfg, x):
        super(CNN_7, self).__init__()
        
        bs = cfg.batch_size
        c = cfg.n_inputs
        K = cfg.cnn
        win,f = K[0]
        n = K[-1]
        
        k=K[1]
        self.conv1 = torch.nn.Conv1d(in_channels=c, out_channels=f, kernel_size=k[0], stride=k[1]).float()
        self.conv1_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv1(x)
        print(x.shape)
        f*=2
        
        k=K[2]
        self.conv2 = torch.nn.Conv1d(in_channels=f//2, out_channels=f, kernel_size=k[0], stride=k[1]).float()
        self.conv2_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv2(x)
        print(x.shape)
        f*=2
        
        k=K[3]
        self.conv3 = torch.nn.Conv1d(in_channels=f//2, out_channels=f, kernel_size=k[0], stride=k[1]).float()
        self.conv3_drop = torch.nn.Dropout(p=cfg.dropout)
        x = self.conv3(x)
        print(x.shape)
        f*=2

        f=f//2
        self.f = f
        x = x.view(-1, f)
        print(x.shape)
        
        self.lin1 = torch.nn.Linear(f, n)
        self.lin1_drop = torch.nn.Dropout(p=cfg.dropout)
        
        x = self.lin1(x)
        print(x.shape)
        
        self.lin2 = torch.nn.Linear(n, 1)
        
        x = self.lin2(x)
        print(x.shape)
        
    def forward(self, x):
       
        x = F.relu(self.conv1(x))
        x = self.conv1_drop(x)
        
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)
        
        x = F.relu(self.conv3(x))
        x = self.conv3_drop(x)
        
#         x = F.relu(self.conv4(x))
#         x = self.conv4_drop(x)
        
        x = x.view(-1, self.f)
        
        x = F.relu(self.lin1(x))
        x = self.lin1_drop(x)
        
        x = self.lin2(x)
        x = F.relu(x)
    
        return(x) 

class CNN_0(torch.nn.Module):
    
    def __init__(self, cfg, x):
        super(CNN_0, self).__init__()
        
        c = cfg.n_inputs
        K = cfg.cnn
        win,f = K[0]
        fin,fout = c,f
        self.j = -1
        
        print(x.shape)
        
        i, conv_layers = 1,[]
        while isinstance(K[i], (list, tuple)):
            k,i = K[i],i+1
            ksize, kstride = abs(k[0]),k[1]
            ## FFT ? #################
            if k[0]<0:
                self.j = i-1
            ##########################
            conv = torch.nn.Conv1d(in_channels=fin, out_channels=fout, kernel_size=ksize, stride=kstride).float()
            conv_layers.append(conv)
            fin,fout = fout,fout*2
            x = conv(x)
            print(x.shape)
        
        f = fin
        self.f = f
        x = x.view(-1, f)
        print(x.shape)
        
        n1 = f
        lin_layers = []
        while i<len(K):
            n2,i = K[i],i+1
            lin = torch.nn.Linear(n1, n2)
            lin_layers.append(lin)
            n1 = n2
            x = lin(x)
            print(x.shape)
        
        lin = torch.nn.Linear(n1, 1)
        lin_layers.append(lin)
        x = lin(x)
        print(x.shape)
        
        self.conv_layers = nn.ModuleList(conv_layers)
        self.lin_layers = nn.ModuleList(lin_layers)
        self.drop = nn.Dropout(p=cfg.dropout)
        
    def forward(self, x):
        
        for j,conv in enumerate(self.conv_layers):
            if j==self.j:
                x = torch.rfft(x, 2, onesided=False)[:,:,:,0]
            x = F.relu(conv(x))
            x = self.drop(x)
        
        x = x.view(-1, self.f)
        
        for i,lin in enumerate(self.lin_layers):
            x = lin(x)
            if i+1==len(self.lin_layers): break
            x = self.drop(F.relu(x))
    
        return(x)

''' merge before FIRST linear layer ''' 
class CNN_10(torch.nn.Module):
    
    def __init__(self, cfg, x):
        super(CNN_10, self).__init__()
        
        bs = cfg.batch_size
        self.c = c = cfg.n_inputs//2
        K = cfg.cnn
        win,f = K[0]
        fin,fout = c,f
        
        x = x[:, :c, :]
        print(x.shape)
        
        i, conv_layers_1, conv_layers_2 = 1,[],[]
        while isinstance(K[i], (list, tuple)):
            k,i = K[i],i+1
            conv1 = torch.nn.Conv1d(in_channels=fin, out_channels=fout, kernel_size=k[0], stride=k[1]).float()
            conv_layers_1.append(conv1)
            conv2 = torch.nn.Conv1d(in_channels=fin, out_channels=fout, kernel_size=k[0], stride=k[1]).float()
            conv_layers_2.append(conv2)
            fin,fout = fout,fout*2
            x = conv1(x)
            print(x.shape)
        
        f = fin
        self.f = f
        x = x.view(-1, f)
        print(x.shape)
        
        x = torch.cat((x,x),1)
        print(x.shape)
        
        n1 = f*2
        lin_layers = []
        while i<len(K):
            n2,i = K[i],i+1
            lin = torch.nn.Linear(n1, n2)
            lin_layers.append(lin)
            n1 = n2
            x = lin(x)
            print(x.shape)
        
        lin = torch.nn.Linear(n1, 1)
        lin_layers.append(lin)
        x = lin(x)
        print(x.shape)
        
        self.conv_layers_1 = nn.ModuleList(conv_layers_1)
        self.conv_layers_2 = nn.ModuleList(conv_layers_2)
        self.lin_layers = nn.ModuleList(lin_layers)
        self.drop = nn.Dropout(p=cfg.dropout)
        
    def forward(self, x):
        c = self.c
        x1 = x[:, :c, :]
        x2 = x[:, c:, :]
        
        for conv in self.conv_layers_1:
            x1 = F.relu(conv(x1))
            x1 = self.drop(x1)
            
        for conv in self.conv_layers_2:
            x2 = F.relu(conv(x2))
            x2 = self.drop(x2)
        
        x1 = x1.view(-1, self.f)
        x2 = x2.view(-1, self.f)
        x = torch.cat((x1,x2),1)
        
        for i,lin in enumerate(self.lin_layers):
            x = lin(x)
            if i+1==len(self.lin_layers): break
            x = self.drop(F.relu(x))
    
        return(x)


''' merge before LAST linear layer '''      
class CNN_11(torch.nn.Module):
    
    def __init__(self, cfg, x):
        super(CNN_11, self).__init__()
        
        bs = cfg.batch_size
        self.c = c = cfg.n_inputs//2
        K = cfg.cnn
        win,f = K[0]
        fin,fout = c,f
        
        x = x[:, :c, :]
        print(x.shape)
        
        i, conv_layers_1, conv_layers_2 = 1,[],[]
        while isinstance(K[i], (list, tuple)):
            k,i = K[i],i+1
            conv1 = torch.nn.Conv1d(in_channels=fin, out_channels=fout, kernel_size=k[0], stride=k[1]).float()
            conv_layers_1.append(conv1)
            conv2 = torch.nn.Conv1d(in_channels=fin, out_channels=fout, kernel_size=k[0], stride=k[1]).float()
            conv_layers_2.append(conv2)
            fin,fout = fout,fout*2
            x = conv1(x)
            print(x.shape)
        
        f = fin
        self.f = f
        x = x.view(-1, f)
        print(x.shape)
        
#        x = torch.cat((x,x),1)
#        print(x.shape)
        
#        n1 = f*2
        n1 = f
        lin_layers_1, lin_layers_2 = [],[]
        while i<len(K):
            n2,i = K[i],i+1
            lin1 = torch.nn.Linear(n1, n2)
            lin_layers_1.append(lin1)
            lin2 = torch.nn.Linear(n1, n2)
            lin_layers_2.append(lin2)
            n1 = n2
            x = lin1(x)
            print(x.shape)
        
        n1 = n1*2
        last_layer = torch.nn.Linear(n1, 1)
        
        x = torch.cat((x,x),1)
        print(x.shape)
        x = last_layer(x)
        print(x.shape)
        
        self.conv_layers_1 = nn.ModuleList(conv_layers_1)
        self.conv_layers_2 = nn.ModuleList(conv_layers_2)
        self.lin_layers_1 = nn.ModuleList(lin_layers_1)
        self.lin_layers_2 = nn.ModuleList(lin_layers_2)
        self.last_layer = last_layer
        self.drop = nn.Dropout(p=cfg.dropout)
        
    def forward(self, x):
        c = self.c
        x1 = x[:, :c, :]
        x2 = x[:, c:, :]
        
        for conv in self.conv_layers_1:
            x1 = F.relu(conv(x1))
            x1 = self.drop(x1)
            
        for conv in self.conv_layers_2:
            x2 = F.relu(conv(x2))
            x2 = self.drop(x2)
        
        x1 = x1.view(-1, self.f)
        x2 = x2.view(-1, self.f)
#        x = torch.cat((x1,x2),1)
        
        for i,lin in enumerate(self.lin_layers_1):
            x1 = F.relu(lin(x1))
            x1 = self.drop(x1)
        
        for i,lin in enumerate(self.lin_layers_2):
            x2 = F.relu(lin(x2))
            x2 = self.drop(x2)
            
        x = torch.cat((x1,x2),1)
        
        x = self.last_layer(x)
    
        return(x)

## 
def torch_mag(x, axis, keepdim=False):
    m = torch.sum(torch.pow(x, 2), axis, keepdim=keepdim)
    m = torch.sqrt(m)
    return m

def preproc(x, cfg):
    G,A = x[:,:3,:], x[:,3:,:]
    
    ## add magnitude of accel and gyro...
    if cfg.add_mag:
        a = torch.norm(A, dim=1, keepdim=True)
        g = torch.norm(G, dim=1, keepdim=True)
        x = torch.cat((G, g, A, a), 1)
    
    if cfg.mod.endswith('b'):
        return x
    
    ## add FFT features...
    if cfg.feats_fft:
        f = torch.rfft(x, 1, onesided=False)
        f = torch.norm(f, dim=-1)
        
        if cfg.feats_raw:
            x = torch.cat((x, f), 1)
        else:
            x = f
    
    return x

def get_conv_layer(k, fin, fout):
    if k[0]>0:
        return torch.nn.Conv1d(in_channels=fin, out_channels=fout, kernel_size=k[0], stride=k[1]).float()
    else:
        return torch.nn.MaxPool1d(kernel_size=-k[0], stride=k[1]).float()
    
def apply_conv_layer(conv, x, bn=None):
    x = conv(x)
    if bn is not None:
        x = bn(x)
    if type(conv).__name__.startswith('Conv'):
        x = F.relu(x)
    return x
    

''' SPLITS into 2 : Time/Freq  --OR--  G/A '''
''' merge before NEGATIVE linear layer '''      
class CNN_12(torch.nn.Module):
    
    def __init__(self, cfg, x, output_dim=1):
        super(CNN_12, self).__init__()
        self.cfg = cfg
        
        if cfg.gpu_preproc:
            x = preproc(x, cfg)

        self.c = c = x.shape[1]//2
        x = x[:, :c, :]
            
        print(x.shape)
        
        K = cfg.cnn
        win,f = K[0]
        fin,fout = c,f
        
        i, conv_layers_1, conv_layers_2 = 1,[],[]
        while isinstance(K[i], (list, tuple)):
            k,i = K[i],i+1
            conv1 = get_conv_layer(k, fin, fout)
            conv2 = get_conv_layer(k, fin, fout)
            conv_layers_1.append(conv1)
            conv_layers_2.append(conv2)
            
            if k[0]>0:
                fin,fout = fout,fout*2
            else:
                fin,fout = fout//2,fout
            
            x = conv1(x)
            print(x.shape)
            #print('fin={},fout={}'.format(fin, fout))
        
        f = fin
        self.f = f
        x = x.view(-1, f)
        print(x.shape)
        
        merged = False
        n1 = f
        lin_layers_1, lin_layers_2, lin_layers = [],[],[]
        while i<len(K):
            n2,i = K[i],i+1
            
            if n2<0 and not merged:
                x = torch.cat((x,x),1)
                print(x.shape)
                n1 = n1*2
                n2 = abs(n2)
                merged = True
                
            if merged:
                lin = torch.nn.Linear(n1, n2)
                lin_layers.append(lin)
                x = lin(x)
            else:
                lin1 = torch.nn.Linear(n1, n2)
                lin_layers_1.append(lin1)
                lin2 = torch.nn.Linear(n1, n2)
                lin_layers_2.append(lin2)
                x = lin1(x)
            n1 = n2
            print(x.shape)
        
        if not merged:
            n1 = n1*2
            x = torch.cat((x,x),1)
            print(x.shape)
        
        lin = torch.nn.Linear(n1, output_dim)
        lin_layers.append(lin)
        x = lin(x)
        print(x.shape)
        
        self.conv_layers_1 = nn.ModuleList(conv_layers_1)
        self.conv_layers_2 = nn.ModuleList(conv_layers_2)
        self.lin_layers_1 = nn.ModuleList(lin_layers_1)
        self.lin_layers_2 = nn.ModuleList(lin_layers_2)
        self.lin_layers = nn.ModuleList(lin_layers)
        
        drop_layers = []
        if not isinstance(cfg.dropout, (list, tuple)):
            drop_layers.append(nn.Dropout(p=cfg.dropout))
        else:
            for drop in cfg.dropout:
                drop_layers.append(nn.Dropout(p=drop))
        self.drop_layers = nn.ModuleList(drop_layers)

    def next_drop(self, i=0):
        drop = self.drop_layers[i]
        i+=1
        if i==len(self.drop_layers):
            i=0
        return i,drop
    
    def next_bnorm(self, i=0):
        drop = self.drop_layers[i]
        i+=1
        if i==len(self.drop_layers):
            i=0
        return i,drop
        
    def forward(self, x):
        if self.cfg.gpu_preproc:
            x = preproc(x, self.cfg)
            
        x1 = x[:, :self.c, :]
        x2 = x[:, self.c:, :]
        
        j=0
        for conv in self.conv_layers_1:
            #x1 = F.relu(conv(x1))
            x1 = apply_conv_layer(conv, x1)
            j,drop = self.next_drop(j)
            x1 = drop(x1)
        
        j=0
        for conv in self.conv_layers_2:
            #x2 = F.relu(conv(x2))
            x2 = apply_conv_layer(conv, x2)
            j,drop = self.next_drop(j)
            x2 = drop(x2)
        
        x1 = x1.view(-1, self.f)
        x2 = x2.view(-1, self.f)
        
        k = j
        for i,lin in enumerate(self.lin_layers_1):
            x1 = F.relu(lin(x1))
            j,drop = self.next_drop(j)
            x1 = drop(x1)
        
        j = k
        for i,lin in enumerate(self.lin_layers_2):
            x2 = F.relu(lin(x2))
            j,drop = self.next_drop(j)
            x2 = drop(x2)
            
        x = torch.cat((x1,x2),1)
        
        for i,lin in enumerate(self.lin_layers):
            x = lin(x)
            if i+1==len(self.lin_layers): break
            x = F.relu(x)
            j,drop = self.next_drop(j)
            x = drop(x)
    
        return(x)

def add_magnitude(x):
    G,A = x[:,:3,:], x[:,3:,:]
    a = torch.norm(A, dim=1, keepdim=True)
    g = torch.norm(G, dim=1, keepdim=True)
    return torch.cat((G, g, A, a), 1)


def apply_stft(spec, x):
    X = []
    for i in range(x.shape[1]):
        X.append(spec(x[:,i,:])[:,None,:,:])
    return torch.cat(X, 1)

class SpecMod1(torch.nn.Module):
    
    def __init__(self, cfg, x, output_dim=1):
        super(SpecMod1, self).__init__()
        self.cfg = cfg
        
        ## add magnitude?
        if cfg.add_mag: x = add_magnitude(x)
        print(x.shape)
        
        n_fft = 32 ## 
        hop_length = n_fft//4
        out_channels = 64
        kernels = [3,3,3,3]
        strides = [2,2,2,2]
        lins = [32]
        
        trainable = False
        spec_layer_1 = Spectrogram.STFT(n_fft=n_fft, hop_length=hop_length, freq_scale='no', device='cpu', trainable=trainable)
        spec_layer_2 = Spectrogram.STFT(n_fft=n_fft, hop_length=hop_length, freq_scale='no', device='cpu', trainable=trainable)
        self.spec_layers = nn.ModuleList([spec_layer_1, spec_layer_2])
        x1 = apply_stft(spec_layer_1, x[:, :x.shape[1]//2, :])
        x2 = apply_stft(spec_layer_2, x[:, x.shape[1]//2:, :])
        print(x1.shape)
        x = torch.cat((x1,x2),1)
        print(x.shape)
        
        self.drop = torch.nn.Dropout(p=cfg.dropout)
#         self.drop = torch.nn.Dropout(p=0.2)
        
        bs, in_channels, h, w = x.shape
        i = 0
        conv_layers, lin_layers = [],[]
        
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernels[i], stride=strides[i], padding=1)
        conv_layers.append(conv)
        x = conv(x)
        print(x.shape)
        i, in_channels, out_channels = i+1, out_channels, out_channels*2
        
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernels[i], stride=strides[i], padding=1)
        conv_layers.append(conv)
        x = conv(x)
        print(x.shape)
        i, in_channels, out_channels = i+1, out_channels, out_channels*2
        
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernels[i], stride=strides[i], padding=1)
        conv_layers.append(conv)
        x = conv(x)
        print(x.shape)
        i, in_channels, out_channels = i+1, out_channels, out_channels*2
        
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernels[i], stride=strides[i], padding=0)
        conv_layers.append(conv)
        x = conv(x)
        print(x.shape)
        
        self.conv_layers = nn.ModuleList(conv_layers)
        
        x = x.view(bs, -1)
        self.n = x.shape[1]
        print(x.shape)
        
        n1 = self.n
        
        for n2 in lins:
            lin_layer = nn.Linear(in_features=n1, out_features=n2)
            lin_layers.append(lin_layer)
            x = lin_layer(x)
            print(x.shape)
            n1 = n2
        
        ## final lin layer
        lin_layer = nn.Linear(in_features=n1, out_features=output_dim)
        lin_layers.append(lin_layer)
        x = lin_layer(x)
        print(x.shape)
        
        self.lin_layers = nn.ModuleList(lin_layers)
        
    def forward(self, x):
        
        ## add magnitude?
        if self.cfg.add_mag: x = add_magnitude(x)
        x1 = apply_stft(self.spec_layers[0], x[:, :x.shape[1]//2, :])
        x2 = apply_stft(self.spec_layers[1], x[:, x.shape[1]//2:, :])
        x = torch.cat((x1,x2),1)
        
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)
            x = self.drop(x)
            
        x = x.view(-1, self.n)
        for i,lin in enumerate(self.lin_layers):
            x = lin(x)
            if i+1==len(self.lin_layers): break
            x = F.relu(x)
            x = self.drop(x)
        
        return x
    
class SpecMod2(torch.nn.Module):
    
    def __init__(self, cfg, x, output_dim=1):
        super(SpecMod2, self).__init__()
        self.cfg = cfg
        self.drop = torch.nn.Dropout(p=cfg.dropout)
        
        # cnn = [(128,64), (32,8), (3,2), (3,2), (3,2), (3,2), 32]
        K = cfg.cnn
        win, out_channels = K[0]
        
        ## add magnitude?
        if cfg.add_mag: x = add_magnitude(x)
        print(x.shape)
        
        bs, in_channels, _ = x.shape
        
        n_fft = K[1][0]         ## 32
        hop_length = K[1][1]    ## n_fft//4
        m = sum([isinstance(x, int) for x in K])
        if m==0:
            kernels, strides = zip(*K[2:])
            lins = []
        else:
            kernels, strides = zip(*K[2:-m])
            lins = K[-m:]
        pads = np.array(kernels)*0 +1
#         pads = np.zeros(len(kernels)) +1
        pads[-1]=0
        
        trainable = False
        spec_layer_1 = Spectrogram.STFT(n_fft=n_fft, hop_length=hop_length, freq_scale='no', device='cpu', trainable=trainable)
        spec_layer_2 = Spectrogram.STFT(n_fft=n_fft, hop_length=hop_length, freq_scale='no', device='cpu', trainable=trainable)
        self.spec_layers = nn.ModuleList([spec_layer_1, spec_layer_2])
        x1 = apply_stft(spec_layer_1, x[:, :x.shape[1]//2, :])
        x2 = apply_stft(spec_layer_2, x[:, x.shape[1]//2:, :])
        print(x1.shape)
        x = torch.cat((x1,x2),1)
        print(x.shape)
        
        i = 0
        conv_layers, lin_layers = [],[]
        
        for k,s,p in zip(kernels, strides, pads):
#             p = (0,0)
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=s, padding=p)
            conv_layers.append(conv)
            x = conv(x)
            print(x.shape)
            in_channels, out_channels = out_channels, out_channels*2
            
        x = x.view(bs, -1)
        self.n = x.shape[1]
        print(x.shape)
        
        n1 = self.n
        
        for n2 in lins:
            lin_layer = nn.Linear(in_features=n1, out_features=n2)
            lin_layers.append(lin_layer)
            x = lin_layer(x)
            print(x.shape)
            n1 = n2
        
        ## final lin layer
        lin_layer = nn.Linear(in_features=n1, out_features=output_dim)
        lin_layers.append(lin_layer)
        x = lin_layer(x)
        print(x.shape)
        
        self.conv_layers = nn.ModuleList(conv_layers)
        self.lin_layers = nn.ModuleList(lin_layers)
        
    def forward(self, x):
        
        ## add magnitude?
        if self.cfg.add_mag: x = add_magnitude(x)
        x1 = apply_stft(self.spec_layers[0], x[:, :x.shape[1]//2, :])
        x2 = apply_stft(self.spec_layers[1], x[:, x.shape[1]//2:, :])
        x = torch.cat((x1,x2),1)
        
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)
            x = self.drop(x)
            
        x = x.view(-1, self.n)
        for i,lin in enumerate(self.lin_layers):
            x = lin(x)
            if i+1==len(self.lin_layers): break
            x = F.relu(x)
            x = self.drop(x)
        
        return x
            
            
def apply_spec_layer(spec, x, i):
    dim = x.shape
    x = spec(x[:,i,:])
    ###################
#     print(dim)
#     print(x.shape)
#     sys.exit()
    ###################
    x = x.view(dim[0], 1, -1)[:,:,:dim[-1]]
    return x

def apply_spec(spec, x):
    dim = x.shape
    X = []
    for i in range(dim[1]):
        X.append(apply_spec_layer(spec, x, i))
    return torch.cat(X, 1)

''' SPLITS into 2 : Time/Freq  --OR--  G/A '''
''' merge before NEGATIVE linear layer '''      
class CNN_12b(torch.nn.Module):
    
    def __init__(self, cfg, x, output_dim=1):
        super(CNN_12b, self).__init__()
        self.cfg = cfg
        
        if cfg.gpu_preproc:
            ####################
            x = preproc(x, cfg)
            ####################
#             '''
#             print(x.shape)
#             spec_layer = Spectrogram.STFT(n_fft=24,
# #                                           freq_bins=None, 
#                                           hop_length=14, 
# #                                           window='hann', 
#                                           freq_scale='no', 
# #                                           center=True, 
# #                                           pad_mode='reflect', 
# #                                           fmin=50,
# #                                           fmax=6000, 
# #                                           sr=22050, 
# #                                           trainable=False, 
# #                                           output_format='Magnitude', 
# #                                           device='cuda:0'
#                                           )
            if self.cfg.feats_fft:
                n_fft = 24 ## 24  32  64
                hop_length = n_fft//2 + 2
                trainable = False
    
                spec_layer_1 = Spectrogram.STFT(n_fft=n_fft, hop_length=hop_length, freq_scale='no', device='cpu', trainable=trainable)
                spec_layer_2 = Spectrogram.STFT(n_fft=n_fft, hop_length=hop_length, freq_scale='no', device='cpu', trainable=trainable)
                x1 = apply_spec(spec_layer_1, x[:, :x.shape[1]//2, :])
                x2 = apply_spec(spec_layer_2, x[:, x.shape[1]//2:, :])
                f = torch.cat((x1,x2),1)
                self.spec_layers = nn.ModuleList([spec_layer_1, spec_layer_2])
        
                if cfg.feats_raw:
                    x = torch.cat((x, f), 1)
                else:
                    x = f

        self.c = c = x.shape[1]//2
        x = x[:, :c, :]
            
        print(x.shape)
        
        K = cfg.cnn
        win,f = K[0]
        fin,fout = c,f
        
        i, conv_layers_1, conv_layers_2 = 1,[],[]
        while isinstance(K[i], (list, tuple)):
            k,i = K[i],i+1
            conv1 = get_conv_layer(k, fin, fout)
            conv2 = get_conv_layer(k, fin, fout)
            conv_layers_1.append(conv1)
            conv_layers_2.append(conv2)
            
            if k[0]>0:
                fin,fout = fout,fout*2
            else:
                fin,fout = fout//2,fout
            
            x = conv1(x)
            print(x.shape)
            #print('fin={},fout={}'.format(fin, fout))
        
        f = fin
        self.f = f
        x = x.view(-1, f)
        print(x.shape)
        
        merged = False
        n1 = f
        lin_layers_1, lin_layers_2, lin_layers = [],[],[]
        while i<len(K):
            n2,i = K[i],i+1
            
            if n2<0 and not merged:
                x = torch.cat((x,x),1)
                print(x.shape)
                n1 = n1*2
                n2 = abs(n2)
                merged = True
                
            if merged:
                lin = torch.nn.Linear(n1, n2)
                lin_layers.append(lin)
                x = lin(x)
            else:
                lin1 = torch.nn.Linear(n1, n2)
                lin_layers_1.append(lin1)
                lin2 = torch.nn.Linear(n1, n2)
                lin_layers_2.append(lin2)
                x = lin1(x)
            n1 = n2
            print(x.shape)
        
        if not merged:
            n1 = n1*2
            x = torch.cat((x,x),1)
            print(x.shape)
        
        lin = torch.nn.Linear(n1, output_dim)
        lin_layers.append(lin)
        x = lin(x)
        print(x.shape)
        
        self.conv_layers_1 = nn.ModuleList(conv_layers_1)
        self.conv_layers_2 = nn.ModuleList(conv_layers_2)
        self.lin_layers_1 = nn.ModuleList(lin_layers_1)
        self.lin_layers_2 = nn.ModuleList(lin_layers_2)
        self.lin_layers = nn.ModuleList(lin_layers)
        
        drop_layers = []
        if not isinstance(cfg.dropout, (list, tuple)):
            drop_layers.append(nn.Dropout(p=cfg.dropout))
        else:
            for drop in cfg.dropout:
                drop_layers.append(nn.Dropout(p=drop))
        self.drop_layers = nn.ModuleList(drop_layers)
        
    
    def next_drop(self, i=0):
        drop = self.drop_layers[i]
        i+=1
        if i==len(self.drop_layers):
            i=0
        return i,drop
    
    def next_bnorm(self, i=0):
        drop = self.drop_layers[i]
        i+=1
        if i==len(self.drop_layers):
            i=0
        return i,drop
        
    def forward(self, x):
        if self.cfg.gpu_preproc:
            x = preproc(x, self.cfg)

            if self.cfg.feats_fft:
                x1 = x[:, :x.shape[1]//2, :]
                x2 = x[:, x.shape[1]//2:, :]
                x1 = apply_spec(self.spec_layers[0], x1)
                x2 = apply_spec(self.spec_layers[1], x2)
                f = torch.cat((x1,x2),1)
                
                if self.cfg.feats_raw:
                    x = torch.cat((x, f), 1)
                else:
                    x = f

        x1 = x[:, :self.c, :]
        x2 = x[:, self.c:, :]
        
        j=0
        for conv in self.conv_layers_1:
            #x1 = F.relu(conv(x1))
            x1 = apply_conv_layer(conv, x1)
            j,drop = self.next_drop(j)
            x1 = drop(x1)
        
        j=0
        for conv in self.conv_layers_2:
            #x2 = F.relu(conv(x2))
            x2 = apply_conv_layer(conv, x2)
            j,drop = self.next_drop(j)
            x2 = drop(x2)
        
        x1 = x1.view(-1, self.f)
        x2 = x2.view(-1, self.f)
        
        k = j
        for i,lin in enumerate(self.lin_layers_1):
            x1 = F.relu(lin(x1))
            j,drop = self.next_drop(j)
            x1 = drop(x1)
        
        j = k
        for i,lin in enumerate(self.lin_layers_2):
            x2 = F.relu(lin(x2))
            j,drop = self.next_drop(j)
            x2 = drop(x2)
            
        x = torch.cat((x1,x2),1)
        
        for i,lin in enumerate(self.lin_layers):
            x = lin(x)
            if i+1==len(self.lin_layers): break
            x = F.relu(x)
            j,drop = self.next_drop(j)
            x = drop(x)
    
        return(x)


''' SPLITS into 4 : TimeG / TimeA / FreqG / FreqA '''
''' merge before NEGATIVE linear layer '''      
class CNN_13(torch.nn.Module):
    
    def __init__(self, cfg, x, output_dim=1):
        super(CNN_13, self).__init__()
        self.cfg = cfg

        if cfg.gpu_preproc:
            x = preproc(x, cfg)
            
        self.c = c = x.shape[1]//4
        x = x[:, :c, :]
            
        print(x.shape)
        
        K = cfg.cnn
        win,f = K[0]
        fin,fout = c,f
        
        i, conv_layers_1g, conv_layers_2g, conv_layers_1a, conv_layers_2a = 1,[],[],[],[]
        
#         bn1g, bn2g, bn1a, bn2a = [],[],[],[] ## batch norm ???
        
        while isinstance(K[i], (list, tuple)):
            k,i = K[i],i+1
            conv1 = get_conv_layer(k, fin, fout)
            conv2 = get_conv_layer(k, fin, fout)
            conv_layers_1g.append(conv1)
            conv_layers_2g.append(conv2)
            
            conv1 = get_conv_layer(k, fin, fout)
            conv2 = get_conv_layer(k, fin, fout)
            conv_layers_1a.append(conv1)
            conv_layers_2a.append(conv2)
            
            ## batch norm ??? ############################
#             bn1g.append(nn.BatchNorm1d(num_features=fout))
#             bn2g.append(nn.BatchNorm1d(num_features=fout))
#             bn1a.append(nn.BatchNorm1d(num_features=fout))
#             bn2a.append(nn.BatchNorm1d(num_features=fout))
            ##############################################
            
            if k[0]>0:
                fin,fout = fout,fout*2
            else:
                fin,fout = fout//2,fout
            
            x = conv1(x)
            print(x.shape)
            #print('fin={},fout={}'.format(fin, fout))
        
        f = fin
        self.f = f
        x = x.view(-1, f)
        print(x.shape)
        
        merged = False
        n1 = f
        lin_layers_1g, lin_layers_2g, lin_layers_1a, lin_layers_2a, lin_layers = [],[],[],[],[]
        while i<len(K):
            n2,i = K[i],i+1
            
            if n2<0 and not merged:
                x = torch.cat((x,x,x,x),1)
                n1 = n1*4
                n2 = abs(n2)
                merged = True
                
            if merged:
                lin = torch.nn.Linear(n1, n2)
                lin_layers.append(lin)
                x = lin(x)
            else:
                lin1 = torch.nn.Linear(n1, n2)
                lin_layers_1g.append(lin1)
                lin2 = torch.nn.Linear(n1, n2)
                lin_layers_2g.append(lin2)
                lin1 = torch.nn.Linear(n1, n2)
                lin_layers_1a.append(lin1)
                lin2 = torch.nn.Linear(n1, n2)
                lin_layers_2a.append(lin2)
                x = lin1(x)
            n1 = n2
            print(x.shape)
        
        if not merged:
            n1 = n1*4
            x = torch.cat((x,x,x,x),1)
            print(x.shape)
        
        lin = torch.nn.Linear(n1, output_dim)
        lin_layers.append(lin)
        x = lin(x)
        print(x.shape)
        
        self.conv_layers_1g = nn.ModuleList(conv_layers_1g)
        self.conv_layers_2g = nn.ModuleList(conv_layers_2g)
        self.conv_layers_1a = nn.ModuleList(conv_layers_1a)
        self.conv_layers_2a = nn.ModuleList(conv_layers_2a)
        
        ## batch norm ??? #############
#         self.bn1g = nn.ModuleList(bn1g)
#         self.bn2g = nn.ModuleList(bn2g)
#         self.bn1a = nn.ModuleList(bn1a)
#         self.bn2a = nn.ModuleList(bn2a)
        ###############################
        
        self.lin_layers_1g = nn.ModuleList(lin_layers_1g)
        self.lin_layers_2g = nn.ModuleList(lin_layers_2g)
        self.lin_layers_1a = nn.ModuleList(lin_layers_1a)
        self.lin_layers_2a = nn.ModuleList(lin_layers_2a)
        
        self.lin_layers = nn.ModuleList(lin_layers)
        
        drop_layers = []
        if not isinstance(cfg.dropout, (list, tuple)):
            drop_layers.append(nn.Dropout(p=cfg.dropout))
        else:
            for drop in cfg.dropout:
                drop_layers.append(nn.Dropout(p=drop))
        self.drop_layers = nn.ModuleList(drop_layers)
    
    def next_drop(self, i=0):
        drop = self.drop_layers[i]
        i+=1
        if i==len(self.drop_layers):
            i=0
        return i,drop
        
    def forward(self, x):
        if self.cfg.gpu_preproc:
            x = preproc(x, self.cfg)
        x1g = x[:, :self.c, :]
        x1a = x[:, self.c:2*self.c, :]
        x2g = x[:, 2*self.c:3*self.c, :]
        x2a = x[:, 3*self.c:, :]
        
        j=0
        for i, conv in enumerate(self.conv_layers_1g):
            x1g = apply_conv_layer(conv, x1g)#, self.bn1g[i])
            j,drop = self.next_drop(j)
            x1g = drop(x1g)
        j=0
        for i, conv in enumerate(self.conv_layers_1a):
            x1a = apply_conv_layer(conv, x1a)#, self.bn1a[i])
            j,drop = self.next_drop(j)
            x1a = drop(x1a)
        j=0
        for i, conv in enumerate(self.conv_layers_2g):
            x2g = apply_conv_layer(conv, x2g)#, self.bn2g[i])
            j,drop = self.next_drop(j)
            x2g = drop(x2g)
        j=0
        for i, conv in enumerate(self.conv_layers_2a):
            x2a = apply_conv_layer(conv, x2a)#, self.bn2a[i])
            j,drop = self.next_drop(j)
            x2a = drop(x2a)
        
        x1g = x1g.view(-1, self.f)
        x1a = x1a.view(-1, self.f)
        x2g = x2g.view(-1, self.f)
        x2a = x2a.view(-1, self.f)
        
        k = j
        for i,lin in enumerate(self.lin_layers_1g):
            x1g = F.relu(lin(x1g))
            j,drop = self.next_drop(j)
            x1g = drop(x1g)
        j = k
        for i,lin in enumerate(self.lin_layers_2g):
            x2g = F.relu(lin(x2g))
            j,drop = self.next_drop(j)
            x2g = drop(x2g)
        j = k
        for i,lin in enumerate(self.lin_layers_1a):
            x1a = F.relu(lin(x1a))
            j,drop = self.next_drop(j)
            x1a = drop(x1a)
        j = k
        for i,lin in enumerate(self.lin_layers_2a):
            x2a = F.relu(lin(x2a))
            j,drop = self.next_drop(j)
            x2a = drop(x2a)
            
        x = torch.cat((x1g,x1a,x2g,x2a),1)
        
        for i,lin in enumerate(self.lin_layers):
            x = lin(x)
            if i+1==len(self.lin_layers): break
            x = F.relu(x)
            j,drop = self.next_drop(j)
            x = drop(x)
    
        return(x)

####################################################################################

def init_tensor(t, *size):
    if t==1:
        return torch.randn(*size).float()
    x = torch.zeros(*size).float()
    if t==0:
        return x
    elif t==2:
        torch.nn.init.xavier_normal_(x)
    elif t==3:
        torch.nn.init.xavier_uniform_(x)
    return x
       
class RNN_1(torch.nn.Module):
    
    def __init__(self, cfg, x):
        super(RNN_1, self).__init__()
        
        K = cfg.rnn
        s = K[0]
        self.h = h = K[1]
        self.c = c = cfg.n_inputs
        self.num_layers = 1
        self.batch_size = cfg.batch_size
        
        print(x.shape)
        
#        self.rnn  = nn.GRU(c, h, self.num_layers, dropout=cfg.dropout, batch_first=True)
        self.rnn  = nn.LSTM(c, h, self.num_layers, dropout=cfg.dropout, batch_first=True)
        
        lin_layers = []
        n1,i = h,2
        while i<len(K):
            n2 = K[i]
            lin = nn.Linear(n1, n2)
            lin_layers.append(lin)
            n1 = n2
        lin_layers.append(nn.Linear(n1, 1))
        self.lin_layers = nn.ModuleList(lin_layers)
        self.drop = nn.Dropout(cfg.dropout)
        
    def init_tensor(self, batch_size):
        return init_tensor(1, self.num_layers, batch_size, self.h)
    
    def init_hidden(self, batch_size):
        ## GRU
#        return self.init_tensor(batch_size)
        ## LSTM
        return (self.init_tensor(batch_size), self.init_tensor(batch_size))
        
    def forward(self, x, hidden=None):
        H = hidden
        if H is None:
            hidden = self.init_hidden(x.shape[0])
        x, hidden = self.rnn(x, hidden)
        
        ## get last output
        x = x.squeeze()[:, -1]
        
        for i,lin in enumerate(self.lin_layers):
            x = lin(x)
            if i+1==len(self.lin_layers): break
            x = self.drop(F.relu(x))
            
        if H is None:
            return x
        return x, hidden
        
    