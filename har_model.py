import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

JSON_CONTENT_TYPE = 'application/json'
NPY_CONTENT_TYPE = 'application/x-npy'

CNN_MODEL_PARAMS = [(32,32), (4,2), (4,2), (4,1), (3,1), 64, 32]
GPU_PREPROC = True
N_LABELS = 4

def preproc(x):
    G,A = x[:,:3,:], x[:,3:,:]
    
    ## add magnitude of accel and gyro...
    a = torch.norm(A, dim=1, keepdim=True)
    g = torch.norm(G, dim=1, keepdim=True)
    x1 = torch.cat((G, g, A, a), 1)
    
    ## add FFT features...
    x2 = torch.rfft(x1, 1, onesided=False)
    x2 = torch.norm(x2, dim=-1)
    
    ## return time and freq features separately...
    return torch.cat((x1, x2), 1)

def get_conv_layer(k, fin, fout):
    return nn.Conv1d(in_channels=fin, out_channels=fout, kernel_size=k[0], stride=k[1]).float()
    
class cnn_har_model(nn.Module):
    
    def __init__(self, K, gpu_preproc=True):
        super(cnn_har_model, self).__init__()
        
        self.gpu = gpu_preproc
        self.c = c = 4# x.shape[1]//4
        
        #K = cfg.cnn
        win,f = K[0]
        fin,fout = c,f
        
        i, conv_layers_1g, conv_layers_2g, conv_layers_1a, conv_layers_2a = 1,[],[],[],[]
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
            fin,fout = fout,fout*2
        
        f = fin
        self.f = f

        n1 = f
        lin_layers_1g, lin_layers_2g, lin_layers_1a, lin_layers_2a, lin_layers = [],[],[],[],[]
        while i<len(K):
            n2,i = K[i],i+1
            lin1 = torch.nn.Linear(n1, n2)
            lin2 = torch.nn.Linear(n1, n2)
            lin_layers_1g.append(lin1)
            lin_layers_2g.append(lin2)
            lin1 = torch.nn.Linear(n1, n2)
            lin2 = torch.nn.Linear(n1, n2)
            lin_layers_1a.append(lin1)
            lin_layers_2a.append(lin2)
            n1 = n2
        
        n1 = n1*4
        lin = torch.nn.Linear(n1, N_LABELS)
        lin_layers.append(lin)
        
        self.conv_layers_1g = nn.ModuleList(conv_layers_1g)
        self.conv_layers_2g = nn.ModuleList(conv_layers_2g)
        self.conv_layers_1a = nn.ModuleList(conv_layers_1a)
        self.conv_layers_2a = nn.ModuleList(conv_layers_2a)
        
        self.lin_layers_1g = nn.ModuleList(lin_layers_1g)
        self.lin_layers_2g = nn.ModuleList(lin_layers_2g)
        self.lin_layers_1a = nn.ModuleList(lin_layers_1a)
        self.lin_layers_2a = nn.ModuleList(lin_layers_2a)
        
        self.lin_layers = nn.ModuleList(lin_layers)
        
    def forward(self, x):
        if self.gpu:
            x = preproc(x)
        
        x1g = x[:, :self.c, :]
        x1a = x[:, self.c:2*self.c, :]
        x2g = x[:, 2*self.c:3*self.c, :]
        x2a = x[:, 3*self.c:, :]
        
        for conv in self.conv_layers_1g:
            x1g = F.relu(conv(x1g))
        for conv in self.conv_layers_1a:
            x1a = F.relu(conv(x1a))
        for conv in self.conv_layers_2g:
            x2g = F.relu(conv(x2g))
        for conv in self.conv_layers_2a:
            x2a = F.relu(conv(x2a))
        
        x1g = x1g.view(-1, self.f)
        x1a = x1a.view(-1, self.f)
        x2g = x2g.view(-1, self.f)
        x2a = x2a.view(-1, self.f)
        
        for lin in self.lin_layers_1g:
            x1g = F.relu(lin(x1g))
        for lin in self.lin_layers_1a:
            x1a = F.relu(lin(x1a))
        for lin in self.lin_layers_2g:
            x2g = F.relu(lin(x2g))
        for lin in self.lin_layers_2a:
            x2a = F.relu(lin(x2a))
            
        x = torch.cat((x1g,x1a,x2g,x2a),1)
        
        for i,lin in enumerate(self.lin_layers):
            x = lin(x)
            if i+1==len(self.lin_layers): break
            x = F.relu(x)
    
        return(x)

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def load_model(model_dir, model_file, gpu_preproc=None):
    if gpu_preproc is None:
        gpu_preproc = GPU_PREPROC
    device = get_device()
    model = cnn_har_model(CNN_MODEL_PARAMS, gpu_preproc)
    with open(os.path.join(model_dir, model_file), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def model_fn(model_dir):
    return load_model(model_dir, model_file='model.pth')

def predict_fn(input_data, model):
    device = get_device()
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(input_data).float().to(device))
#    output = torch.softmax(output, 1)
    output = output.cpu().data.numpy().astype(np.float32)
#    output = output.round(4)
    return output

################################################################
'''
The methods input_fn and output_fn are optional and if omitted, 
SageMaker will assume the input and output objects are of type 
NPY format with Content-Type application/x-npy
(https://course.fast.ai/deployment_amzn_sagemaker.html)
'''

def input_fn(request_body, content_type=JSON_CONTENT_TYPE):
    data = np.array(eval(request_body))
    return data
#     
# def output_fn(prediction, content_type=JSON_CONTENT_TYPE):
#     p = prediction['img']
#     original_size = prediction['size']
#     denorm = DeProcess(imagenet_stats, size, padding, original_size)
#     pred = denorm(p)
#     if content_type == JSON_CONTENT_TYPE: 
#         return json.dumps({'prediction': image_to_base64(pred).decode()})
