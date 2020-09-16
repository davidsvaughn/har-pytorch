import torch
from torch import nn

def add_magnitudes(x):
    G,A = x[:,:3,:], x[:,3:,:]
    
    ## add magnitude of accel and gyro...
    a = torch.norm(A, dim=1, keepdim=True)
    g = torch.norm(G, dim=1, keepdim=True)
#    x = torch.cat((G, g, A, a), 1)
    
    return torch.cat((G, g, A, a), 1)