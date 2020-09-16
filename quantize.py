import numpy as np
import pandas as pd
import os,inspect

def norm(x, c, o=0):
    return np.clip(x-o, -c, c)/c

def denorm(x, c, o=0):
    return x*c + o

## assumes x is normalized to [-1..1]
def mulaw(x, u):
    return np.sign(x) * np.log(1 + u * np.minimum(1, abs(x))) / np.log(1+u)

def invmu(x, u):
    return np.sign(x) * ((1+u)**abs(x)-1) / u

def quant(x, v):
#    xq = np.ceil(x * v).astype(np.int32)
    xq = np.floor(x * v).astype(np.int32)
    return xq

def compress(x, u, v):
    q = mulaw(x, u)     ## warp...
    return quant(q, v)  ## quantize...

def decompress(x, u, v, v2=None):
    v2 = v if v2 is None else v2
    
#    xq = (x-v)/v2           ## un-quantize... (original)
    ## -- OR --
    xq = 2*x/(v+v2-1) -1    ## like 8-bit (quantize.py)
    
    return invmu(xq, u) ## un-warp...

'''
Parameters for 5-bit quantization...
'''
O = np.array([0, 0, 0, 20, 0, 40])              ## offsets
C = np.array([250, 250, 250, 150, 150, 150])    ## bounds
U = np.array([12, 8, 12, 6, 4, 6])              ## mu-law parameters

''' 5-bit quantize function '''
def quantize(X, b=5, u=U, c=C, o=O):
    v = 2**(b-1)
    Xn = norm(X, c, o) ## normalize to [-1..1]
    Xq = np.clip(compress(Xn, u=u, v=v) + v, 0, 2*v-1)
    
    ''' Delimiter Value...
        only affects positive values of g_x (first gyro)... '''
    idx = X[:,0]>0
    xn = Xn[idx]
    v2 = v-1
    xq = np.clip(compress(xn, u=u, v=v2) + v, 0, v+v2-1)
    Xq[idx,0] = xq[:,0]
    
    ## return
    return Xq

''' 5-bit dequantize function '''
def dequantize(x, b=5, u=U, c=C, o=O, norm=True):
    v = 2**(b-1)
    
    ## delim...
    xc = x-v
    idx = xc[:,0]>0
    xd = x[idx]
    v2 = v-1
    
    ##
    x1 = decompress(x, u, v)
    x1d = decompress(xd, u, v, v2)
    x1[idx,0] = x1d[:,0]
    
    return x1 if norm else denorm(x1, c, o)

#----------------------------------------------------------  
'''
Make 8bit-5bit conversion table
'''
def save_tsv(X, fn):
    with open(fn, 'w') as f:
        for x in X:
            f.write('{}\n'.format('\t'.join(x.astype(np.str))))
             
def make_conversion_table():
    X = np.repeat(np.arange(256)[:,None],6,1)
    X2 = q1.inverse_mulaw_quantize(X)
    X3 = quantize(X2)
    fn = 'convert_8bit_to_5bit.tsv'
    save_tsv(X3, fn)
    
class BitConverter(object):
    def __init__(self, fn='convert_8bit_to_5bit.tsv', path=None):
        if path is None:
            path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        fn = os.path.join(path, fn)
        ## load conversion table
        df = pd.read_csv(fn, sep='\t', header=None)
        T = df.values
        self.T = np.vstack([T, np.ones([1,6])*np.nan])
        
    def convert(self, X):
        if X is None or len(X)==0:
            return X
#         if np.any([x is None for x in X]):
#             return X
        X[np.isnan(X)] = self.T.shape[0]-1
        X = X.astype(np.int32)
        return np.column_stack([self.T[X[:,i],i] for i in range(6)])

BC = None

def dequant(X, quant=5, norm=True):
    global BC
    if quant==8:
        if BC is None:
            BC = BitConverter()
        X = BC.convert(X)
    return dequantize(X, norm=norm)

