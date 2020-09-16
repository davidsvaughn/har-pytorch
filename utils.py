from __future__ import print_function, division

import sys
import csv
import os, errno, re
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os,sys,errno
import random
import collections
import shutil

from scipy import signal
import matplotlib.pyplot as plt
from statsmodels import robust

#import tensorflow as tf
#from keras.backend.tensorflow_backend import dtype

class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self
        
def ps(x, s):
    #print('{} : {}'.format(s, x.shape))
    print('{} : {}'.format(s, tf.shape(x)))
    #print('{} : {}'.format(s, x.get_shape()))

def mkdirs(s):
    try:
        os.makedirs(s)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(s):
            pass

def chkfile(fn):
    path,_ = os.path.split(fn)
    if not os.path.isdir(path):
        mkdirs(path)
        
def purge(path, pattern=None):
    for f in os.listdir(path):
        if pattern and (not re.search(pattern, f)):
            continue
        os.remove(os.path.join(path, f))
        
def clear_path(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            
def remove_file(file_path):
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)
        
def remove_path(dir_path):
    try:
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
    except Exception as e:
        print(e)
        
def get_seed():
    t = int( time.time() * 1000.0 )
    seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >>  8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)
    seed = seed % 2**30-1#    2**32-1
    print('GET_SEED = {}'.format(seed))
    return seed

rng = 0
def seed_random(seed=None, quiet=False):
    global rng
    if seed==None or seed<=0:
        seed = get_seed()
    if not quiet:
        print('RAND_SEED == {}\n'.format(seed))
    random.seed(seed)
    np.random.seed(seed=seed)
    rng = np.random.RandomState(seed)
#     return seed
    return rng, seed

def shuffle(x, rng):
    #random.shuffle(x, random=rng)
    p = rng.permutation(len(x))
    x=x[p]
    return x
    
def unison_sorted_lists(ls, col=0, order=1):
    return list(map(list, zip(*sorted(list(zip(*ls)), key=lambda pair: order*pair[col]))))

def sorty(X, Y, order=-1):
    return [x for _, x in sorted(zip(Y,X), key=lambda pair: order*pair[0])]

def topk_mean(x, y, k=None):
    x = sorty(x,y)
    if k is None: k=len(x)
    return np.mean(x[:k])

def topk_weighted_mean(x, y, k=None, f=None):
    x,y = unison_sorted_lists([x,y], col=1, order=-1)
    x,y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    if f is None: f=lambda x:x
    if k is None: k=x.shape[0]
#     print(list(x[:k]))
#     print(list(f(y[:k])))
    return np.average(x[:k], weights=f(y[:k]))
  
'''
def unison_shuffled_lists(*ls):
    l =list(zip(*ls))
    random.shuffle(l)
    return zip(*l)

def unison_shuffled_arrays(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
'''

CLIP_THRESHOLDS = [300., 300., 300., 2., 2., 2.]
NORM_THRESHOLDS = [2000., 2000., 2000., 2., 2., 2.]

def clip_maxmin(x, clip_thresh=None):
    if clip_thresh is None:
        clip_thresh = np.array(CLIP_THRESHOLDS, dtype=np.float32)
    return np.clip(x, -clip_thresh, clip_thresh)

def normalize_maxmin(x, hi=None, lo=None):
    if hi is None:
        hi = CLIP_THRESHOLDS
    if not isinstance(hi, collections.Iterable):
        hi = hi *np.ones(x.shape[1])
    hi = np.array(hi, dtype=np.float32)
    if lo is None:
        lo = -hi
    else:
        if not isinstance(lo, collections.Iterable):
            lo = lo *np.ones(x.shape[1])
        lo = np.array(lo, dtype=np.float32)
            
    x = np.asarray(x, dtype=np.float32)
    diffs = hi - lo
#     print(np.max(x, axis=0))
#     print(np.min(x, axis=0))
    for i in np.arange(x.shape[1]):
        x[:, i] = (x[:, i]-lo[i])/diffs[i]
    return x

def normalize_std(x, mu=None, std=None):
    ''' Normalizes all sensor channels: (x-mu)/(2*std) '''
    x = np.array(x, dtype=np.float32)    
    if mu is None: 
        mu = np.mean(x, axis=0)
        #mu = np.median(x, axis=0)
    if std is None: 
        std = np.std(x, axis=0)
        #std = robust.mad(x, axis=0)
    std += 0.000001
    return (x-mu)/(std * 2) # 2 is for having smaller values

def normalize_mad(x):
    x = np.array(x, dtype=np.float32)
    med = np.median(x, axis=0) 
    mad = robust.mad(x, axis=0)
    return (x-med)/(mad * 2) # 2 is for having smaller values

def normalize(x, all_axis=False, mode='std'):
    if mode=='maxmin':
        all_axis=False
    
    if all_axis:
        n,m = x.shape
        x = merge_axes(x)
        
    ##########
    if mode=='std':
        x = normalize_std(x)
    elif mode=='mad':
        x = normalize_mad(x)
    elif mode=='maxmin':
        x = normalize_maxmin(x)
    ##########
    
    if all_axis:
        x = split_axes(x, d=int(m/2))
        
    return x

def merge_axes(x):
    n,m = x.shape
    y,k = [],int(m/2)
    for i in range(k):
        y.append(x[:,[i,i+k]])
    y = np.vstack(y)
    return y
    
def split_axes(x, d=3):
    n = int(x.shape[0]/d)
    x0,x1,k = [],[],n
    
    for i in range(d):
        x0.append(x[k-n:k,0])
        x1.append(x[k-n:k,1])
        k+=n
        
    y = np.column_stack(x0 + x1)
    return y

def nuniq(a):
    b = np.sort(a,axis=0)
    return (b[1:] != b[:-1]).sum(axis=0)+1

def rand_proj(x, m, p=[0,0,0,0,-1,1]):
#     k,n = x.shape
#     y = np.zeros([k,m])
#     for i in range(k):
#         y[i] = np.random.choice(p, size=[m,n]).dot(x[i])
#     return y
    p = np.array([[-3.**.5/2., 3.**.5/2., 0],[-.5, -.5, 1]], dtype=np.float32).transpose()
    return x.dot(p)

def rand_proj_gauss(x, m):
    from sklearn import random_projection
    transformer = random_projection.GaussianRandomProjection(n_components=m)
    #y = transformer.transform(x)
    y = transformer.fit_transform(x)
    return y

def quantize(x, q=100, x_range=[1.0]):#g_range=4000., a_range=400.
    
    if x_range is None:
        x_range = np.max(x, axis=0) - np.min(x, axis=0)
        print(np.max(x, axis=0))
        print(np.min(x, axis=0))
        print(x_range)
    else:
        x_range = np.array(x_range, dtype=np.float32)
        n = x.shape[1]
        if n>x_range.shape[0]:
            x_range = np.ones([n])*x_range
        
    qq = x_range/q
    y = np.round(x/qq)
    
    #u = nuniq(y)
    
    return y - q/2 # center around 0

def repair_diff(x, q):
    while True:
        A1,B1 = np.where(x>q)
        A1,B1 = np.flip(A1,0),np.flip(B1,0)
        for i,a in enumerate(A1):
            b = B1[i]
            v = x[a,b] - q
            x[a,b] = q
            x[a-1,b] += v
        A2,B2 = np.where(x<-q)
        A2,B2 = np.flip(A2,0),np.flip(B2,0)
        for i,a in enumerate(A2):
            b = B2[i]
            v = x[a,b] + q
            x[a,b] = -q
            x[a-1,b] += v
        if len(A1)+len(A2)==0:
            break
        print('loop')
    return x

def cosine(a, b):
    return (a * b).sum(axis=1) / (np.linalg.norm(a,axis=1) * np.linalg.norm(b,axis=1))

def angle(a, b):
    return np.arccos(cosine(a, b))

def package_diffs(x):
    #x = np.array(x, dtype=np.float32)
    x = np.expand_dims(x,-1)# make Nx1 matrix
    x = np.vstack([x[0], x])# fix off-by-one
    return x
    
def diff_angles(x):
    y = cosine(x[1:], x[:-1])
    return package_diffs(y)

def diff_norms(x):
    y = np.linalg.norm(x[1:]-x[:-1], axis=1)
    return package_diffs(y)

def _one_hot(i, n):
    a = np.zeros(n, 'uint8')
    a[i] = 1
    return a

def one_hot(y_, n_values=None):
    """
    Function to encode output labels from number indexes.
    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    if isinstance(y_, (int, float)):
        return _one_hot(y_-1, n_values)
    y_ = np.asarray(y_, dtype=np.int32) - 1
    y_ = y_.reshape(len(y_))
    if n_values is None:
        n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def reverse_one_hot(y_):
    """
    Function to reverse one-hot coding
    E.g.: [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]] --> [[5], [0], [3]] 
    """
    flat_y = []
    for index, entry in enumerate(y_, start=0):
    #   print("---  y_ [", index, "]:",  entry)
        for bit, code in enumerate(entry):
            if(code > 0):
                flat_y.append(bit)

    return flat_y

def read_cols(fn, cols=None, sep="\t", header=None, type='float32'):
    df = pd.read_csv(fn, sep=sep, header=header)#.sort_values(by=col)
    if cols is None:
        cols = range(len(df.columns))
    vals = df[df.columns[cols]].values.astype(type)
    return vals

def output_data(out_file, X, y):
    chkfile(out_file)
    with open(out_file, 'w') as f:
        for i, row in enumerate(X):
            s = '\t'.join(['{0:0.8g}'.format(float(x)) for x in row])
            f.write('{0:0}\t{1}\n'.format(int(y[i]), s))
            f.flush()
            
def specgram(x, fs=5.0, 
             nperseg=16, 
             step=8,
             nfft=None,
             scaling='density',
             ):
    f, t, Sxx = signal.spectrogram(x, fs=fs,
                                   nperseg=nperseg, 
                                   noverlap=nperseg-step,
                                   nfft=nfft,
                                   scaling=scaling,
                                   )
    return Sxx, f, t



def string2tf(s):
    return tf.py_func(lambda: s, [], tf.string)

def log_shape(LOG, x, name):
    if LOG is not None:
        #LOG.append((string2tf(name), x))
        LOG.append((string2tf(name), tf.shape(x)))
        
def reject_outliers(x, m=2.):
    d = np.abs(x - np.median(x))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return x[s<m]

def meany(x, m=2.):
    return np.mean(reject_outliers(x, m=m))

def center_columns(X, m=2.):
    y = []
    for i in range(X.shape[1]):
        y.append( X[:,i] - meany(X[:,i], m=m) )
    return np.column_stack(y)

def center_column(X, m=2.):
    mu = [meany(X[:,i], m=m) for i in range(X.shape[1])]
    i = np.argmax(np.abs(mu))
    X[:,i] -= mu[i]
    return X
    
def eagerex():
    try:
        tf.enable_eager_execution()
        print('Eager Execution enabled!')
    except ValueError:
        if tf.executing_eagerly():
            print('Eager Execution already enabled!')
        else:
            print('Problem enabling Eager Execution!')
#############################################################

class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WHITE = '\033[37m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    RED = '\033[31m'
    MAGENTA = '\033[35m'
    BLACK = '\033[30m'
    BHEADER = BOLD + '\033[95m'
    BOKBLUE = BOLD + '\033[94m'
    BOKGREEN = BOLD + '\033[92m'
    BWARNING = BOLD + '\033[93m'
    BFAIL = BOLD + '\033[91m'
    BUNDERLINE = BOLD + '\033[4m'
    BWHITE = BOLD + '\033[37m'
    BYELLOW = BOLD + '\033[33m'
    BGREEN = BOLD + '\033[32m'
    BBLUE = BOLD + '\033[34m'
    BCYAN = BOLD + '\033[36m'
    BRED = BOLD + '\033[31m'
    BMAGENTA = BOLD + '\033[35m'
    BBLACK = BOLD + '\033[30m'
    
    @staticmethod
    def cleared(s):
        return re.sub("\033\[[0-9][0-9]?m", "", s)

def red(message):
    return BColors.RED + str(message) + BColors.ENDC

def b_red(message):
    return BColors.BRED + str(message) + BColors.ENDC

def b_blue(message):
    return BColors.BLUE + str(message) + BColors.ENDC

def b_yellow(message):
    return BColors.BYELLOW + str(message) + BColors.ENDC

def green(message):
    return BColors.GREEN + str(message) + BColors.ENDC

def b_green(message):
    return BColors.BGREEN + str(message) + BColors.ENDC

#=============================================================
# mu-Law

# https://www.dsprelated.com/showcode/125.php

# def mulaw(x, mu=255.):
#     # Non-linear Quantization
#     # mulaw: mulaw nonlinear quantization
#     # x = input vector
#     # Cx = mulaw compressor output
#     # Xmax = maximum of input vector x
#     x = np.array(x, dtype=np.float32)
#     Xmax  = max(abs(x))
#     x = x/Xmax
#     if np.log(1+mu) != 0:
#         Cx = (np.log(1+mu * abs(x)) / np.log(1+mu));
#     else:
#         Cx = x;
#     return Cx#, Xmax
# 
# def invmulaw(y, mu=255.):
#     # Non-linear Quantization
#     # invmulaw: inverse mulaw nonlinear quantization
#     # x = output vector 
#     # y = input vector (using mulaw nonlinear comression)
#     y = ((1+mu)**(abs(y))-1)/mu
#     return y  
  
def mu(x, b=8., u=255.):
    x = np.array(x, dtype=np.float32)
    x = x/(2**(b-1))
    y = np.sign(x) * np.log(1 + u*abs(x)) / np.log(1+u);
    return y

def invmu(y, b=8., u=255.):
    y = np.sign(y)*((1+u)**(abs(y))-1)/u
    y = np.ceil(y*(2**(b-1)))
    return y

def bitscale(x, b):
    bb = 2.**(b-1)
    return np.floor(bb * x)/bb

#########################################################

def read_lines(fn):
    with open(fn) as f:
        lines = [line.strip() for line in f]
    return lines

def write_lines(fn, lines):
    with open(fn, 'w') as f:
        for line in lines:
            f.write(line + '\n')

if __name__ == "__main__":
    WIN=24
    STEP=12
    BLIP=5
    
    file_path   = '/home/david/code/python/revibe-ml/data/revibe/FidgetSampleArray.txt'
    data = read_lines(file_path)[0]
    data = np.array(list(map(int, data.split(','))))
    data = data.reshape([-1,7])
    t = data[:,0]
    dt = np.diff(t)
    idx = np.where(dt>BLIP)[0]+1
    data_sets = np.split(data, idx)
    
    windows=[]
    for ds in data_sets:
        n = len(ds)
        if n<WIN: continue
        idx = range(n+1)
        idx = np.array(idx[WIN:n+1:STEP])
        if idx[-1]!=n: idx=np.append(idx,n)
        w = [ds[i-WIN:i] for i in idx]
        windows.extend(w)
    