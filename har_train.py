import os, sys
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import stats 
import random, time
import matplotlib.pyplot as plt
import seaborn as sb
sb.set(color_codes=True, style="ticks")

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from apex import amp

# import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

import models_pytorch as mods
import data_feeder as df
from data_feeder import adict
from cross_entropy import CrossEntropyLoss

import models_timeseries as mts

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('USING -----> {}'.format(device))


##################################################################################
'''
python -u har_pytorch.py | tee log.txt
'''
##################################################################################

class Config(object):
    
    def get_seed(self):
        t = int( time.time() * 1000.0 )
        seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >>  8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)
        seed = seed % 2**30-1#    2**32-1
        return seed

    def seed_random(self, seed=None, quiet=False):
        if seed==None or seed<=0:
            seed = self.get_seed()
        if not quiet:
            print('RAND_SEED == {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed=seed)
        self.rng = np.random.RandomState(seed)
        return seed

    ## DOES NOT change array
    def shuffle(self, *args):
        p = self.rng.permutation(args[0].shape[0])
        return [x[p] for x in args] if len(args)>1 else args[0][p]
    
    ## CHANGES LIST
    def shuffle_list(self, L):
        self.rng.shuffle(L)
        
    def feature_dump(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))
        
    def __init__(self, X_train=None, X_test=None):
       
        self.model_path = None
        self.blacklist = []
        self.seed = None
        self.nz = 1
        self.device = device
        
        ###########################################################################
        ''' HAR Settings '''
        '''
        self.mod    = 'CNN_13' ## SPLITS INTO 4 !!! (HAR)
        self.cnn    = [(32,32), (4,2), (4,2), (4,1), (3,1), 64, 32] ## HAR
        
        self.data_path = '/home/david/data/revibe/boris/npy/har1'
        self.label_names = [' None', 'Fidget', ' Walk', '  Run']
#         self.model_path = 'har_model.pth'
#         self.seed = 351056104
#         self.seed = 506865387
        '''
        
        ###########################################################################

        ''' Sit/Stand Settings '''
        self.mod    = 'CNN_12' ## SPLITS INTO 2 !!! (SIT)
#         self.mod    = 'CNN_12b'
        self.cnn    = [(128,128), (8,4), (8,3), (6,3), 64, 32] ## SIT
        
        self.mod    = 'SPEC2' #  SPEC1  SPEC2
        self.cnn    = [(128,64), (32,8), (3,2), (3,2), (3,2), (3,2), 32]
        self.cnn    = [(128,64), (32,8), (3,2), (3,2), (3,2), (3,2), 64, 32]
#         self.cnn    = [(128,64), (32,8), (3,2), (3,2), (3,2), (3,2), 128, 64, 32]

#         self.cnn    = [(128,64), (32,8), ((4,1),(2,1)), ((1,4),(1,2)), (3,2), (3,2), 64, 32]
#         self.cnn    = [(128,64), (32,8), ((4,2),(2,1)), ((2,4),(1,2)), (4,2), (4,2), 64, 32]
        
#         self.cnn    = [(128,128), (24,10), (4,2), (3,2), (3,2), 64, 32]

        self.data_path = '/home/david/data/revibe/boris/npy/sit1' ## 256/0.95
#         self.data_path = '/home/david/data/revibe/boris/npy/sit2' ## 256/0.95
        
        self.nz = 0.75 ## 0.75 ## skip blocks having over 75% whitespace
        self.label_names = ['Sitting', 'Standing']
#         self.model_path = 'sit_model.pth'
#         self.seed = 926723882
#         self.seed = 544541619

        self.seed = 706354485

        ###########################################################################
        
        self.labels = [''] + ['.{}.'.format(s.strip().lower()) for s in self.label_names[1:]]
        self.label_fxn = lambda s: np.nonzero([int(l in s) for l in self.labels])[0][-1]
        
#         self.blacklist = ['.run.']
        self.blacklist_fxn = lambda s: int(any(t in s for t in self.blacklist))
        
        self.training_epochs = 300
        
        self.batch_size = 200 # 200
        self.dropout = 0.5  ## 0.4

        self.opt = 'adam'       ##  adam rms
        self.learning_rate = 0.001
        
        self.eval1 = 'acc'  ## acc pr roc
        self.eval2 = 'acc'  ## acc pr loss
        
        self.train_step = 0.33
        self.test_step = 0.33
        
#         self.train_step = 0.25
#         self.test_step = 0.25
        
        ''' training for research... '''
        self.valid_split = 0.2  ## 0.15
        self.test_split = -1    ## -1 = average ALL folds
        self.kfold = 5          ## <2 = NO kfold validation
        
        ''' training for production... '''
        self.valid_split = 0.1
        self.test_split = 0.05
        
        self.test_id = None
#        self.test_id = 'proj3.' # 1 3 6
        
        ## features...
        self.gpu_preproc = True
        self.feats_raw = True
        self.feats_fft = True
        self.add_mag = True

        ## augmentation
        self.aug_factor = 0.25 # 0.25 ## 1 3 0.5 
        self.rot = True
        self.rot_factor = 0.25
        self.time_warp = True
        self.knot_freq = 25
        self.sigma = 0.2
        self.sigma_cut = 25
        self.aug_valid = False
        
        ## top-K
        self.k = 20 # 10
        self.f = 10 # 10
        ## checkpointer (#keep)
        #self.n = 3 # 10

        '''########################################################################'''
        ## har...
#         self.mod    = 'CNN_13' ## SPLITS INTO 4 !!! (HAR)
#         self.cnn    = [(32,32), (4,2), (4,2), (4,1), (3,1), 64, 32] ## HAR    
        ## sit...
#         self.mod    = 'CNN_12' ## SPLITS INTO 2 !!! (SIT)
#         self.cnn    = [(128,128), (8,4), (8,3), (6,3), 64, 32] ## SIT       
#         self.cnn    = [(128,64), (10,4), (8,3), (6,3), 64, 32]
#         self.cnn    = [(128,64), (6,3), (6,3), (6,3), (3,1), 64, 32]
#         self.cnn    = [(128,64), (12,4), (8,2), (6,2), (4,2), 64, 32] ## ***********
#         self.cnn    = [(128,64), (8,2), (6,2), (-4,2), (4,2), (4,2), 64, 32]   
#         self.cnn    = [(32,32), (4,2), (4,2), (4,1), (3,1), 32]
#         self.cnn    = [(32,32), (4,2), (4,2), (4,1), (3,1), -32]
#         self.cnn    = [(32,32), (4,2), (4,2), (4,1), (3,1), 128, 64, 32]
#         self.cnn    = [(32,32), (4,2), (4,2), (4,1), (3,1), 128, 64, -32]
#         self.mod    = 'MTS_FCN' ## Fully Covolutional Baseline
#         self.mod    = 'MTS_INC' ## InceptionTime
        
        '''########################################################################'''
        self.seed = self.seed_random(seed=self.seed)
        self.gain = None
        self.fp16 = False
        self.label_smooth = 0
#         self.label_smooth = 0.2
        if self.cnn is not None:
            self.window_size = self.cnn[0][0]
        if self.test_step<1:
            self.test_step = int(self.test_step*self.window_size)
        if self.train_step<1:
            self.train_step = int(self.train_step*self.window_size)
        if self.aug_factor==0:
            self.aug_valid = False
        if self.test_id is not None:
            self.kfold = 0
        
        self.feature_dump()

################################################################################

def checknan(x):
    if (x != x).any():
        idx = np.where(np.isnan(x))[0]
        print(idx)
        raise ValueError('NaN encountered!')
        
def isint(x):
    try: return x == int(x)
    except: return False
    
def isnum(x):
    try: return abs(x)>=0
    except: return False

def unison_sorted_lists(ls, col=0, order=1):
    return list(map(list, zip(*sorted(list(zip(*ls)), key=lambda pair: order*pair[col]))))

def topk_weighted_mean(x, y, k=None, f=None):
    if np.sum(y)==0:
        return 0.0
    x,y = unison_sorted_lists([x,y], col=1, order=-1)
    x,y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    if f is None: f=lambda x:x
    if k is None: k=x.shape[0]
#     print(list(x[:k]))
#     print(list(f(y[:k])))
    return np.average(x[:k], weights=f(y[:k]))

def pr_auc(y, x):
#    auc = metrics.average_precision_score(y, x)
    p,r,t = metrics.precision_recall_curve(y, x)
    auc = metrics.auc(p, r)
    return auc

def pr_curve(Y, S, show=True):
    T = np.unique(S)
    P,R = [],[]
    for t in T:
        B = np.array([int(s > t) for s in S])
        p = metrics.precision_score(Y, B)
        r = metrics.recall_score(Y, B)
        if p==0 and r==0:
            continue
        P.append(p)
        R.append(r)
    P.append(P[-1])
    R.append(0)
    P,R = np.array(P),np.array(R)
    if show:
        plt.plot(R,P)
        plt.show()
    return P, R, T

def qwk(x,y):
    x = x - x.mean()
    y = y - y.mean()
    return 2*np.dot(x,y)/(np.dot(x,x) + np.dot(y,y))

def get_score(T, P, fxn=None, Pint=None):
    fxn = cfg.eval1 if fxn is None else fxn
    if T is None or P is None:
        return 0.0
    if fxn == 'roc':
        return metrics.roc_auc_score(T, P)
    if fxn == 'pr':
        return metrics.average_precision_score(T, P)
    if fxn == 'qwk':
        return qwk(T, P)
    if Pint is None:
        Pint = P.round()
    if fxn == 'acc':
        return metrics.accuracy_score(T, Pint)
    if fxn == 'f1':
        return metrics.f1_score(T, Pint)
    
def get_binary_score(T, proba, label, fxn=None):
    t = (T==label).astype(np.int32)
    p = proba[:,label]
    pint = (np.argmax(proba,1)==label).astype(np.int32)
    return get_score(t,p,fxn,pint)

from typing import List, Optional
def print_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List] = None,
    labels: Optional[List] = None,
    hide_zeroes: bool = False,
    hide_diagonal: bool = False,
    hide_threshold: Optional[float] = None,
):
    """Print a nicely formatted confusion matrix with labelled rows and columns.

    Predicted labels are in the top horizontal header, true labels on the vertical header.

    Args:
        y_true (np.ndarray): ground truth labels
        y_pred (np.ndarray): predicted labels
        labels (Optional[List], optional): list of all labels. If None, then all labels present in the data are
            displayed. Defaults to None.
        hide_zeroes (bool, optional): replace zero-values with an empty cell. Defaults to False.
        hide_diagonal (bool, optional): replace true positives (diagonal) with empty cells. Defaults to False.
        hide_threshold (Optional[float], optional): replace values below this threshold with empty cells. Set to None
            to display all values. Defaults to None.
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    if label_names is None:
        label_names = labels
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    # find which fixed column width will be used for the matrix
    columnwidth = max(
        [len(str(x)) for x in label_names] + [5]
    )  # 5 is the minimum column width, otherwise the longest class name
    empty_cell = ' ' * columnwidth

    # top-left cell of the table that indicates that top headers are predicted classes, left headers are true classes
    padding_fst_cell = (columnwidth - 3) // 2  # double-slash is int division
    fst_empty_cell = padding_fst_cell * ' ' + 'T/P' + ' ' * (columnwidth - padding_fst_cell - 3)

    # Print header
    print('\n  CONFUSION MATRIX -----------')
    print('    ' + fst_empty_cell, end=' ')
    for label in label_names:
        print(f'{label:{columnwidth}}', end=' ')  # right-aligned label padded with spaces to columnwidth

    print()  # newline
    # Print rows
    for i, label in enumerate(label_names):
        print(f'    {label:{columnwidth}}', end='')  # right-aligned label padded with spaces to columnwidth
        for j in range(len(label_names)):
            # cell value padded to columnwidth with spaces and displayed with 1 decimal
            cell = f'{cm[i, j]:{columnwidth}}'
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=' ')
        print()    
###################################################

cfg = Config()

npy_files = [os.path.join(cfg.data_path, f) for f in os.listdir(cfg.data_path) if f.endswith('.npy')]

if cfg.blacklist_fxn is not None:
    npy_files = np.array([f for f in npy_files if not cfg.blacklist_fxn(f)])

cfg.shuffle_list(npy_files)

## split by label...
lab_files = [[] for _ in cfg.labels]
for f in npy_files:
    lab_files[cfg.label_fxn(f)].append(f)
lab_files = [np.array(f) for f in lab_files]

def get_files(idx):
    return np.hstack([f[i] for f,i in zip(lab_files, idx)])

def get_splits(k=-1, test_id=None):
    N = [len(f) for f in lab_files]
    
    ## get test sets...
    if test_id is None:
        if k>-1:
            print('\n--------------- FOLD {} ------------------------------'.format(k+1))
            test_idx = [np.arange(k, n, cfg.kfold) for n in N]
        else: ## 0<x<1
            test_idx = [np.arange(int(cfg.test_split * n)) for n in N]
    else:
        print('NOT YET IMPLEMENTED!')##see stepcount_pytorch.py
        sys.exit()
    
    ## get train sets...
    train_idx = [np.array(list(set(np.arange(n)) - set(t))) for n,t in zip(N, test_idx)]

    ## re-shuffle train sets...
    cfg.seed_random(seed=cfg.seed + int(k))
    train_idx = [cfg.shuffle(t) for t in train_idx]
    
    ## get valid sets...
    N = [int(cfg.valid_split * len(t)) for t in train_idx]
    val_idx = [np.arange(n) for n in N]
    val_idx = [np.array(sorted(t[v])) for t,v in zip(train_idx, val_idx)]
    
    ## restore train set...
    train_idx = [np.array(list(set(t) - set(v))) for t,v in zip(train_idx, val_idx)]
    
    return get_files(train_idx), get_files(test_idx), get_files(val_idx)

def get_sets(train_files, test_files, valid_files):
    train_batcher = df.SigBatcher(train_files, cfg, train=True)
    test_batcher = df.SigBatcher(test_files, cfg, train=False)
    valid_batcher = df.SigBatcher(valid_files, cfg, train=True if cfg.aug_valid else False)
    num_train = sum([b.n for b in train_batcher.batch_stream()])
    print('\nTRAIN_SET SIZE: {0}\t\t\tHIST(y): {1}'.format(num_train, train_batcher.y_hist))
    
    #######################################
    
#     for b in train_batcher.batch_stream():
#         print(b.X.mean(0))
    
    #######################################
    
    ''' get test & valid set '''
    test_set = test_batcher.get_samples()
    valid_set = valid_batcher.get_samples(rand_shift=False)
    
    print('TEST_SET SHAPE: {0}\t{1}\tHIST(y): {2}'.format(test_set.X.shape, test_set.y.shape, test_batcher.y_hist))
    print('VALID_SET SHAPE: {0}\t{1}\tHIST(y): {2}'.format(valid_set.X.shape, valid_set.y.shape, valid_batcher.y_hist))
    
    ''' get/set input feature dimension '''
    cfg.n_inputs = test_set.X.shape[-1]
    print('NUM INPUT COLUMNS: {}\n'.format(cfg.n_inputs))
    
    return train_batcher, test_set, valid_set

def get_model(x):
    mod = None
    if cfg.mod.startswith('CNN'):
        if cfg.mod == 'CNN_0': mod = mods.CNN_0
        elif cfg.mod == 'CNN_12': mod = mods.CNN_12
        elif cfg.mod == 'CNN_12b': mod = mods.CNN_12b
        elif cfg.mod == 'CNN_13': mod = mods.CNN_13
        model = mod(cfg, x, output_dim=len(cfg.labels))
    elif cfg.mod.startswith('MTS'):#MTS_FCN
        if cfg.mod == 'MTS_FCN':
            model = mts.FCNBaseline(in_channels=cfg.n_inputs, num_pred_classes=len(cfg.labels))
        elif cfg.mod == 'MTS_INC':
            model = mts.InceptionModel(in_channels=cfg.n_inputs, num_pred_classes=len(cfg.labels),
                                       kernel_sizes=12, ##  12  8
                                       num_blocks=2,
                                       out_channels=128,#[256, 128, 64],
                                       bottleneck_channels=32,#[64, 32, 16],  32
                                       use_residuals=True)
    elif cfg.mod.startswith('SPEC'):
        if cfg.mod == 'SPEC1': mod = mods.SpecMod1
        elif cfg.mod == 'SPEC2': mod = mods.SpecMod2
        model = mod(cfg, x, output_dim=len(cfg.labels))
        
    model = model.to('cuda:0')
    
    print('\nMODEL SHAPES')
    for i in model.parameters():
        print(i.shape)
        
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    if cfg.fp16: ## apex ???
        model, optim = amp.initialize(model, optim, opt_level='O1')
    
#     criterion = torch.nn.CrossEntropyLoss()
    criterion = CrossEntropyLoss(smooth_eps=cfg.label_smooth)
    
    return model, optim, criterion

def eval_set(d, model, criterion):
    if d is None: return None,None,None,None
    with torch.no_grad():
        model.train(False)
        model.eval()
        x = np.transpose(d.X, (0,2,1))# if cfg.mod.startswith('CNN') else d.X
        x = torch.from_numpy(x).float().to('cuda:0')
        y = torch.from_numpy(d.y).to('cuda:0')
        outputs = model(x)
        loss = criterion(outputs, y)
        model.train(True)
    _, py = torch.max(outputs, 1)
    P = py.cpu().data.numpy()
    T = d.y
    proba = torch.softmax(outputs, 1).cpu().data.numpy()
    return 1-np.float(loss.cpu().data.numpy()), P, T, proba

#########################################################
        
def train_loop(k=-1, test_id=None):

    train_files, test_files, valid_files = get_splits(k, test_id)
#     [print(f) for f in valid_files]
#     sys.exit()

    train_batcher, test_set, valid_set = get_sets(train_files, test_files, valid_files)
    
    b = next(train_batcher.batch_stream())
    x = np.transpose(b.X, (0,2,1))# if cfg.mod.startswith('CNN') else b.X
    x = torch.from_numpy(x).float()
    
    model, optim, criterion = get_model(x)
    
    best_test, best_valid = -10000000.0, -10000000.0
    test_scores, valid_scores, da_scores = [],[],[]
    test_fidget_scores, test_walk_scores = [],[]
       
    print('\nEPOCH\tTRAIN(raw/aug)\tVALID:{}\tTEST:{}\tTOP-K TEST\tR/MS'.format(cfg.eval2, cfg.eval1), end='')
    if len(cfg.labels)>2:
        print('\tTEST fidg/walk\tTOP-K TEST fidg/walk', end='')
    print('')
    
    for i in range(cfg.training_epochs+1):
    
        P, T, S, Ltrain, N = [], [], [], 0, 0
        t_start = time.time()
        for b in train_batcher.batch_stream():
            
            if b.n<2: continue
            ## prepare tensors
            x = np.transpose(b.X, (0,2,1))# if cfg.mod.startswith('CNN') else b.X
            checknan(x)
            x = torch.from_numpy(x).float().to('cuda:0')
            y = torch.from_numpy(b.y).to('cuda:0')
    
            ## forward pass...
            outputs = model(x)
            checknan(outputs)
            
            loss = criterion(outputs, y)
            Ltrain += loss.item()*b.n
            N += b.n
            
            ## backward pass...
            optim.zero_grad()
            
            if cfg.fp16:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            ## optimizer step...
            optim.step()
            
            ## save predictions
            _, py = torch.max(outputs, 1)
    
            P.append(py.cpu().data.numpy())
            T.append(b.y)
            S.append(b.t)
            
        Ptrain = np.hstack(P)
        Ttrain = np.hstack(T)
        Strain = [s for ss in S for s in ss]
        Ltrain /= N
        
        ## compute rows/millisecond
        ms = (time.time() - t_start)*1000
        rate = int(cfg.window_size*N/(ms * (1+cfg.aug_factor)))
#         print('{}ms, {}r/ms'.format(ms, rate))
        
        ## get TEST and VALID predictions, and loss...
        Ltest, Ptest, Ttest, proba_test = eval_set(test_set, model, criterion)
        Lval, Pval, Tval, proba_val = eval_set(valid_set, model, criterion)
        
#        if i>2: sys.exit()
        
        ## compute SCORING METRIC for TRAIN, TEST, VALID
#        score_train, score_test = 1-Ltrain, 1-Ltest
        
        ## separate out augmented train data...    
        i_nda = [s.endswith('.nda') for s in Strain]
        i_da = [s.endswith('.da') for s in Strain]
        M = adict({'da':adict(), 'nda':adict()})
        M.nda.Ttrain, M.nda.Ptrain = Ttrain[i_nda], Ptrain[i_nda]
        M.da.Ttrain, M.da.Ptrain = Ttrain[i_da], Ptrain[i_da]
        ##
#        score_train = get_score(Ttrain, Ptrain)
        score_nda = get_score(M.nda.Ttrain, M.nda.Ptrain)
        score_da = get_score(M.da.Ttrain, M.da.Ptrain)
        
        ## TOP-K
        
#        score_valid = get_score(Tval, Pval)
        score_valid = Lval if cfg.eval2=='loss' else get_score(Tval, Pval, cfg.eval2)
        score_test = get_score(Ttest, Ptest)
#        score2_test = get_score(Ttest, Ptest, cfg.eval2)
        
        test_scores.append(score_test)
        valid_scores.append(score_valid)
        da_scores.append(score_da)
        
        test_topk = topk_weighted_mean(test_scores, valid_scores, k=cfg.k, f=lambda x: x**cfg.f)
#        val_topk = topk_weighted_mean(valid_scores, test_scores, k=cfg.k, f=lambda x: x**cfg.f)
        
        ######################################
        ## BINARY METRICS...
        if len(cfg.labels)>2:

            ## fidget...
            lab = 1
    #         val_fpr = get_binary_score(Tval, proba_val, lab, fxn='pr')
    #         val_facc = get_binary_score(Tval, proba_val, lab, fxn='acc')
    #         test_fpr = get_binary_score(Ttest, proba_test, lab, fxn='pr')
            test_facc = get_binary_score(Ttest, proba_test, lab, fxn='acc')
            
            ## walk...
            lab = 2 # 2
    #         val_wpr = get_binary_score(Tval, proba_val, lab, fxn='pr')
    #         val_wacc = get_binary_score(Tval, proba_val, lab, fxn='acc')
    #         test_wpr = get_binary_score(Ttest, proba_test, lab, fxn='pr')
            test_wacc = get_binary_score(Ttest, proba_test, lab, fxn='acc')
            
            test_fidget_scores.append(test_facc)
            test_walk_scores.append(test_wacc)

            test_topk_fidget = topk_weighted_mean(test_fidget_scores, valid_scores, k=cfg.k, f=lambda x: x**cfg.f)
            test_topk_walk = topk_weighted_mean(test_walk_scores, valid_scores, k=cfg.k, f=lambda x: x**cfg.f)

        ######################################
        
        best_test = max(best_test, score_test)
        star_test = '*' if best_test==score_test else ''
        best_valid = max(best_valid, score_valid)
        star_valid = '*' if best_valid==score_valid else ''
        
        if best_valid==score_valid and i>10:
            Pval_best = Pval
            Ptest_best = Ptest
            valid_star_test = score_test
            if cfg.model_path is not None:
                torch.save(model.state_dict(), cfg.model_path)
    
        #print('{0}\t{1:0.4}\t\t{2:0.4}{3}'.format(i, abs(score_train), abs(score_test), best))
        print('{0} [{8}]\t{1:0.3f}/{2:0.3f}\t\t{3:0.3f}{4}\t\t{5:0.3f}{6}\t\t{7:0.3f}\t\t{9}'.format(i, abs(score_nda), abs(score_da), abs(score_valid), star_valid, abs(score_test), star_test, test_topk, k+1, rate), end='')
        
        if len(cfg.labels)>2: ## binary metrics....
            r = 3
    #         print('\tFIDGET pr:{}\tacc:{}\tWALK pr:{}\tacc:{}'.format(test_fpr, test_facc, test_wpr, test_wacc), end='')
            print('\t{0:0.3f} {1:0.3f}\t{2:0.3f} {3:0.3f}'.format(test_facc, test_wacc, test_topk_fidget, test_topk_walk), end='')
        
        print('')
    
    print('\nVALID SET -------------------------------------------')
    print_confusion_matrix(Tval, Pval_best, label_names=cfg.label_names)
    print('\n  CLASSIFICATION REPORT -----------------------------')
    print(metrics.classification_report(Tval, Pval_best, digits=3))
    print('-----------------------------------------------------\n')
    
    print('\nTEST SET --------------------------------------------')
    print_confusion_matrix(Ttest, Ptest_best, label_names=cfg.label_names)
    print('\n  CLASSIFICATION REPORT -----------------------------')
    print(metrics.classification_report(Ttest, Ptest_best, digits=3))
    print('-----------------------------------------------------\n')
    
    if len(cfg.labels)>2:
        return np.array([test_topk, test_topk_fidget, test_topk_walk])
    return np.array([test_topk, valid_star_test])
    
#####################################################

if isint(cfg.test_split):
    if cfg.kfold<2:
        print('kfold < 2!!!')
        sys.exit()
    if cfg.test_split >= cfg.kfold:
        print('test_split >= kfold!!!')
        sys.exit()
    if cfg.test_split>-1:
        train_loop(k=cfg.test_split)
    else:
        test_scores = np.column_stack([train_loop(k=j) for j in range(cfg.kfold)]).transpose()
        mean_scores = test_scores.mean(0)
        print('\nTop-{} Mean Accuracy -----'.format(cfg.k))
        print('   Total Fidget   Walk')
        print(test_scores.round(4))
        print('------------------------\n {}'.format(mean_scores.round(4)))
elif isnum(cfg.test_split): # 0<x<1
    train_loop()
else:
    train_loop(test_id=cfg.test_split)