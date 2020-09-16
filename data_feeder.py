import os, sys
import numpy as np
from scipy.fftpack import dct
import time
import random
from sklearn import metrics
from augment import DA_TimeWarp, DA_Rotation
import torch

from quant_test import bounce

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def checknan(x):
    if (x != x).any():
        idx = np.where(np.isnan(x))[0]
        print(idx)
        raise ValueError('NaN encountered!')

class Config(object):
    def __init__(self, seed=None):
        self.seed = None if seed is None else seed
        self.rng, self.seed = self.seed_random(seed=self.seed)
        self.window_size = 32
        self.train_step = 0.33
        self.test_step = 0.33
        self.batch_size = 50
        
        if self.test_step<1:
            self.test_step = int(self.test_step*self.window_size)
        if self.train_step<1:
            self.train_step = int(self.train_step*self.window_size)

        self.feats_raw = True
        self.feats_fft = True
        self.add_mag = True
        self.bounce = False
        self.dct_compress = False
        
        ## augmentation
        self.rot = True
        self.time_warp = True
        self.knot_freq = 50
        self.sigma = 0.2
        self.sigma_cut = 100
        self.aug_factor = 0
        ##
        self.label_fxn = lambda s: int('.fidget.' in s)
        
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
        rng = np.random.RandomState(seed)
        return rng, seed

    def shuffle(self, *args):
        p = self.rng.permutation(args[0].shape[0])
        return [x[p] for x in args] if len(args)>1 else args[0][p]

def ed(x, axis=-1):
    return np.expand_dims(x,axis)

def nz(X):
    return np.mean(np.all(abs(X)<0.000001, 1))
        
class SigSampler(object):
    def __init__(self, file_name, cfg, train=True):
        self.fn = file_name
        self.cfg = cfg
        self.tag = os.path.splitext(os.path.basename(file_name))[0]
        self.train = train
        self.step = cfg.train_step if train else cfg.test_step
        self.label = cfg.label_fxn(self.fn)
        self.X = np.load(self.fn)
        self.size = self.X.shape[0]
        
    def make_idx(self, rand_shift=True):
        i = self.cfg.window_size
        if self.train and rand_shift:
            i += self.cfg.rng.randint(self.step)
            i = min(i, self.size)
        self.idx = np.array(range(i, self.size+1, self.step))
        if self.train:
            self.idx = self.cfg.shuffle(self.idx)
    
    def get_mean(self, i, c):
        if c >= self.X.shape[1]:
            return None
        return self.X[i-self.cfg.window_size:i, c].mean()
    
    ## for step1->c=7, step2->c=6
    def get_steps(self, i, c=6):
        return self.get_mean(i, c=c)
        
    def sample_stream(self, augment=False, rand_shift=True):
        self.make_idx(rand_shift=rand_shift)
        X = self.X[:,:6]
        if augment:
            sigma = self.cfg.sigma * min(1.0, self.size/self.cfg.sigma_cut)
            iz = np.all(abs(X)<0.000001, 1)
            G,A = X[:,:3], X[:,3:6]
            if self.cfg.time_warp:
                A,G = DA_TimeWarp([A,G], sigma=sigma, kf=self.cfg.knot_freq, rng=self.cfg.rng)
            if self.cfg.rot:
                A,G = DA_Rotation([A,G], rng=self.cfg.rng, f=self.cfg.rot_factor)
            X = np.hstack([G,A])
            ## reset zeros to zero...
            X[iz,:] = 0
#         if self.cfg.bounce:
#             X = bounce(X, u=self.cfg.U, b=self.cfg.B, c=self.cfg.C, o=self.cfg.O)
        if self.cfg.add_mag and not self.cfg.gpu_preproc:
            G,A = X[:,:3], X[:,3:6]
            a = ed(np.sqrt(np.sum(np.square(A), axis=1)))
            g = ed(np.sqrt(np.sum(np.square(G), axis=1)))
            X = np.hstack([G, g, A, a])
        checknan(X)
        for i in self.idx:
            t = '{}[{}].{}'.format(self.tag, i, 'da' if augment else 'nda')
            q = self.get_steps(i)
            x = X[i-self.cfg.window_size:i]
            if 0<self.cfg.nz<1 and nz(x)>self.cfg.nz:
#                 print(nz(x))
                continue
            #####################
            if self.cfg.feats_fft and not self.cfg.gpu_preproc:
                f = abs(np.fft.fft(x, axis=0))
                if self.cfg.feats_raw:
                    x = np.hstack([x, f])
                else:
                    x = f
            #####################
            yield x, t, q
            
    def get_samples(self, augment=False, rand_shift=True):
#         X,T,Q = zip(*[xts for xts in self.sample_stream(augment=augment, rand_shift=rand_shift)])
        samples = [xts for xts in self.sample_stream(augment=augment, rand_shift=rand_shift)]
        if len(samples)>0:
            X,T,Q = zip(*samples)
            Y = [self.label for _ in range(len(X))]
            return X,Y,T,Q
        return None

class SigBatcher(object):
    def __init__(self, file_names, cfg, train=True):
        self.file_names = file_names
        self.cfg = cfg
        self.train = train
        self.sig_samplers = []
        for fn in file_names:
            ss = SigSampler(fn, cfg, train=train)
            if ss.size >= cfg.window_size:
                self.sig_samplers.append(ss)
            
    def get_samples(self, rand_shift=True):
        X,Y,T,Q = [],[],[],[]
        for ss in self.sig_samplers:
            samples = ss.get_samples(augment=False, rand_shift=rand_shift)
            if samples is None: continue
            x,y,t,q = samples
            X.extend(x)
            Y.extend(y)
            T.extend(t)
#            S.extend(s)
            Q.extend(q)
            if self.train and (self.cfg.time_warp or self.cfg.rot):
                for i in range(int(np.ceil(self.cfg.aug_factor))):
                    if self.cfg.aug_factor<1:
                        if self.cfg.rng.rand()>self.cfg.aug_factor:
                            break
                    samples = ss.get_samples(augment=True, rand_shift=rand_shift)
                    if samples is None: continue
                    x,y,t,q = samples
                    X.extend(x)
                    Y.extend(y)
                    T.extend(t)
                    Q.extend(q)

        X,Y,T,Q = np.array(X), np.array(Y), np.array(T), np.array(Q)
        self.y_mean = Y.mean()
        self.y_hist = np.unique(Y, return_counts=True)[1]
        if self.train:
            X,Y,T,Q = self.cfg.shuffle(X, Y, T, Q)
        return self.package(X, Y, T, Q)
    
    def batch_stream(self):
        samp = self.get_samples()
        X, Y, T, Q = samp.X, samp.y, samp.t, samp.q
        bs = self.cfg.batch_size
        for i in range(0, X.shape[0], bs):
            yield self.package(X[i:i+bs], Y[i:i+bs], T[i:i+bs], Q[i:i+bs])
            
    def package(self, X, y, t=None, q=None):
        return adict({ 'X': X, 'y': y, 't': t, 'q':q, 'n': X.shape[0] })


##########################################################################
        
def test1():
    data_path = '/home/david/data/revibe/boris/data1'
    fn = '{}/proj1.dan2.21629.fidget.10.40.37.164.npy'.format(data_path)
    
    cfg = Config()
    ss = SigSampler(fn, cfg)
    print(ss.size)
    X,Y,T,Q = ss.get_samples(augment=True)
    X = np.array(X)
    print(X.shape)
    return ss

def test2():
#    data_path = '/home/david/data/revibe/boris/data1'
#    data_path = '/home/david/data/revibe/boris/data2'
    data_path = '/home/david/data/revibe/boris/test'
    npy_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
    
    cfg = Config()
    train = True
    sb = SigBatcher(npy_files, cfg, train)
#    S = sb.get_samples()
#    print(S.X.shape)
#    print(S.y.sum()/len(S.y))
    return sb

def test3():
    data_path = '/home/david/data/revibe/boris/test'
    npy_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
    cfg = Config()
    
    cfg.gpu_preproc = True
    cfg.add_mag = False
    cfg.feats_fft = False
#    cfg.swap_ag = True
    
    train = False
    sb = SigBatcher(npy_files, cfg, train)
    return sb
    
#############################
## same as torch.norm
def torch_mag(x, dim, keepdim=False):
    m = torch.sum(torch.pow(x, 2), dim, keepdim=keepdim)
    m = torch.sqrt(m)
    return m

if __name__ == "__main__":
    sb = test3()
    b = next(sb.batch_stream())
    X = b.X
    X = np.transpose(X, (0,2,1))
    print(X.shape)

    Xt = torch.from_numpy(X).float()
    
#    Gt = Xt[:,:3,:]
#    At = Xt[:,3:,:]
    m1 = torch.norm(Xt[:,:3,:], dim=1, keepdim=True)
    m2 = torch.norm(Xt[:,3:,:], dim=1, keepdim=True)
    Xr = torch.cat((Xt, m1, m2), 1)
    
    Xf = torch.rfft(Xr, 1, onesided=False)
    Xf = torch.norm(Xf, dim=-1)
    
    sys.exit()
############################################
#    ss = test1()
    sb = test2()
    
    S = sb.get_samples()
    print(S.X.shape)
    b = next(sb.batch_stream())
    print(b.X.shape)
    X = b.X
#    print(X[0,:,0])

#    sys.exit()
    
############################################
## investigate PyTorch FFT vs Numpy FFT.....
    
    #Xr = X[:,:,:8]
    #Xf = X[:,:,8:]
    
    x = X[0]
    xr = x[:,:8]
    xf = x[:,8:]
    
    d = -1
    
    print(xr.shape)
    
    print(xf[:,d])
    
    xff = np.fft.fft(xr, axis=0)
    #print(xff[:,d])
    print(abs(xff)[:,d])
    
    Txr = torch.from_numpy(np.transpose(xr, (1,0))).float()
    xf2 = torch.rfft(Txr, 1, onesided=False)#[:,:,:].cpu().data.numpy()
    xf2 = torch.sqrt(torch.sum(torch.pow(xf2,2),-1)).cpu().data.numpy().transpose()
    
    print(xf2[:,d])
    
    sys.exit()
    ##################################3
    
    #Xr = np.transpose(Xr, (0,2,1))
    Xrt = torch.from_numpy(Xr).float()
    Xf2 = torch.rfft(Xrt, 1, onesided=False)[:,:,:,0].cpu().data.numpy()
    #Xf2 = np.transpose(Xf2, (0,2,1))
    
    xf
    Xf[0]
    Xf2[0]
    
    ###############
    
    z=xr[:,0]
    zt = torch.from_numpy(z).float()
    
    abs(np.fft.fft(z))
