import os,sys
import json, random
import numpy as np
import traceback
import asyncio
import onnxruntime

# import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

import quantize as qz
import calories as cal

import warnings
warnings.filterwarnings("ignore")

#####################################

## sitting/standing related parameters....

BLIP = 3        ## seconds
DORMANT = 200   ## NaN (instead of zeros) if no data more than this # seconds
FILL_SIT = 400  ## fill back in if below (seconds)
FILL_STAND = 30 ## fill back in if below (seconds)
SS_SMOOTH = 100 ## seconds...re-evaluate isolated intervals smaller than this...

## toggle sitting/standing correction for fidget/walk/run...
SS_CORR = 1
## output sit/stand intervals...
SS = 1

## other parameter default values....
QUANT = 8
GPU = 0
DELTA = 0
PID,BDAY,SEX,HT,WT = None,None,None,None,None

NUM_FEATURES = 6
WINDOW_SIZE = 32
STEP = 8

SIT_WINDOW_SIZE = 128
SIT_STEP = 32
SIT_NZ = 0.75

##
CHUNK_SIZE = 500
LOG_LEVEL = 1 # 1

## output: t1, t2, label, steps, dist(ft), cals
HAR_MODEL = 'har_model.onnx'
STEP_MODEL = 'step_model.onnx'
SIT_MODEL = 'sit_model.onnx'
DELTA_MODEL = 'delta_model.onnx'
SS_MODEL = 'ss_model.onnx'

###############################################################################

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

## for each index i into x, find index j into y that minimizes abs(x[i]-y[j])
def xminy(x, y):
    return abs(x[:, None] - y[None, :]).argmin(axis=-1)
#     return (x[:, None] - y[None, :]).argmax(axis=-1)

def uniq(x):
    nz = np.nonzero(np.diff(np.insert(x,0,x[0]-1)))[0]
    return x[nz], np.diff(np.append(nz, len(x))), nz

def nz(X):
    #return np.mean(np.all(abs(X)<0.000001, 1))
    return np.mean(abs(X)<0.000001)

def update_params(p, params, name, default, cast):
    p[name] = cast(params[name]) if name in params else default

def parse_params(params={}):
    p = adict()
    
    update_params(p, params, 'quant', QUANT, int)
    
    update_params(p, params, 'pid', PID, int)
    update_params(p, params, 'birthday', BDAY, str)
    update_params(p, params, 'sex_id', SEX, int)
    update_params(p, params, 'height', HT, int)
    update_params(p, params, 'weight', WT, int)
    
    update_params(p, params, 'chunk_size', CHUNK_SIZE, int)
    update_params(p, params, 'log_level', LOG_LEVEL, int)
    
    update_params(p, params, 'win', WINDOW_SIZE, int)
    update_params(p, params, 'step', STEP, int)
    update_params(p, params, 'nf', NUM_FEATURES, int)
    update_params(p, params, 'blip', BLIP, int)
    update_params(p, params, 'delta', DELTA, int)
    
    update_params(p, params, 'sit_win', SIT_WINDOW_SIZE, int)
    update_params(p, params, 'sit_step', SIT_STEP, int)
    update_params(p, params, 'sit_nz', SIT_NZ, float)
    update_params(p, params, 'ss_corr', SS_CORR, int)
    
    update_params(p, params, 'ss_smooth', SS_SMOOTH, int)
    update_params(p, params, 'dormant', DORMANT, int)
    update_params(p, params, 'fill_sit', FILL_SIT, int)
    update_params(p, params, 'fill_stand', FILL_STAND, int)
    
    update_params(p, params, 'ss', SS, int)
    
    p.gpu = False ## generate magnitudes and FFT in numpy

    ## get demographic data....
#    dd = cal.request_demo_data(p.pid) ## API call to separate lambda fxn (lambda_demo_data)    
    if p.pid is None:
        p.dd = cal.make_demo_data(bday=p.birthday, ht=p.height, wt=p.weight, sex=p.sex_id)
    else:
        p.dd = cal.get_demo_data(p.pid)     ## internal DB call
    
    return p

def sliding_windows(data_sets, win, step, gpu, nz_max=None):
    W,I,L = [],[],[]
    for ds in data_sets:
        n = len(ds)
        if n<win: continue
        idx = np.arange(win, n+1, step)
        if idx[-1]!=n: 
            idx = np.append(idx, n)
            L.append(idx[-1]-idx[-2])
        else:
            L.append(0)
        for i in idx:
            x = ds[i-win:i]
            if nz_max is not None and nz(x[:,1:])>nz_max:
#                 print('NZ: {} - {}\t{}-{}'.format(Time(x[0,0]), Time(x[-1,0]), int(x[0,0]), int(x[-1,0])))
                continue
            if not gpu: ## add FFT of data
                f = abs(np.fft.fft(x[:,1:], axis=0))
                x = np.hstack([x, f])
            W.append(x)
        I.append(len(W))
        
    # get time windows
    if len(W)==0:
        return [], 200
    
    T = np.array([[w[0,0],w[-1,0]] for w in W]).astype(np.int64)
    X = np.array([w[:,1:].astype(np.float32) for w in W])
    I = np.array(I[:-1])
    L = np.array(L)
    
    return X,T,I,L

def expand_gaps(data_sets, v, c, idx, tj, p):
    dim = data_sets[0].shape[1]-1
    new_sets = [[data_sets[0]]]
    for i,j in enumerate(tj):
        a,b = v[j-1],v[j]+1
        if (b-a)<p.dormant: ## fill in gap with zeros...
            a += 0.2*c[j-1]
            b -= 0.2*c[j]
            t = np.arange(a, b, 0.19).astype(np.int64)
            d = np.zeros([len(t),dim])
            e = np.hstack([t[:,None], d])
            new_sets[-1].append(e)
        else: ## else start a new data block...
            new_sets.append([])
        new_sets[-1].append(data_sets[i+1])
    return [np.vstack(d) for d in new_sets]

def is_sorted(a):
    return np.all(a[:-1] <= a[1:])
        
def process_input(D, p):
    win  =p.win
    
    if p.log_level>2:
        print('input:')
        print(D)
    
    if p.quant>0:
        if len(D.strip())==0: return [], 200
        D = np.array(list(map(int, D.split(','))), dtype=np.float64)
        D = D.reshape([-1, p.nf+1])
        if D.shape[0]<win: return [], 200
        
        if p.delta:
            D = D.cumsum(axis=0)
        
        # fix ordering!!!
        i,j = 0,1
        if not is_sorted(D[:,0]):
            D = np.hstack([np.expand_dims(np.arange(D.shape[0]),1), D])
            D = D[np.lexsort((D[:,0], D[:,1]))]
            i,j = 1,2

        t = D[:,i].astype(np.int64)
        D = D[:,j:]
        
        # de-quantize
        D = qz.dequant(D, quant=p.quant)
        
        ## testing
#         plt.plot(D[:,:3]); plt.show()
#         plt.plot(D[:,3:]); plt.show()
        
    else:
        t = D[:,0].astype(np.int64)
        D = D[:,1:]
    
    # add magnitudes
    if not p.gpu:
        G,A = D[:,:3], D[:,3:6]
        a = np.sqrt(np.sum(np.square(A), axis=1))
        g = np.sqrt(np.sum(np.square(G), axis=1))
        D = np.hstack([G, g[:,None], A, a[:,None]])
    D = np.hstack([t[:,None], D])
    
    ## find splits...
    v,c,idx = uniq(t)
    j = np.diff(v, prepend=v[0]-1)
    tj = np.where(j>p.blip)[0]
    splits = idx[tj]
    data_sets = np.split(D, splits)
    
    har_data = sliding_windows(data_sets, p.win, p.step, p.gpu)
    data_sets = expand_gaps(data_sets, v, c, idx, tj, p)
    sit_data = sliding_windows(data_sets, p.sit_win, p.sit_step, p.gpu, p.sit_nz)
    
    return har_data, sit_data

def smooth_labels(y, thresh=10):
    x = y.copy()
    id0 = x==0
    x[id0]=1
    id3 = x==3
    x[id3]=2
    v,c,_ = uniq(x)
    
    for i in np.arange(1,len(v)-1):
        if v[i]!=1: continue
        ## v[i]==1....
        if c[i-1]>thresh and c[i+1]>thresh and c[i]<min(c[i-1], c[i+1]):
            v[i]=2
            
    ## re-assemble label vector from v,c
    x = np.hstack([np.ones(c[i])*n for i,n in enumerate(v)]).astype(np.int32)
    x[id0]=0
    x[id3]=3
    return x

def smooth_labels_2(y, thresh=2):
    x = y.copy()
    v,c,_ = uniq(x)
    
    for i in np.arange(1,len(v)-1):
        if c[i]>1: continue
        if v[i-1] != v[i+1]: continue
        if c[i-1]>thresh and c[i+1]>thresh:
            v[i]=v[i-1]
            
    ## re-assemble label vector from v,c
    x = np.hstack([np.ones(c[i])*n for i,n in enumerate(v)]).astype(np.int32)
    return x

def smooth_labels_3(y, thresh=2):
    x = y.copy()
    v,c,_ = uniq(x)
    for i in np.arange(1,len(v)-1):
        if v[i]==1: continue
        if v[i-1]!=1 or v[i+1]!=1: continue
        if c[i-1]>thresh and c[i+1]>thresh and c[i]==1:
            v[i]=1
    ## re-assemble label vector from v,c
    x = np.hstack([np.ones(c[i])*n for i,n in enumerate(v)]).astype(np.int32)
    return x

SS_SESS = None
def process_sit_output(data, B, D, p, verbose=False):
    global SS_SESS
    SS_SESS = onnxruntime.InferenceSession(SS_MODEL) if SS_SESS is None else SS_SESS
    
    _,T,I,_ = data
    TT = np.split(T, I) ## timestamps
    BB = np.split(B, I) ## sit label (sitting/standing)
    DD = np.split(D, I) ## delta label (sit down/stand up)
    Z = []
    ts,bs = [],[]
    L = []
    for tt,bb,dd in zip(TT,BB,DD):
        if len(tt)<p.blip: continue
        tmin = np.min(tt)
        tmax = np.max(tt)
        n = tmax-tmin+1
        h = [[] for _ in range(n)]
        g = [[] for _ in range(n)]
        for t,b,d in zip(tt,bb,dd):
            for i in np.arange(t[0]-tmin, t[1]-tmin+1):
                h[i-1].append(b)
                g[i-1].append(d)
        t = np.array([i for i in range(tmin, tmax+1)])
        b = np.array([np.array(l).mean(0) for l in h])
        d = np.array([np.array(l).mean(0) for l in g])

        ## check for nan and fix...
        try:
            nan = np.any(np.isnan(b), 1)
        except TypeError:
            nan = np.array([np.isnan(_b.sum()) for _b in b])
        if np.any(nan):
            i,j = np.where(nan)[0],np.where(~nan)[0]
            k = xminy(i,j)
            b[i] = b[j[k]]
            d[i] = d[j[k]]
            
        bp = np.array([_b[1] for _b in b]).round(4)     ## standing probability
        sitp = np.array([_d[1] for _d in d]).round(4)   ## sit_down probability
        standp = np.array([_d[2] for _d in d]).round(4) ## stand_up probability
        Q = np.column_stack([bp, sitp, standp])
        
        inputs = {SS_SESS.get_inputs()[0].name: Q}
        b = np.array(SS_SESS.run(None, inputs)).squeeze().argmax(1)
        
        '''
        ############################################################
        if STATS:
            Q = np.column_stack([t, bp, sitp, standp])
            np.savetxt('out{}.txt'.format(out), Q, fmt='%d\t%0.4f\t%0.4f\t%0.4f')
            out+=1
            print('saving!!')
        ############################################################
        '''
        ts.append(t)
        bs.append(b)
        L.extend(t[b==1])
    
    ''' return '''
    code = 200
    L = set(L)
    return L, ts, bs

def combine(ts, bs):
    splits = np.array([len(b) for b in bs[:-1]]).cumsum()
    T = np.hstack(ts)
    B = np.hstack(bs)
    return T, B, splits

def process_har_output(data, Y, S, L, ts, bs, p, verbose=False):
    if len(ts)>0:
        T, B, splits = combine(ts, bs)
        if len(Y)==0:
            return '[]', T, B, splits, code
    ##################
    _,T2,I,_ = data
    TT = np.split(T2, I) ## timestamps
    YY = np.split(Y, I) ## har label 
    SS = np.split(S, I) ## steps
    
    ##################
    if len(ts)==0:
        ## if no sit/stand data (cuz win<24 sec)....
        splits = []
        tmin = np.min(T)
        tmax = np.max(T)
        T = np.array([i for i in range(tmin, tmax+1)])
        B = T*0
    ##################
    
    dd = p.dd
    X = []
    for tt,yy,ss in zip(TT,YY,SS):
        if len(tt)<p.blip: continue
        tmin = np.min(tt)
        tmax = np.max(tt)
        n = tmax-tmin+1
        f = [[] for _ in range(n)]
        g = [[] for _ in range(n)]
        for t,y,s in zip(tt,yy,ss):
            for i in np.arange(t[0]-tmin, t[1]-tmin+1):
                f[i-1].append(y)
                g[i-1].append(s)
        t = np.array([i for i in range(tmin, tmax+1)])
        
        ## HAR labels....
        y = np.array([np.array(l).mean(0) for l in f])
        
        ## check for nan and fix...
        try:
            nan = np.any(np.isnan(y), 1)
        except TypeError:
            nan = np.array([np.isnan(_y.sum()) for _y in y])
        if np.any(nan):
            i,j = np.where(nan)[0],np.where(~nan)[0]
            k = xminy(i,j)
            y[i] = y[j[k]]
        try:
            y = np.array(y).argmax(1)
        except ValueError:
            y = np.array([np.argmax(_y) for _y in y])
            
        ## smooth over questionable fidget labels (sandwiched between walking or running)...
        y = smooth_labels(y)
        y = smooth_labels_2(y)
        if verbose: print('\n{}'.format(y))
        
        ## re-build sitting/standing array...
        b = np.array([int(_t in L) for _t in t]) ## 0=sitting, 1=standing (state event)
        
        ## SS => HAR
        ## fidget only if also sitting...
        y[y==1] = y[y==1]*(1-b[y==1])
        ## walk/run only if standing...
#         y[y>1] = y[y>1]*b[y>1]

        ## HAR => SS
#         b[y>1] = 1## if walking or running, then standing
        idx = np.in1d(T, t[y>1])
        B[idx] = 1

        ## STEPS.....
        sps = 5.26 * np.array([np.nanmean(z) for z in g]) ## steps/sec for each sec in t
        mps = sps * ((y==2)*dd.ws + (y==3)*dd.rs) / 63360  ## miles/sec for each sec
        mph = mps * 3600 ## mph for each sec
        
#         if verbose:
#             import matplotlib.pyplot as plt
#             plt.plot(mph)
#             plt.show()
        
        ## find connected regions... label==[1,2,3]
        v,c,z = uniq(y)
        for k in range(len(v)):
            if v[k]==0: continue ## skip regions with HAR==0 label
            if c[k]<p.blip: continue
            
            label = v[k]
            i,j = z[k], z[k]+c[k]
            t1,t2 = t[i],t[j-1] ## start,end times
            
            steps, dist, cals = 0,0,0
            ## if walk/run (not fidget)...
            if label>1:
                steps = sps[i:j].sum() ## sum over seconds...
                dist = mps[i:j].sum() ## sum over seconds...
                cals = cal.calsum(mph[i:j], dd.wt, label)
                ## if verbose: print('{0}\t{1:0.2f} steps\t{2:0.2g} miles\t{3:0.2g} kCals'.format(label, steps, dist, cals))
                steps = int(np.round(steps))
                dist = int(np.round(dist*5280)) ## convert miles to feet
                cals = int(np.round(cals*1000)) ## convert kCals to cals
#                 if verbose: print('{0}\t{1} steps\t{2} feet\t{3} cals'.format(label, steps, dist, cals))
            X.append([t1, t2, label, steps, dist, cals])
    
    ##
    if len(B)<30:
        B = B*0 + int(np.mean(B).round())
    
    ''' return '''
    code = 200
    X = np.array(X).flatten()
    return str(X.astype(np.int32).tolist()), T, B, splits, code

def summ_sit_data(T, B, splits, p):
    tt = np.split(T, splits) ## timestamps
    bb = np.split(B, splits) ## sit label (sitting/standing)
    
    Z = []
    for t,b in zip(tt,bb):
        ## find connected regions... label==[1]
        v,c,z = uniq(b)
        for k in range(len(v)):
            i,j = z[k],z[k]+c[k]
            t1,t2 = t[i],t[j-1] ## start,end times
            Z.append([t1, t2, v[k]])
    
    ''' re-fill smaller dormant gaps.... '''
    Z = np.array(Z)
    x,y,z = uniq(Z[:,-1])
    
    ## dormant gap after sitting, before standing...
    for i,z2 in enumerate(Z):
        if i==0: continue
        if z2[2]!=1: continue
        if Z[i-1,2]!=0: continue
        z1 = Z[i-1]
        if z2[0]-z1[1]<p.blip: continue
        if z1[1]-z1[0]>p.ss_smooth and z2[0]-z1[1]<p.fill_sit:
            Z[i-1,1] = z2[0]-1
    
    ## dormant gap between 2 identical labels...
    idx = np.where(y>1)[0]
    if len(idx)>0:
        blist = []
        for j in idx:
            fill = p.fill_stand if x[j] else p.fill_sit
            for i in range(z[j]+1, z[j]+y[j]):
                t1,t2 = Z[i-1,1], Z[i,0]
                if t2-t1<fill:
                    Z[i,0] = Z[i-1,0]
                    blist.append(i-1)
        k = np.ones(Z.shape[0]).astype(np.bool)
        k[blist] = False
        Z = Z[k]
    
    ### change labels: [0,1] -> [1,2]
    Z[:,-1] = Z[:,-1]+1
    ### check for duplicate start/end times.....
    if np.unique(Z[:,0]).shape[0] < Z.shape[0]:
        print('DUPLICATE START TIMES')
    if np.unique(Z[:,1]).shape[0] < Z.shape[0]:
        print('DUPLICATE END TIMES')
    
    return str(Z.astype(np.int32).flatten().tolist())

HAR_SESS, STEP_SESS, SIT_SESS, DELTA_SESS = None, None, None, None
async def model_loop(X, p, model='har', verbose=False):
    global HAR_SESS, STEP_SESS, SIT_SESS, DELTA_SESS
    if model=='har':
        HAR_SESS = onnxruntime.InferenceSession(HAR_MODEL) if HAR_SESS is None else HAR_SESS
        sess = HAR_SESS
    elif model=='step':
        STEP_SESS = onnxruntime.InferenceSession(STEP_MODEL) if STEP_SESS is None else STEP_SESS
        sess = STEP_SESS
    elif model=='sit':
        SIT_SESS = onnxruntime.InferenceSession(SIT_MODEL) if SIT_SESS is None else SIT_SESS
        sess = SIT_SESS
    else:
        DELTA_SESS = onnxruntime.InferenceSession(DELTA_MODEL) if DELTA_SESS is None else DELTA_SESS
        sess = DELTA_SESS
        
    ys = []
    N = p.chunk_size
    
    ## loop over batches...
    for i in range(0, len(X), N):
        x = np.stack(X[i:i+N])
        x = np.transpose(x, (0,2,1)).astype(np.float32)
        inputs = {sess.get_inputs()[0].name: x}
        y = np.array(sess.run(None, inputs)).squeeze()
        ys.append(y)
        
    ## post process results
    if len(ys)==0:
        return np.array([])
    if len(ys[0].shape)>1:
        return np.vstack(ys)
    if len(ys)==1:
        return ys[0]
    return np.hstack(ys)

def propa(x, n=1):
    for i in range(n):
        xx = x[1:]+x[:-1]
        x[:-1] += xx
        x[1:] += xx
        x = (x>0).astype(np.int32)
    return x

def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis, keepdims=True))
    return e_x / e_x.sum(axis, keepdims=True)

def run_async(data, params={}):
    p = parse_params(params)
    return_empty = '[]', ('[]' if p.ss else None), 200
    
    data = process_input(data, p)
    if len(data[0])==0:
        return return_empty
    
    har_data, sit_data = data
    
    ## invoke models...
    Y_func = model_loop(har_data[0], p, model='har')
    S_func = model_loop(har_data[0], p, model='step')
    B_func = model_loop(sit_data[0], p, model='sit')
    D_func = model_loop(sit_data[0], p, model='delta')
    
    loop = asyncio.get_event_loop()
    Y, S, B, D = loop.run_until_complete(asyncio.gather(Y_func, S_func, B_func, D_func))
    
    if len(B)<2 or len(D)<2 or len(B.shape)<2 or len(D.shape)<2:
        return return_empty
    
    ## apply softmax ???
    B = softmax(B)
    D = softmax(D)
    
    ## process output...
    L, ts, bs = process_sit_output(sit_data, B, D, p)
    if len(ts)==0 and len(har_data)==2:
        return return_empty
    X, T, B, splits, code = process_har_output(har_data, Y, S, L, ts, bs, p)
    Z = summ_sit_data(T, B, splits, p)
    
    if p.log_level>1:
        print('labels: {}'.format(X))
    
    return X, (Z if p.ss else None), code
        
def package_return_string(s, code=200):
    return {
        "statusCode": code,
        "body": json.dumps(s)
    }
    
def lambda_handler(event, context):
    try:
        try:
            body = json.loads(event['body'])
        except KeyError:
            body = json.loads(json.dumps(event))
        
        data = body['data']
        del body['data']
        params = {}
        if 'meta' in body:
            params = body['meta']
            del body['meta']
            params = dict((k.strip(), v.strip()) for k,v in (item.split('=') for item in params.split(';')))
        params.update(body)
        update_params(params, params, 'chunk_size', os.environ['CHUNK_SIZE'], int)
        update_params(params, params, 'log_level', os.environ['LOG_LEVEL'], int)
        
        X, Z, code = run_async(data, params)
        
        output = X if Z is None else '{}|{}'.format(X,Z)
        return package_return_string(output, code)
        
    except Exception as ex:
        ###############
        print('ERROR:')
        print(body)
        print(data)
        ###############
        exception_type = ex.__class__.__name__
        exception_message = str(ex)
        exception_traceback = traceback.format_exc()
        exception_obj = {
            "isError": True,
            "type": 'Lambda: {}'.format(exception_type),
            "message": exception_message,
            "traceback": exception_traceback,
            "params": params
#             "data": data
        }
        ## wrap exception inside LambdaException
        exception_json = json.loads(json.dumps(exception_obj))
        raise LambdaException(exception_json)

class LambdaException(Exception):
    pass
