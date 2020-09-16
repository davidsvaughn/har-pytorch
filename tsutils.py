import sys, os
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import total_ordering
import time

# import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
# import dbutils as db
# from dbutils import CREDS

def measure_time(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        #print('%r (%r, %r) %2.2f sec' % (f.__name__, args, kw, te-ts))
        print('%r %2.2f sec' % (f.__name__, te-ts))
        return result
    return timed

def isnum(x):
    return isinstance(x, (int, float, np.int32, np.int64, np.float32, np.float64))
    
def isvalidtimestamp(ts, t1='2016-01-01'):
    ts1 = pd.Timestamp(t1).timestamp() # 1451606400
    ts2 = Time(datetime.now()).add_days(1).ts
    return ts1 < ts < ts2

def offset_hours(dt):
    try:
        dt2 = datetime(2001,1,5,10)
        s = dt.tzinfo.utcoffset(dt2).seconds
        d = dt.tzinfo.utcoffset(dt2).days
        off = (s + d*60*60*24)/3600
        return -off
    except:
        return 0
        
class Ttype(Enum):
    NP = 1
    PD = 2
    DT = 3
    TS = 4
    STR = 5
    TM = 6
    DTD = 7
    UNK = 8

def tname(x):
    obj = type(x)
    return ".".join([obj.__module__, obj.__name__])
   
def ttype(x):
    name = tname(x)
    if name == 'numpy.datetime64':
        return Ttype.NP
    if name == 'pandas._libs.tslibs.timestamps.Timestamp':
        return Ttype.PD
    if name == 'datetime.datetime':
        return Ttype.DT
    if name == 'datetime.date':
        return Ttype.DTD
    if name == 'builtins.str':
        return Ttype.STR
    if isnum(x) and isvalidtimestamp(x):
        return Ttype.TS
    if name.endswith('.Time'):
        return Ttype.TM
    #####    
    print('ERROR! UNKNOWN TYPE: {} ({})'.format(name, x))
    raise ValueError('ERROR! UNKNOWN TYPE: {} ({})'.format(name, x))
    return Ttype.UNK

def trunc_hour(t):
    return t.replace(second=0, minute=0)
def trunc_day(t):
    return t.replace(second=0, minute=0, hour=0)
def trunc_time(t, mode):
    if mode == 'H':
        return trunc_hour(t)
    if mode == 'D':
        return trunc_day(t)

''' input: int/float timestamp '''
def _tod(t):
    tt = datetime.utcfromtimestamp(t)
    return tt.hour + tt.minute/60.# + tt.second/3600.
def tod(tt):
    return np.array([_tod(t) for t in tt])
    
def _remove_tz(t):
    return t.astimezone(tz=None)
    
def remove_tz(T):
    return np.array([_remove_tz(t) for t in T])

@total_ordering        
class Time(object):
    def __init__(self, t):
        tp = ttype(t)
        if tp is Ttype.DT:
            T = t.replace(tzinfo=None)
            if t.tzinfo is not None:
                h = offset_hours(t)
                T += timedelta(hours=h)
            self.T = T
        elif tp is Ttype.DTD:
            self.T = Time(str(t)).T
        elif tp is Ttype.TM:
            self.T = t.T
        elif tp is Ttype.PD:
            if t.tzinfo is not None:
                t = _remove_tz(t)
            self.T = t.to_pydatetime()
        elif tp is Ttype.NP:
            self.T = pd.Timestamp(t).to_pydatetime()
        elif tp is Ttype.TS:
            self.T = datetime.utcfromtimestamp(t)
        elif tp is Ttype.STR:
            self.T = pd.Timestamp(t).to_pydatetime()
    
    def __str__(self):
        return '{}'.format(self.T)
    def __repr__(self):
        return 'Time({})'.format(self)
    
    def __eq__(self, other):
        return (self.ts == other.ts)
    def __ne__(self, other):
        return not (self == other)
    def __lt__(self, other):
        return (self.ts < other.ts)

    @property    
    def dt(self):# datetime
        return self.T
    @property
    def ts(self):# timestamp
        return self.T.replace(tzinfo=timezone.utc).timestamp()
    @property
    def tod(self):# time of day
        return self.T.hour + self.T.minute/60. + self.T.second/3600.
    @property
    def dp(self):# date part
        return self.trunc_day()
    @property    
    def hour(self):
        return self.T.hour
    @property    
    def minute(self):
        return self.T.minute
    @property    
    def second(self):
        return self.T.second
    
    @property    
    def h(self):
        return self.T.hour + self.T.minute/60. + (self.T.second + self.T.microsecond/1000000)/3600.
    
    @property    
    def m(self):
        return self.T.minute + (self.T.second + self.T.microsecond/1000000)/60.
    
    @property    
    def s(self):
        return self.T.second + self.T.microsecond/1000000
    
    def strftime(self, fmt='%Y-%m-%dT%H:%M:%S.%fZ'):
        return self.T.strftime(fmt)# "%Y-%m-%d"
    
    def replace(self, **kwargs):
        return Time(self.T.replace(**kwargs))
            
    def trunc_hour(self, inplace=False):
        if inplace:
            self.T = trunc_hour(self.T)
        else:
            return Time(trunc_hour(self.T))
    
    def trunc_day(self, inplace=False):
        if inplace:
            self.T = trunc_day(self.T)
        else:
            return Time(trunc_day(self.T))
    
    def trunc_time(self, mode, inplace=False):
        if inplace:
            self.T = trunc_time(self.T, mode)
        else:
            return Time(trunc_time(self.T, mode))
    
    def add_hours(self, h, inplace=False):
        if inplace:
            self.T += timedelta(hours=h)
        else:
            return Time(self.T + timedelta(hours=h))
            
    def add_days(self, d, inplace=False):
        if inplace:
            self.T += timedelta(days=d)
        else:
            return Time(self.T + timedelta(days=d))
    
    @staticmethod    
    def apply_offsets(T, offsets):
        for t in T:
            t.add_hours(offsets[t.strftime("%Y-%m-%d")], inplace=True)
    
    @staticmethod    
    def trunc_times(T, mode):
        return np.array([t.trunc_time(mode) for t in T])
        
    @staticmethod       
    def aggsum(df, v2, v1='log_date', mode='D'):
        Time.df_update(df, v=v1)
        #T = df[v1]
        X = df[v2]
        T = [t.ts for t in Time.trunc_times(df[v1], mode)]
        data = db.aggsum(T, X.astype(np.float64)).astype(np.float64)
        df = pd.DataFrame({v1:data[:,0], v2:data[:,1]})
        Time.df_update(df, v=v1)
        return df
    
    @staticmethod       
    def aggmin(df, v2, v1='log_date', mode='D'):
        Time.df_update(df, v=v1)
        #T = df[v1]
        X = df[v2]
        T = [t.ts for t in Time.trunc_times(df[v1], mode)]
        data = db.aggmin(T, X.astype(np.float64)).astype(np.float64)
        df = pd.DataFrame({v1:data[:,0], v2:data[:,1]})
        Time.df_update(df, v=v1)
        return df
        
    @staticmethod        
    def df_update(df, v='log_date', typ=None, pid=None, verbose=False):
        tt = df[v]
        if ttype(next(iter(tt))) is not Ttype.TM:
            tt = [Time(t) for t in tt]
        if pid is not None:
            offsets = db.get_offsets(pid, conn_local, verbose=verbose)
            Time.apply_offsets(tt, offsets)
        if typ=='ts':
            tt = [t.ts for t in tt]
        elif typ=='dt':
            tt = [t.dt for t in tt]
#        elif typ=='tod':
#            tt = [t.tod for t in tt]
        df[v] = tt
        
    @staticmethod
    def df_add(df, typ, v='log_date'):
        tt = df[v]
        if ttype(next(iter(tt))) is not Ttype.TM:
            tt = [Time(t) for t in tt]
        if typ=='ts':
            tt = [t.ts for t in tt]
        elif typ=='tod':
            tt = [t.tod for t in tt]
        df[typ] = tt
    
    '''
    df = df.assign(e=pd.Series(x).values)
    '''
########################################################################
    
flatten = lambda l: [item for sublist in l for item in sublist]
Q = [[0,0,0,0,0],[0,0,0,1,1],[0,0,1,1,2],[0,1,1,2,3],[0,1,2,3,4]]
def smooth_ts(z):
    u,n = np.unique(z, return_counts=True)
    for i in range(len(n)-1):
        if n[i]>5 and n[i+1]<5:
            n[i]-=1
            n[i+1]+=1
    if n[-1]>5:
        z = z[:-(n[-1]-5)]
        n[-1]=5
    if n[0]>5:
        z = z[n[0]-5:]
        n[0]=5
    z = np.array(flatten([u[i]*np.ones([m],'int') for i,m in enumerate(n)]), dtype=np.int32)
    r = n[0]
    R = list(range(r))
    for i in n[1:-1]:
        m = min(4, i-1)
        R = R + list(Q[m] + r)
        r += i
    if len(n)>1:
        R = R + list(np.arange(n[-1]) + r)
    t = z[R]
    t = t + [(i+5- int(n[0]))%5/5 for i in range(len(R))]
    return t,R

def smooth_timestamps(T,X):
    Z = (np.array(T.tolist())/1000000000).astype(np.int32)
    d = np.diff(Z)
    W = np.where(d>1)[0].astype(np.int32)+1
    W = np.insert(W,0,0)
    W = np.append(W,len(Z))
    tt,xx=[],[]
    #### 
    for j in range(1,len(W)):
        z = Z[W[j-1]:W[j]]
        t,R = smooth_ts(z)
        x = X[R+W[j-1],]
        tt.append(t)
        xx.append(x)
    ####
    T = np.hstack(tt)
    X = np.vstack(xx)
    return T,X

def empty(dims, val=np.nan):
    return np.zeros(dims) + val
    
def pad_gaps(T, X, val=np.nan):
    dim = X.shape[1]
    d = np.diff(T).round(1)
    W = np.where(d>0.2)[0].astype(np.int32)  
    for w in np.flipud(W):
        n = int(5*d[w])-1
        z = np.linspace(T[w]+0.2,T[w+1],n,endpoint=False)
        T = np.insert(T, w+1, z)
        X = np.insert(X, w+1, empty([n,dim], val=val), 0) 
    return T,X

###############################################################################
''' testing '''
if __name__ == "__main__":
    
    conn_revibe = db.get_conn(CREDS.revibe)
    # conn_local = db.get_conn(CREDS.local) ## (log_date<'2019-07-20')
    conn = conn_revibe
    
    pd.Timestamp('2019-01-01T00:00:00Z').timestamp()
    
    pid = 133
    
    ''' response table '''
    v1 = 'log_date'
    v2 = 'response_type_id'
    
    sql = "SELECT log_date, response_type_id FROM public.response WHERE (person_id={}) AND (response_type_id>0) ORDER BY log_date;".format(pid)
    df = db.run_sql(sql, conn, True)
    
    
    '''    
    df = df[[v1,v2]]
    T0 = df[v1].values
    
    T1 = np.array([pd.Timestamp(t) for t in T0])
    T2 = np.array([int(t.timestamp()) for t in T1])
    T3 = [int(trunc_day(t).timestamp()) for t in T1]
    
    t1 = T0[0]
    t2 = T1[0]
    t3 = T2[0]
    
    print(ttype(t1))
    print(ttype(t2))
    print(ttype(t3))
    
    t4 = '2018-09-20 17:15:59'
    t5 = '2018-09-20T17:15:59Z'
    '''
    #sys.exit()
    
    ###############################################################################
    ''' activity response '''
    
    pid = 5065
    mo='04';day='25'
    t1 = '2019-{}-{}T00:00:00.000Z'.format(mo, day)
    t2 = '2019-{}-{}T23:59:59.000Z'.format(mo, day)
    
    a1 = 'log_date'
    a2 = 'focus'  # focus_span  focus
    tt = 60
    ## log_date | response | response_yes | response_no | focus | focus_span | point | day
    sql = "SELECT * FROM public.activity_response({}::integer, '{}'::timestamp, '{}'::timestamp, {}, array[0], 'all', true);".format(pid, t1, t2, tt)
    df = db.run_sql(sql, conn_local, True)
    
    Time.df_update(df, pid=pid)
    
    df = df[[a1, a2]]
    df = df.dropna(thresh=2)

    #Time.df_update(df, pid=pid)
    
    sys.exit()
    
    ###############################################################################
    
    sql = "SELECT * FROM public.fidget_summary WHERE (person_id=491) AND (fidget_score>0) AND (log_date BETWEEN '2019-01-16T00:00:00.000Z' AND '2019-01-16T23:59:59.000Z') ORDER BY log_date;"
    df = db.run_sql(sql, conn, True)
    df = df[['log_date','fidget_score']]
    print(df)
    
    Time.df_update(df)
    print(df)
    
    Time.df_update(df, typ='ts')
    print(df)
    
    Time.df_update(df)
    print(df)