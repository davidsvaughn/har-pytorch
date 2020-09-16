import numpy as np
import pandas as pd
import json
from datetime import datetime
import psycopg2
import functools
import requests

##############################################################
## https://www.exrx.net/Calculators/WalkRunMETs
## https://www.cdc.gov/growthcharts/clinical_charts.htm
## https://help.fitbit.com/articles/en_US/Help_article/1141
##############################################################

URL = 'https://f73lzrw31i.execute-api.us-west-2.amazonaws.com/default/demo_data_server'
HEADER = {'x-api-key': 'XXXXXX'}

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def tofloat(x):
    try:
        return float(x.strip())
    except:
        return None

@functools.lru_cache(maxsize=250)
def request_demo_data(pid):
    payload = {'pid': pid}
    r = requests.post(URL, headers=HEADER, data=json.dumps(payload))
    return adict((k.strip("' "), tofloat(v)) for k,v in (item.split(':') for item in r.text[2:-2].split(',')))

#############################################################################################
#############################################################################################

revibe = adict()
revibe.DBNAME = 'revibe'
revibe.HOST = 'prd.c5fw7irdcxik.us-west-2.rds.amazonaws.com'
#revibe.PORT = '5432'
revibe.USER = 'dave'
revibe.PASS = 'tnoiSLoHjEBZE6JKsFgY'
revibe.SSLMODE = 'require'

CONN = None

def get_conn_string(creds):
    conn_str = 'host='+ creds.HOST \
    +' dbname='+ creds.DBNAME +' user=' + creds.USER \
    +' password='+ creds.PASS \
    + (' sslmode='+ creds.SSLMODE if 'SSLMODE' in creds else '') \
    + (' port='+ creds.PORT if 'PORT' in creds else '')
    return conn_str

def get_conn(creds):
    conn_str = get_conn_string(creds)
    return psycopg2.connect(conn_str)

def run_sql(sql, verbose=False):
    global CONN
    if CONN is None:
        CONN = get_conn(revibe)
    if verbose: print(sql)
    with CONN:
        data = pd.read_sql(sql, CONN)
    if verbose: print(data.shape)
    return (data)

def get_pid_data(pid):
    table = 'private.person_demographic_view'
    sql_command = "SELECT * FROM {} WHERE (person_id={});".format(table, pid)
    df = run_sql(sql_command)
    if df.size==0:
        raise ValueError('SQL returned no records:\n\t{}'.format(sql_command))
    data = adict()
    bday = df.birthday.values[0]
    sex = df.sex_id.values[0]
    grade = df.grade.values[0]
    ht = df.height.values[0]
    wt = df.weight.values[0]
    wrist = df.wrist_id.values[0]
    data.pid = pid
    data.age = None
    if bday is not None:
        data.bday = str(bday)
        bday = pd.Timestamp(str(bday)).to_pydatetime()
        data.age = np.round((datetime.now()-bday).total_seconds() / (60*60*2*365), 2) ## in months
    data.sex = None if sex==0 or sex>2 else sex
    data.grade = None if grade==0 else grade
    data.ht = None if ht==0 else ht
    data.wt = None if wt==0 else wt
    data.wrist = None if wrist==0 else wrist
    return data

## revised Harris-Benedict BMR equations...
def bmr_hb(dd, sex=None):
    try:
        sex = dd.sex if sex is None else sex
        if sex==1:
            return 6.078*dd.wt + 12.192*dd.ht - 0.473*dd.age + 88.4
        if sex==2:
            return 4.196*dd.wt + 7.874*dd.ht - 0.36*dd.age + 447.6
        return None
    except ex:
        return None

## basal metabolic rate (kCals per day)
def BMR(dd):
    try:
        if dd.sex is None:
            bmr = (bmr_hb(dd,1) + bmr_hb(dd,2))/2
        else:
            bmr = bmr_hb(dd)
        return int(round(bmr))
    except ex:
        return None

## find index j into y that minimizes abs(x-y[j])
def xminy(x, y):
    return abs(x-y).argmin(axis=-1)

class GrowthChart(object):
    ## columns: age	ht_boy	ht_girl	wt_boy	wt_girl
    def __init__(self, fn='growth.tsv'):#, path=None):
#        if path is None: path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#        fn = os.path.join(path, fn)
        df = pd.read_csv(fn, sep='\t')
        self.G = df.values
        self.S = np.array([[0.415, 0.413], [0.675, 0.57]])
        
    def fill_data(self, d):
        if d.age is None:
            if d.ht is None or d.wt is None:
                raise ValueError('Either birthday, or both height and weight, must be non-null')
        else:
            row = xminy(d.age, self.G[:,0])
        cols = np.array([d.sex] if d.sex is not None else [1, 2])
        if d.ht is None:
            d.ht = self.G[row, cols].mean()
        if d.wt is None:
            d.wt = self.G[row, cols+2].mean()
        d.ws = np.round(d.ht * self.S[0, cols-1].mean(), 2) ## walk stride
        d.rs = np.round(d.ht * self.S[1, cols-1].mean(), 2) ## run stride
        #d.bmr = BMR(d) ## basal metabolic rate (kCals per day)

GC = None

@functools.lru_cache(maxsize=250)
def get_demo_data(pid):
    data = get_pid_data(pid)
    global GC
    if GC is None:
        GC = GrowthChart()
    GC.fill_data(data)
    return data

def fixnum(x, dtype=float):
    if x is None: return None
    x = dtype(x)
    if x==0: return None
    return x

def validate_demo_data(data):
    data.ht = fixnum(data.ht)
    data.wt = fixnum(data.wt)
    data.sex = fixnum(data.sex, int)
    if data.sex is not None and data.sex>2:
        data.sex = None
    data.age = None
    if data.bday is None:
        if data.ht is None or data.wt is None:
            raise ValueError('Either birthday, or both height and weight, must be non-null')
    else:
        bday = pd.Timestamp(str(data.bday)).to_pydatetime()
        data.age = np.round((datetime.now()-bday).total_seconds() / (60*60*2*365), 2) ## in months
#        data.bday = data.bday.strftime('%Y-%m-%d')

@functools.lru_cache(maxsize=250)
def make_demo_data(bday=None, ht=None, wt=None, sex=None):
    data = adict()
    data.bday = bday or None
    data.ht = ht or None
    data.wt = wt or None
    data.sex = sex or None
    validate_demo_data(data)

    #########
    global GC
    if GC is None:
        GC = GrowthChart()
    GC.fill_data(data)
    return data

## s : speed in mph... sec by second vector of speeds....
## w : weight in lbs
## mode : 2=='walk', 3=='run'
## returns : calories summed across all seconds
def calsum(s, w, mode=2):
    su, wu = 26.8, 2.2
    s = s*su
    w = w/wu
    if mode==3:## run mode
        vo = 0.2*s
    else:      ## walk mode == 2
        fwvo = 21.11 - 0.3593*s + 0.003*s*s - 3.5
        wvo = 0.1*s
        d = 30
        a = np.clip((s-(100-d))/(2*d), 0, 1)
        vo = wvo*(1.-a) + fwvo*a
    #############################
    return np.sum(vo*w) / 12000.0
    
###################################
if __name__ == "__main__":
    pid = 135       ## 135,"1974-05-28",1,0,74,196,1
    pid = 169       ## 169,"1980-12-01",1,12,72,170,2
    pid = 18947     ## 18947,"2010-08-28",0,0,0,0,0
    pid = 10885     ## 
    
#    dd = request_demo_data(pid)
#    print(dd)
    
#     dd = get_demo_data(pid)
#     print(dd)
    #############
    
    dd = make_demo_data(bday='2010-08-28', ht='54.035', wt='69.69', sex='3')
#    dd = make_demo_data(ht='70', wt='120', sex='2')
    
    print(dd)
    