import os, sys, errno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

## https://github.com/Turanga1/Automated-Essay-Scoring/blob/master/0_EDA_and_Topic_Modeling_with_LDA.ipynb
'''
cat train_rel_2.tsv | iconv -f ISO-8859-1 -t ascii//translit > asap2_train.tsv
cat test_rel_2.tsv | iconv -f ISO-8859-1 -t ascii//translit > asap2_test.tsv
'''
DATASET_DIR = '/home/david/data/hugface/asap/asap2'
TRAIN_FILE = 'asap2_train.tsv'
TEST_FILE = 'asap2_test.tsv'

def qwk(x, y):
    x,y = np.array(x)-np.mean(x), np.array(y)-np.mean(y)
    return 2*np.dot(x,y)/(np.dot(x,x) + np.dot(y,y))

def hh(X, i):
    g = X.get_group(i)
    y1 = g['y'].values
    y2 = g['y2'].values
    return qwk(y1,y2)
    
def compute_hh(X, topics):
    print('topic\thh')
    for t in topics:
        k = hh(X, t)
        print('{}\t{}'.format(t, np.round(k,4)))

def mkdirs(s):
    try:
        os.makedirs(s)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(s):
            pass
        
def load_data(fn, path=None):
    fn = fn if path is None else os.path.join(path, fn)
    D = pd.read_csv(fn, sep='\t')
    D = D.rename(columns={'EssaySet': 'topic', 'EssayText':'essay', 'Score1': 'y', 'Score2': 'y2'})
    D = D.dropna(axis=1)
    return D

def save_ystat(Ystat, topics, fn, path=None):
    fn = fn if path is None else os.path.join(path, fn)
    with open(fn, 'w') as f:
        for t in topics:
            f.write('{}\t{}\t{}\n'.format(t, Ystat['min'][t], Ystat['max'][t]))
    
def save_data(fn, x, y, header='sentence\tlabel', path=None):
    fn = fn if path is None else os.path.join(path, fn)
    with open(fn, 'w') as f:
        if header is not None:
            f.write(header + '\n')
        for xx,yy in zip(x,y):
            f.write('{}\t{}\n'.format(xx, yy.round(4)))
            
def make_set(i, X_train, X_test):
    path = os.path.join(DATASET_DIR, 'set{}'.format(i))
    mkdirs(path)
    
    g = X_train.get_group(i)
    train_x = g['essay'].values
    y = g['y'].values
    ymin,ymax = y.min(),y.max()
    train_y = (y-ymin)/(ymax-ymin)
    save_data('train.tsv', train_x, train_y, path=path)
    
    g = X_test.get_group(i)
    dev_x = g['essay'].values
    y = g['y'].values
    dev_y = (y-ymin)/(ymax-ymin)
    save_data('dev.tsv', dev_x, dev_y, path=path)
    
    fn = os.path.join(path, 'yrange.tsv')
    with open(fn, 'w') as f:
        f.write('{}\t{}'.format(ymin, ymax))

## get train set...
D = load_data(TRAIN_FILE, DATASET_DIR)
X_train = D.groupby(['topic'])
print(D.head())

topics = D['topic'].unique()
Ystat = D.groupby(['topic'])['y'].agg(['min','max','count','nunique'])
save_ystat(Ystat, topics, 'ystats.tsv', DATASET_DIR)
print(Ystat)

## explore data....
D.groupby('topic').agg('count').plot.bar(y='essay', rot=0, legend=False)
plt.title('Essay count by topic #')
plt.ylabel('Count')
plt.show()

D['word_count'] = D['essay'].str.strip().str.split().str.len()
D.hist(column='word_count', by='topic', bins=25, sharey=True, sharex=True, layout=(2, len(topics)//2), figsize=(7,4), rot=0) 
plt.suptitle('Word count by topic #')
plt.xlabel('Number of words')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(DATASET_DIR, 'word_counts.png'))
plt.show()

print('\nTRAIN SET HH')
compute_hh(X_train, topics)

## get test set...
D = load_data(TEST_FILE, DATASET_DIR)
X_test = D.groupby(['topic'])

#print('\nTEST SET HH')
#compute_hh(X_test, topics)
#sys.exit()

#topics = [4]
for t in topics:
    make_set(t, X_train, X_test)
