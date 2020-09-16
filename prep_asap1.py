import os, sys, errno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

## https://github.com/Turanga1/Automated-Essay-Scoring/blob/master/0_EDA_and_Topic_Modeling_with_LDA.ipynb
'''
cat vendor_training_set_all_info.tsv | iconv -f ISO-8859-1 -t ascii//translit > asap1_train.tsv
cat vendor_test_set_all_info.tsv | iconv -f ISO-8859-1 -t ascii//translit > asap1_test.tsv
'''

DATASET_DIR = '/home/david/data/hugface/asap/asap1'
TRAIN_FILE = 'asap1_train.tsv'
TEST_FILE = 'asap1_test.tsv'

def mkdirs(s):
    try:
        os.makedirs(s)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(s):
            pass
        
def load_data(fn, path=None):
    fn = fn if path is None else os.path.join(path, fn)
    D = pd.read_csv(fn, sep='\t')
    D = D.rename(columns={'essay_set': 'topic', 'domain1_score': 'y', 'domain2_score': 'y2'})
    D = D.drop(columns=['rater1_domain1','rater2_domain1','rater3_domain1','rater1_domain2','rater2_domain2'])
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
            
def make_set(i, X_train, X_test, ycol='y', id=None):
    id = i if id is None else '{}{}'.format(i, id)
    path = os.path.join(DATASET_DIR, 'set{}'.format(id))
    mkdirs(path)
    
    g = X_train.get_group(i)
    train_x = g['essay'].values
    y = g[ycol].values
    ymin,ymax = y.min(),y.max()
    train_y = (y-ymin)/(ymax-ymin)
    save_data('train.tsv', train_x, train_y, path=path)
    
    g = X_test.get_group(i)
    dev_x = g['essay'].values
    y = g[ycol].values
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
D.hist(column='word_count', by='topic', bins=25, sharey=True, sharex=True, layout=(2, 4), figsize=(7,4), rot=0) 
plt.suptitle('Word count by topic #')
plt.xlabel('Number of words')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(DATASET_DIR, 'word_counts.png'))
plt.show()

## get test set...
D = load_data(TEST_FILE, DATASET_DIR)
X_test = D.groupby(['topic'])

#topics = [4]
make_set(2, X_train, X_test, ycol='y', id='a')
make_set(2, X_train, X_test, ycol='y2', id='b')
sys.exit()

for t in topics:
    if t==2: continue
    make_set(t, X_train, X_test)
