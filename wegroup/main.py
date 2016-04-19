import os, sys

sys.path.append("/home/stan/Documents/dev/webEcon/wegroup/")
import utility as u
import numpy as np
import pandas as pd
from sklearn import preprocessing as pre

train_file = "data_train.txt"
test_file = "shuffle_data_test.txt"
data_folder = 'data'
columns = ['click', 'weekday', 'hour', 'timestamp', 'log_type', 'user_id', 'user_agent', 'ip', 'region', 'city',
           'ad_exchange', 'domain', 'url', 'anon_url_id', 'ad_slot_id', 'width', 'height', 'visibility', 'format',
           'price', 'creative_id', 'key_page_url', 'advertiser_id', 'user_tags']

train, test = u.import_tr_te(train_file, test_file, columns, data_folder)
train['iid'] = train.index
test['iid'] = test.index
train['sample'] = 'train'
test['sample'] = 'test'
test['click'] = np.NaN
join = pd.concat([train, test], ignore_index=True)
join['id'] = join.index  # test real id is "iid"

bad_features = ['log_type', 'anon_url_id', 'advertiser_id']  # no variance
non_features = ['id', 'user_id', 'ip', 'timestamp']  # too many levels to account for

# few levels, able to convert to dummies:
to_be_cat_feats = ['key_page_url', 'creative_id', 'format', 'user_agent', 'ad_exchange', 'sys', 'browser']
# dunno if to convert region, city to dummies
freq_feats = ['ad_slot_id', 'domain', 'url']  # features that can have popular levels
cont = ['width', 'area', 'height', 'price']

for col in freq_feats:
    df = u.count_vect(join, col)
    join = pd.merge(join, df, on=col, how='left')

simple_feat_funct = ['tfidf', 'split_user_agent', 'split_timestamp', 'part_od_day', 'multiply']
for f in simple_feat_funct:
    df = u.f(join)
    join = pd.merge(join, df, on='id', how='left')

for col in to_be_cat_feats:
    df = u.get_dummies(join, col)
    join = pd.merge(join, df, on='id', how='left')

for c in cont:
    join[c] = pre.scale(join[c])

#polynomial features
#feature selection
#truncate tfidf - truncated svd
