import os, sys
# sys.path.append("/home/stan/Documents/dev/webEcon/wegroup/")
import utility as u
import numpy as np
import pandas as pd
from sklearn import preprocessing as pre
import pickle
import xgboost as xgb


def execute():
    bad_features = ['log_type', 'anon_url_id', 'advertiser_id']  # no variance
    non_features = ['click', 'id', 'user_id', 'ip', 'timestamp', 'iid', 'user_tags', 'sample',
                    'timestamp']  # for ids too many levels to account for

    # few levels, able to convert to dummies:
    to_be_cat_feats = ['key_page_url', 'creative_id', 'format', 'user_agent', 'ad_exchange', 'sys', 'browser']
    # dunno if to convert region, city to dummies
    freq_feats = ['ad_slot_id', 'domain', 'url']  # features that can have popular levels
    cont = ['width', 'area', 'height', 'price']

    # # IMPORT DATA
    train_file = "data_train.txt"
    test_file = "shuffle_data_test.txt"
    data_folder = 'data'
    columns = ['click', 'weekday', 'hour', 'timestamp', 'log_type', 'user_id', 'user_agent', 'ip', 'region', 'city',
               'ad_exchange', 'domain', 'url', 'anon_url_id', 'ad_slot_id', 'width', 'height', 'visibility', 'format',
               'price', 'creative_id', 'key_page_url', 'advertiser_id', 'user_tags']

    print("Started import of files...")
    train, test = u.import_tr_te(train_file, test_file, columns, data_folder)
    train['iid'] = train.index
    test['iid'] = test.index
    train['sample'] = 'train'
    test['sample'] = 'test'
    test['click'] = np.NaN
    join = pd.concat([train, test], ignore_index=True)
    join['id'] = join.index  # test real id is "iid"
    print("Import of files finished.")
    print("Join table: ")
    print(join.head())

    # FEAT ENGINEERING

    print("Creating frequency features...")
    for col in freq_feats:
        print(col)
        non_features.append(col)
        df = u.count_vect(join, col)
        join = pd.merge(join, df, on=col, how='left')

    print("Creating simple modified features...")
    simple_feat_funct = [u.tfidf, u.split_user_agent, u.split_timestamp, u.part_of_day, u.multiply]
    for f in simple_feat_funct:
        print(f)
        df = f(join)
        join = pd.merge(join, df, on='id', how='left')
    #
    # print("Creating dummy features...")
    # for col in to_be_cat_feats:
    #     print(col)
    #     non_features.append(col)
    #     df = u.get_dummies(join, col)
    #     join = pd.merge(join, df, on='id', how='left')

    print("Normalizing continuous variables...")
    for c in cont:
        print(c)
        join[c] = pre.scale(join[c])

    features = [col for col in join.columns if col not in bad_features and col not in non_features]

    join = join.fillna(value=0)
    join = join.set_index('sample')

    dataX = join.loc['train'][features]
    dataXte = join.loc['test'][features]

    for col in dataX.columns:
        dataX[col] = dataX[col].astype('float64')
    dataY = join['click'].astype('float64')

    xtrain = xgb.DMatrix(dataX.values, label=dataY.values)
    xtest = xgb.DMatrix(dataXte.values)

    print("Dumping features to file...")
    xtest.save_binary('xtest.buffer')
    xtrain.save_binary('xtrain.buffer')

    # polynomial features
    # feature selection
    # truncate tfidf - truncated svd
