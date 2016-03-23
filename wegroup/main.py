import create_feats as cf
import xgboost as xgb
import pickle

cf.execute()

xtrain = xgb.DMatrix('xtrain.buffer')
param = {}
param['objective'] = 'multi:softprob'
# param['eta'] = 1
# param['gamma']= 4
# param['max_depth'] = 8
# param['min_child_weight']=4
# param['max_delta_step']=4
# param['subsample']=0.5

param['nthread'] = 3
param['num_class'] = 2
param['eval_metric'] = 'auc'
print(xgb.cv(param, xtrain, num_boost_round=5, nfold=3, metrics=['auc'], seed=2016))
