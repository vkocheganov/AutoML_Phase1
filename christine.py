__author__ = 'vmkochegvirtual'

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
import time
import numpy as np
from sets import Set
from time import gmtime, strftime

from preprocess import GBT_params
from utils import make_classification

print(strftime("%Y-%m-%d %H:%M:%S"))

def christine_predict(train_data,labels,valid_data,test_data,output_dir):
    print("make christine prediction\n")
    start_time = time.time()
    np_seed = int(time.time())
    np.random.seed(np_seed)
    print ("np seed = " , np_seed)
    print(train_data.shape)

    select_clf = ExtraTreesClassifier(n_estimators=5000,max_depth=5)
    select_clf.fit(train_data, labels)
    my_mean =0.0001#np.mean(select_clf.feature_importances_)#np.percentile(select_clf.feature_importances_,20)
    train_data = select_clf.transform(train_data,threshold=my_mean )
    valid_data = select_clf.transform(valid_data,threshold=my_mean )
    test_data = select_clf.transform(test_data,threshold=my_mean)

    print(my_mean)
    print(train_data.shape)

    ######################### Make validation/test predictions
    n_features=train_data.shape[1]
    gbt_features=int(n_features**0.5)
    gbt_params=GBT_params(n_iterations=30000,depth=10, learning_rate=0.01,subsample_part=0.6,n_max_features=gbt_features,min_samples_split=4, min_samples_leaf=2)
    gbt_params.print_params()

    return make_classification(gbt_params, train_data, labels, valid_data, test_data)
