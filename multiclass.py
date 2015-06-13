__author__ = 'vmkocheg'

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble, linear_model
from sklearn.cross_validation import KFold
import time
import numpy as np
from sets import Set
from time import gmtime, strftime

from preprocess import Preprocess_data,GBT_params,Choose_variables
from utils import make_classification,make_classification_random_forest

print(strftime("%Y-%m-%d %H:%M:%S"))

np.set_printoptions(suppress=True)

def multiclass_predict(train_data,labels,valid_data,test_data,output_dir,time_budget):
    print("make multiclass prediction\n")
    start_time = time.time()
    np_seed = int(time.time())
    np.random.seed(np_seed)
    print ("np seed = " , np_seed)
    print(train_data.shape)

    print("train_data.shape == %d\n",train_data.shape)
    n_features = train_data.shape[1]
    FS_iterations = 5000/labels.shape[1]
    select_clf = ExtraTreesClassifier(n_estimators=FS_iterations,max_depth=5)
    select_clf.fit(train_data, labels)
    my_mean =1./(10*n_features)
    train_data = select_clf.transform(train_data,threshold=my_mean )
    valid_data = select_clf.transform(valid_data,threshold=my_mean )
    test_data = select_clf.transform(test_data,threshold=my_mean)

    print(my_mean)
    print(train_data.shape)
    # print(np.sort(select_clf.feature_importances_))
    # print(np.sort(select_clf.feature_importances_[np.where(select_clf.feature_importances_ > 0)]))

    ######################### Make validation/test predictions
    n_features=train_data.shape[1]
    if n_features < 700:
        gbt_features=n_features
    else:
        gbt_features=int(n_features**0.5)
    gbt_iterations= (time_budget / 1000.) * 3000000/(gbt_features * labels.shape[1])
    gbt_params=GBT_params(n_iterations=20000,depth=8, learning_rate=0.01,subsample_part=0.6,n_max_features=gbt_features,min_samples_split=5, min_samples_leaf=3)
    gbt_params.print_params()

    return make_classification(gbt_params, train_data, labels, valid_data, test_data)
