__author__ = 'vmkochegvirtual'
#
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
import time
import numpy as np

from time import gmtime, strftime
from utils import make_classification
print(strftime("%Y-%m-%d %H:%M:%S"))
from preprocess import GBT_params

def madeline_predict(train_data,labels,valid_data,test_data,output_dir):
    print("make madeline prediction\n")
    start_time = time.time()
    np_seed = int(time.time())
    np.random.seed(np_seed)
    print ("np seed = " , np_seed)
    print(train_data.shape)

    select_clf = ExtraTreesClassifier(n_estimators=5000,max_depth=4)
    select_clf.fit(train_data, labels)
    my_mean =np.percentile(select_clf.feature_importances_,94)

    train_data = select_clf.transform(train_data,threshold=my_mean )
    valid_data = select_clf.transform(valid_data,threshold=my_mean )
    test_data = select_clf.transform(test_data,threshold=my_mean)

    print(train_data.shape)
    # print(np.where(select_clf.feature_importances_ > my_mean))
    # print(np.sort(select_clf.feature_importances_))

    ######################### Make validation/test predictions
    n_features=train_data.shape[1]
    gbt_features=n_features
    gbt_params=GBT_params(n_iterations=30000,depth=9, learning_rate=0.01,subsample_part=0.6,n_max_features=gbt_features,min_samples_split=4, min_samples_leaf=2)
    gbt_params.print_params()
    return make_classification(gbt_params, train_data, labels, valid_data, test_data)
