__author__ = 'vmkochegvirtual'
#
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble, linear_model
from sklearn.cross_validation import KFold
import time
from calc_cv_scores import Calc_CV_ERROR
from libs.libscores import *
from libs.data_io import *
import numpy

from time import gmtime, strftime
from calc_cv_scores import Calc_CV_ERROR,make_cross_validation
from preprocess import Preprocess_data,GBT_params,Choose_variables
from utils import make_classification,make_classification_random_forest
print(strftime("%Y-%m-%d %H:%M:%S"))



np.set_printoptions(suppress=True)

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
    gbt_params=GBT_params(n_iterations=15000,depth=8, learning_rate=0.01,subsample_part=0.6,n_max_features=gbt_features,min_samples_split=6, min_samples_leaf=3)
    gbt_params.print_params()
    return make_classification(gbt_params, train_data, labels, valid_data, test_data)
