__author__ = 'vmkochegvirtual'
#
from sklearn import ensemble, linear_model
from sklearn.cross_validation import KFold
import time
from calc_cv_scores import Calc_CV_ERROR
from libs.libscores import *
from libs.data_io import *
import numpy

from time import gmtime, strftime
from calc_cv_scores import Calc_CV_ERROR,make_cross_validation
from preprocess import Preprocess_data,GBT_params
from utils import make_classification
print(strftime("%Y-%m-%d %H:%M:%S"))



np.set_printoptions(suppress=True)

print("start loading")
start_time = time.time()
train_data = np.loadtxt('input/madeline/madeline_train.data')
test_data = np.loadtxt('input/madeline/madeline_test.data')
valid_data = np.loadtxt('input/madeline/madeline_valid.data')
labels = np.loadtxt('input/madeline/madeline_train.solution')
print("end loading , %d" % (start_time - time.time()))

(train_data,valid_data,test_data)=Preprocess_data(train_data, valid_data, test_data, labels)
n_features=train_data.shape[1]

######################### Make validation/test predictions

# gbt_params=GBT_params(n_iterations=5000,depth=6, learning_rate=0.01,subsample_part=0.6,n_max_features=(n_features/2))
# gbt_params.print_params()
#
# start_time = time.time()
# make_classification(gbt_params, train_data, labels, valid_data, test_data, 'res/madeline_valid_001.predict', 'res/madeline_test_001.predict')
# print("build ended %d seconds" % (time.time() - start_time))
#
# exit(1)


########################## Make cross validation
########################## Make cross validation
gbt_params_begin=GBT_params(n_iterations=3000,depth=5, learning_rate=0.005,subsample_part=0.7,n_max_features=50)
gbt_params_mult_factor=GBT_params(n_iterations=1,depth=2, learning_rate=2,subsample_part=1,n_max_features=2)
gbt_params_add_factor=GBT_params(n_iterations=2000,depth=1, learning_rate=0,subsample_part=1,n_max_features=0)
gbt_params_num_iter=GBT_params(n_iterations=3,depth=3, learning_rate=3,subsample_part=1,n_max_features=3)

cv_folds=5
(cv_res,cv_times)=make_cross_validation(train_data, labels, cv_folds, gbt_params_begin, gbt_params_mult_factor, gbt_params_add_factor, gbt_params_num_iter)

print("Cross Validation is complete")
print ("cv_res: ", cv_res)
print("cv_times: ", cv_times)

np.savetxt('res/madeline.crossvalidation_re1s', cv_res, '%1.5f')
np.savetxt('res/madeline.crossvalidation_times1', cv_times, '%1.5f')
exit(1)