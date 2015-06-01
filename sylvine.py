__author__ = 'vmkochegvirtual'
from sets import Set
#
#order = christine, jasmine.py, madeline, philippine, sylvine
from sklearn import ensemble, linear_model
from sklearn.cross_validation import KFold
from sklearn.ensemble  import ExtraTreesClassifier
import time
import numpy
from calc_cv_scores import Calc_CV_ERROR
from libs.libscores import *
from libs.data_io import *

from time import gmtime, strftime
from calc_cv_scores import Calc_CV_ERROR
from preprocess import Preprocess_data,GBT_params,Choose_variables
from utils import make_classification
from calc_cv_scores import make_cross_validation
print(strftime("%Y-%m-%d %H:%M:%S"))

imports_file='res/sylvine.importances'
start_time = time.time()
train_data = np.loadtxt('input/sylvine/sylvine_train.data')
test_data = np.loadtxt('input/sylvine/sylvine_test.data')
valid_data = np.loadtxt('input/sylvine/sylvine_valid.data')
labels = np.loadtxt('input/sylvine/sylvine_train.solution')
print("end loading , %d" % (start_time - time.time()))

start_time = time.time()
np_seed = int(time.time())
np.random.seed(np_seed)
print ("np seed = " , np_seed)

#(train_data,valid_data,test_data)=Preprocess_data(train_data, valid_data, test_data, labels)
select_clf = ExtraTreesClassifier(n_estimators=1000, max_depth=5)
print(train_data.shape)
select_clf.fit(train_data, labels)
train_data = select_clf.transform(train_data,threshold='0.5*mean')
valid_data = select_clf.transform(valid_data,threshold='0.5*mean')
test_data = select_clf.transform(test_data,threshold='0.5*mean')
print(np.sort(select_clf.feature_importances_))
print(train_data.shape)

my_mean =np.mean(select_clf.feature_importances_)
print("mean = %f\n" % my_mean)
print(np.where(select_clf.feature_importances_ > 0.5*my_mean))

#exit(1)

# var_names = np.loadtxt('../../../selected_input/sylvine_train.data.csv', dtype=str,delimiter=',')
# print(var_names)
# selected_var_num = var_names.shape[0]
# var_indices=np.zeros(selected_var_num,dtype=int)
# for i in range(selected_var_num):
#     var_indices[i] = int(var_names[i][1:])
# print(var_indices)
#
# (train_data, valid_data, test_data) = Choose_variables(var_indices, train_data, valid_data, test_data)

######################### Make validation/test predictions

n_features=train_data.shape[1]
#gbt_features=int(n_features**0.5)
gbt_features=n_features
gbt_params=GBT_params(n_iterations=14000,depth=6, learning_rate=0.01,subsample_part=0.6,n_max_features=gbt_features,min_samples_split=5, min_samples_leaf=2)
gbt_params.print_params()

make_classification(gbt_params, train_data, labels, valid_data, test_data, 'res/sylvine_valid_001.predict', 'res/sylvine_test_001.predict')
np.savetxt('res/sylvine.seed', np.array([np_seed]),"%d")

print("build ended %d seconds" % (time.time() - start_time))

exit(1)


########################## Make cross validation
gbt_params_begin=GBT_params(n_iterations=10000,depth=7, learning_rate=0.005,subsample_part=0.6,n_max_features=(n_features/2))
gbt_params_mult_factor=GBT_params(n_iterations=1,depth=1, learning_rate=2,subsample_part=1,n_max_features=1)
gbt_params_add_factor=GBT_params(n_iterations=3000,depth=1, learning_rate=0,subsample_part=1,n_max_features=5)
gbt_params_num_iter=GBT_params(n_iterations=4,depth=4, learning_rate=4,subsample_part=1,n_max_features=3)
#gbt_params_num_iter=GBT_params(n_iterations=1,depth=1, learning_rate=1,subsample_part=1,n_max_features=1)

cv_folds=5
(cv_res,cv_times)=make_cross_validation(train_data, labels, cv_folds, gbt_params_begin, gbt_params_mult_factor, gbt_params_add_factor, gbt_params_num_iter)

print("Cross Validation is complete")
print ("cv_res: ", cv_res)
print("cv_times: ", cv_times)

np.savetxt('res/sylvine.crossvalidation_res2', cv_res, '%1.5f')
np.savetxt('res/sylvine.crossvalidation_times2', cv_times, '%1.5f')
exit(1)
