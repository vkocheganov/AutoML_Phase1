__author__ = 'vmkochegvirtual'

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble, linear_model
from sklearn.cross_validation import KFold
import time
import numpy
from libs.libscores import *
from libs.data_io import *
import numpy
from sets import Set
from time import gmtime, strftime

from calc_cv_scores import Calc_CV_ERROR,make_cross_validation
from preprocess import Preprocess_data,GBT_params,Choose_variables
from utils import make_classification,make_classification_random_forest

print(strftime("%Y-%m-%d %H:%M:%S"))

np.set_printoptions(suppress=True)

start_time = time.time()
train_data = np.loadtxt('input/christine/christine_train.data')
test_data = np.loadtxt('input/christine/christine_test.data')
valid_data = np.loadtxt('input/christine/christine_valid.data')
labels = np.loadtxt('input/christine/christine_train.solution')
print("Loading data is complete, %d" % (start_time - time.time()))

start_time = time.time()
np_seed = int(time.time())
np.random.seed(np_seed)
print ("np seed = " , np_seed)

#(train_data,valid_data,test_data)=Preprocess_data(train_data, valid_data, test_data, labels)

# Choose Ideal preselected features

select_clf = ExtraTreesClassifier(n_estimators=2000,max_depth=3,min_samples_split=20, min_samples_leaf=5)

print(train_data.shape)
select_clf.fit(train_data, labels)
my_mean =np.percentile(select_clf.feature_importances_,20)
train_data = select_clf.transform(train_data,threshold=my_mean)
valid_data = select_clf.transform(valid_data,threshold=my_mean)
test_data = select_clf.transform(test_data,threshold=my_mean)
print(np.sort(select_clf.feature_importances_))
print(train_data.shape)
#exit(1)
# var_names = np.loadtxt('../../../selected_input/christine_train.data.csv', dtype=str,delimiter=',')
# print(var_names)
# selected_var_num = var_names.shape[0]-1
# var_indices=np.zeros(selected_var_num,dtype=int)
# for i in range(selected_var_num):
#     var_indices[i] = int(var_names[i][1:])
# print(var_indices)
#
# (train_data, valid_data, test_data) = Choose_variables(var_indices, train_data, valid_data, test_data)

# pca = PCA(n_components=50)
# pca.fit(train_data)
# print("explained variance")
# print(pca.explained_variance_ratio_)
# print(np.sum(pca.explained_variance_ratio_))
#
# train_data = pca.transform(train_data)
# valid_data = pca.transform(valid_data)
# test_data = pca.transform(test_data)

#exit(1)

######################### Make validation/test predictions
n_features=train_data.shape[1]
#gbt_features=n_features
gbt_features=int(n_features**0.5)
gbt_params=GBT_params(n_iterations=15000,depth=7, learning_rate=0.01,subsample_part=0.6,n_max_features=gbt_features,min_samples_split=5, min_samples_leaf=2)
gbt_params.print_params()

make_classification(gbt_params, train_data, labels, valid_data, test_data, 'res/christine_valid_001.predict', 'res/christine_test_001.predict')
# forest_params=GBT_params(n_iterations=15000,depth=10, learning_rate=0.01,subsample_part=0.6,n_max_features=gbt_features,min_samples_split=10, min_samples_leaf=4)
# make_classification_random_forest(gbt_params, train_data, labels, valid_data, test_data, 'res/christine_valid_001.predict', 'res/christine_test_001.predict')
print("build ended %d seconds" % (time.time() - start_time))
np.savetxt('res/christine.seed', np.array([np_seed]),"%d")

exit(1)

########################## Make cross validation
gbt_params_begin=GBT_params(n_iterations=2000,depth=5, learning_rate=0.005,subsample_part=0.7,n_max_features=100)
gbt_params_mult_factor=GBT_params(n_iterations=1,depth=1, learning_rate=2,subsample_part=1,n_max_features=3)
gbt_params_add_factor=GBT_params(n_iterations=1000,depth=2, learning_rate=0,subsample_part=1,n_max_features=0)
gbt_params_num_iter=GBT_params(n_iterations=4,depth=3, learning_rate=3,subsample_part=1,n_max_features=3)
#gbt_params_num_iter=GBT_params(n_iterations=1,depth=1, learning_rate=1,subsample_part=1,n_max_features=1)

cv_folds=5
(cv_res,cv_times)=make_cross_validation(train_data, labels, cv_folds, gbt_params_begin, gbt_params_mult_factor, gbt_params_add_factor, gbt_params_num_iter)

print("Cross Validation is complete")
print ("cv_res: ", cv_res)
print("cv_times: ", cv_times)

np.savetxt('res/christine.crossvalidation_res1', cv_res, '%1.5f')
np.savetxt('res/christine.crossvalidation_times1', cv_times, '%1.5f')
exit(1)

