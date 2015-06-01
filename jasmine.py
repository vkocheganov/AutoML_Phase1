__author__ = 'vmkochegvirtual'

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble, linear_model
from sklearn.cross_validation import KFold
import time
from calc_cv_scores import Calc_CV_ERROR
from libs.libscores import *
from libs.data_io import *

from time import gmtime, strftime
from calc_cv_scores import Calc_CV_ERROR,make_cross_validation
from preprocess import Preprocess_data,GBT_params,Choose_variables
from utils import make_classification
print(strftime("%Y-%m-%d %H:%M:%S"))

#-------------------------------
print("start loading")
start_time = time.time()
train_data = np.loadtxt('input/jasmine/jasmine_train.data')
test_data = np.loadtxt('input/jasmine/jasmine_test.data')
valid_data = np.loadtxt('input/jasmine/jasmine_valid.data')
labels = np.loadtxt('input/jasmine/jasmine_train.solution')
print("end loading , %d seconds" % (start_time - time.time()))

start_time = time.time()
np_seed = int(time.time())
np.random.seed(np_seed)
print ("np seed = " , np_seed)

# (train_data,valid_data,test_data)=Preprocess_data(train_data, valid_data, test_data, labels)
# exit(1)
select_clf = ExtraTreesClassifier(n_estimators=1000,max_depth=4)
print(train_data.shape)
select_clf.fit(train_data, labels)
my_mean =np.percentile(select_clf.feature_importances_,40)

# train_data = select_clf.transform(train_data,threshold='median')
# valid_data = select_clf.transform(valid_data,threshold='median')
# test_data = select_clf.transform(test_data,threshold='median')
# train_data = select_clf.transform(train_data,threshold='mean')
# valid_data = select_clf.transform(valid_data,threshold='mean')
# test_data = select_clf.transform(test_data,threshold='mean')

# train_data = select_clf.transform(train_data,threshold=0.0001)
# valid_data = select_clf.transform(valid_data,threshold=0.0001)
# test_data = select_clf.transform(test_data,threshold=0.0001)

train_data = select_clf.transform(train_data,threshold=my_mean)
valid_data = select_clf.transform(valid_data,threshold=my_mean)
test_data = select_clf.transform(test_data,threshold=my_mean)

my_mean =np.median(select_clf.feature_importances_)
print("mean = %f\n" % my_mean)
print(np.where(select_clf.feature_importances_ > my_mean))
print(np.sort(select_clf.feature_importances_))
print(train_data.shape)
#exit(1)

n_features=train_data.shape[1]
#gbt_features=int(n_features**0.5)
# gbt_features=n_features
# gbt_params=GBT_params(n_iterations=5000,depth=5, learning_rate=0.01,subsample_part=0.7,n_max_features=gbt_features)
# params =gbt_params
# start_time = time.time()
# params.print_params()
# clf = ensemble.GradientBoostingClassifier(n_estimators=params.n_iterations,learning_rate=params.learning_rate, max_depth=params.depth, subsample=params.subsample_part, max_features=int(params.n_max_features))
# cv_folds = 5
# cv_score = Calc_CV_ERROR(clf,train_data, labels, cv_folds)
# print ("CV score = %1.5",cv_score)
#
# exit(1)

# var_names = np.loadtxt('../../../selected_input/jasmine_train.data.csv', dtype=str,delimiter=',')
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
gbt_params=GBT_params(n_iterations=14000,depth=6, learning_rate=0.012,subsample_part=0.6,n_max_features=gbt_features,min_samples_split=5, min_samples_leaf=2)
gbt_params.print_params()

make_classification(gbt_params, train_data, labels, valid_data, test_data, 'res/jasmine_valid_001.predict', 'res/jasmine_test_001.predict')
print("build ended %d seconds" % (time.time() - start_time))
np.savetxt('res/jasmine.seed', np.array([np_seed]),"%d")

exit(1)
#

########################## Make cross validation
# gbt_params_begin=GBT_params(n_iterations=3000,depth=4, learning_rate=0.007,subsample_part=0.7,n_max_features=n_features)
# gbt_params_mult_factor=GBT_params(n_iterations=1,depth=1, learning_rate=1,subsample_part=1,n_max_features=1)
# gbt_params_add_factor=GBT_params(n_iterations=2000,depth=1, learning_rate=0.003,subsample_part=0,n_max_features=0)
# gbt_params_num_iter=GBT_params(n_iterations=3,depth=3, learning_rate=3,subsample_part=1,n_max_features=1)


# gbt_params_begin=GBT_params(n_iterations=10000,depth=5, learning_rate=0.01,subsample_part=0.8,n_max_features=(n_features**0.5))
# gbt_params_mult_factor=GBT_params(n_iterations=1,depth=1, learning_rate=2,subsample_part=1,n_max_features=1)
# gbt_params_add_factor=GBT_params(n_iterations=3000,depth=2, learning_rate=0,subsample_part=1,n_max_features=0)
# gbt_params_num_iter=GBT_params(n_iterations=3,depth=3, learning_rate=3,subsample_part=1,n_max_features=1)
#gbt_params_num_iter=GBT_params(n_iterations=1,depth=1, learning_rate=1,subsample_part=1,n_max_features=1)

gbt_params_begin=GBT_params(n_iterations=8000,depth=6, learning_rate=0.007,subsample_part=0.7,n_max_features=n_features)
gbt_params_mult_factor=GBT_params(n_iterations=1,depth=1, learning_rate=1,subsample_part=1,n_max_features=1)
gbt_params_add_factor=GBT_params(n_iterations=2000,depth=1, learning_rate=0.003,subsample_part=0,n_max_features=0)
gbt_params_num_iter=GBT_params(n_iterations=2,depth=2, learning_rate=2,subsample_part=1,n_max_features=1)

cv_folds=5
(cv_res,cv_times)=make_cross_validation(train_data, labels, cv_folds, gbt_params_begin, gbt_params_mult_factor, gbt_params_add_factor, gbt_params_num_iter)

print("Cross Validation is complete")
print ("cv_res: ", cv_res)
print("cv_times: ", cv_times)

np.savetxt('res/jasmine.crossvalidation_res1', cv_res, '%1.5f')
np.savetxt('res/jasmine.crossvalidation_times1', cv_times, '%1.5f')
exit(1)

