__author__ = 'vmkochegvirtual'
#order = christine, jasmine.py, philippine, madeline, sylvine
from sklearn import ensemble, linear_model
from sklearn.cross_validation import KFold
import time

from libs.libscores import *
from libs.data_io import *


def Calc_CV_ERROR(classifier, data, solution, cv_folds):
    cv_scores = np.zeros(cv_folds)

    kf=KFold(len(solution), cv_folds, indices=False)
    cv_iter = 0
    for cv_train, cv_test in kf:
        print("cv iteration %d" % cv_iter)
        classifier.fit(data[cv_train], solution[cv_train])
        cv_test_pred = classifier.predict_proba(data[cv_test])[:,1]
        cv_scores[cv_iter] = bac_metric(cv_test_pred, solution[cv_test], task='binary.classification')
        cv_iter += 1;

    return cv_scores.mean()


np.set_printoptions(suppress=True)

print("start loading")
start_time = time.time()
train_data = np.loadtxt('input/sylvine/sylvine_train.data')
test_data = np.loadtxt('input/sylvine/sylvine_test.data')
valid_data = np.loadtxt('input/sylvine/sylvine_valid.data')
labels = np.loadtxt('input/sylvine/sylvine_train.solution')
print("end loading , %d" % (start_time - time.time()))


clf4 = ensemble.GradientBoostingClassifier(n_estimators=1000,learning_rate=0.03)

Nfolds = 5

#print Calc_CV_ERROR(clf4,train_data, labels, Nfolds)
#exit (1)

# iteration_range = range(0,1)
# learning_range = range(0,1)
# max_depth_range = range(0,2)

#iteration_range = range(0,5)
iteration_range = range(0,4)
learning_range = range(0,6)
#max_depth_range = range(0,4)
max_depth_range = range(0,2)

cv_iterations =len(iteration_range) * len(learning_range) * len(max_depth_range)
cv_res=np.zeros(cv_iterations)
cv_times=np.zeros(cv_iterations)

print cv_res.shape

#base_iterations = 500
base_iterations = 1000
base_learning_rate = 0.001
#base_max_depth=2
base_max_depth=3

cur_iter = 0
_iterations=base_iterations
# for n_iterations in iteration_range:
#     _learning_rate = base_learning_rate
#     for n_learning_rate in learning_range:
#         _max_depth = base_max_depth
#         for n_max_depth in max_depth_range:
#             print("cur iter = %d, _iterations = %d, _learning_rate = %f, _max_depth = %d" % (cur_iter, _iterations, _learning_rate, _max_depth))
#             start_time = time.time()
#             clf4 = ensemble.GradientBoostingClassifier(n_estimators=_iterations,learning_rate=_learning_rate, max_depth=_max_depth)
#             cv_res[cur_iter] = Calc_CV_ERROR(clf4,train_data, labels, Nfolds)
#             print ("CV score = %1.5",cv_res[cur_iter])
#             cv_times[cur_iter]=  time.time() - start_time
#             print ("CV time = %d",cv_times[cur_iter])
#             _max_depth = _max_depth * 2
#             cur_iter +=1
#         _learning_rate = _learning_rate * 3
#     _iterations = _iterations * 2
#
# np.savetxt('CV_results/sylvine/cv_scores',cv_res,'%1.5f')
# np.savetxt('CV_results/sylvine/cv_times',cv_times,'%d')
# exit(1)








clf4 = ensemble.GradientBoostingClassifier(n_estimators=15000,learning_rate=0.005, max_depth=10)

print("model building")
start_time = time.time()
clf4.fit(train_data, labels)
print("build ended %d seconds" % (time.time() - start_time))

print("prediction")
start_time = time.time()
test_preds = clf4.predict_proba(test_data)[:,1]
valid_preds = clf4.predict_proba(valid_data)[:,1]
print("prediction ended %d" % (time.time() - start_time))

np.savetxt('res/sylvine_test_001.predict', test_preds, '%1.5f')
np.savetxt('res/sylvine_valid_001.predict', valid_preds, '%1.5f')