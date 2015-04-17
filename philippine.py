__author__ = 'vmkochegvirtual'

from sklearn import ensemble, linear_model
from sklearn.cross_validation import KFold
import time

from libs.libscores import *
from libs.data_io import *


def Calc_CV_ERROR(classifier, data, solution, cv_folds):
    print("\n")
    cv_scores = np.zeros(cv_folds)

    kf=KFold(len(solution), cv_folds, indices=True)
    cv_iter = 0
    for cv_train, cv_test in kf:
        print("cv iteration %d" % cv_iter)

    #     clf2.fit(data[cv_train], solution[cv_train])
    #     clf3.fit(data[cv_train], solution[cv_train])
    #     cv_test_pred = (clf2.predict_proba(data[cv_test])[:,1] + clf3.predict_proba(data[cv_test])[:,1])/2

        #    clf1.fit(train_data[cv_train], labels[cv_train])
        #     clf2.fit(data[cv_train], solution[cv_train])
        #     clf3.fit(data[cv_train], solution[cv_train])
        #    cv_test_pred = (clf1.predict_proba(train_data[cv_test])[:,1] + clf2.predict_proba(train_data[cv_test])[:,1] + clf3.predict_proba(train_data[cv_test])[:,1])/3
        classifier.fit(data[cv_train], solution[cv_train])
        cv_test_pred = classifier.predict_proba(data[cv_test])[:,1]
        cv_scores[cv_iter] = bac_metric(cv_test_pred, solution[cv_test], task='binary.classification')
        cv_iter += 1;

    return cv_scores.mean()


np.set_printoptions(suppress=True)

print("start loading")
start_time = time.time()
train_data = np.loadtxt('input/philippine/philippine_train.data')
test_data = np.loadtxt('input/philippine/philippine_test.data')
valid_data = np.loadtxt('input/philippine/philippine_valid.data')
labels = np.loadtxt('input/philippine/philippine_train.solution')
print("end loading , %d" % (start_time - time.time()))

forest_feat =(train_data.shape[1])**0.5
#clf2 = linear_model.LogisticRegression(C=100.)
#clf3 = linear_model.LogisticRegression(C=1.)
clf4 = ensemble.GradientBoostingClassifier(n_estimators=15000, max_features=int(forest_feat),learning_rate=0.003, max_depth=10)
#exit(1)
#clf5 = ensemble.RandomForestClassifier(n_estimators=1000, max_depth=4)
print ensemble
print train_data.shape[1]
Nfolds = 5

#print Calc_CV_ERROR(clf4,train_data, labels, Nfolds)
#exit (1)













print("model building")
start_time = time.time()
clf4.fit(train_data, labels)
print("build ended %d seconds" % (time.time() - start_time))

print("prediction")
start_time = time.time()
test_preds = clf4.predict_proba(test_data)[:,1]
valid_preds = clf4.predict_proba(valid_data)[:,1]
print("prediction ended %d" % (time.time() - start_time))

np.savetxt('res/philippine_test_001.predict', test_preds, '%1.5f')
np.savetxt('res/philippine_valid_001.predict', valid_preds, '%1.5f')
