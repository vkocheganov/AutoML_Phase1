__author__ = 'vmkocheg'

from sets import Set
import numpy
import time
from preprocess import GBT_params
from sklearn import ensemble

def make_classification(params, train_data, solution, valid_data, test_data, valid_filename, test_filename):
    classifier = ensemble.GradientBoostingClassifier(n_estimators=params.n_iterations, max_features=int(params.n_max_features),  max_depth = params.depth, learning_rate = params.learning_rate, subsample=params.subsample_part)
    classifier.fit(train_data, solution)

    test_preds = classifier.predict_proba(test_data)[:,1]
    valid_preds = classifier.predict_proba(valid_data)[:,1]

    numpy.savetxt(test_filename, test_preds, '%1.5f')
    numpy.savetxt(valid_filename, valid_preds, '%1.5f')

def make_cross_validation(params, train_data, solution, valid_data, test_data, valid_filename, test_filename):
    classifier = ensemble.GradientBoostingClassifier(n_estimators=params.n_iterations, max_features=int(params.n_max_features),  max_depth = params.depth, learning_rate = params.learning_rate, subsample=params.subsample_part)
    classifier.fit(train_data, solution)

    test_preds = classifier.predict_proba(test_data)[:,1]
    valid_preds = classifier.predict_proba(valid_data)[:,1]

    numpy.savetxt(test_filename, test_preds, '%1.5f')
    numpy.savetxt(valid_filename, valid_preds, '%1.5f')