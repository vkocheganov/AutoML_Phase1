__author__ = 'vmkocheg'

from sets import Set
import numpy

class GBT_params:
    # n_iterations=1000
    # depth=5
    # learning_rate=0.1
    # subsample_part=0.7
    # n_max_features=-1
    def __init__(self, n_iterations=1000, depth=5, learning_rate=0.1, subsample_part=0.7, n_max_features=-1):
        self.n_iterations=n_iterations
        self.depth=depth
        self.learning_rate=learning_rate
        self.subsample_part=subsample_part
        self.n_max_features=n_max_features
    def print_params(self):
        print ("n_iterations = %d depth=%d learning_rate = %f, subsample_part = %f, n_max_features=%d"% (self.n_iterations, self.depth, self.learning_rate, self.subsample_part, self.n_max_features))

def Preprocess_data(train_data, valid_data, test_data, solution):
    print("train_data shape before preprocessing %d %d"%train_data.shape)

    n_features=train_data.shape[1]
    idxs_to_del=Set()

    for i in range(n_features):
        if(numpy.unique(train_data[:,i]).shape[0] == 1):
            idxs_to_del.add(i)
    idxs_to_del_array=numpy.array(list(idxs_to_del))
    train_data = numpy.delete(train_data, idxs_to_del_array,1)
    valid_data = numpy.delete(valid_data, idxs_to_del_array,1)
    test_data = numpy.delete(test_data, idxs_to_del_array,1)
    print("train_data shape after deleting constants %d %d"%train_data.shape)
    idxs_to_del=Set()
    n_features=train_data.shape[1]
    corr_array=numpy.corrcoef(train_data.transpose())
    corr_array_bool = (numpy.absolute(corr_array) > 0.999).astype(int)
    for i in range(n_features-1):
        idxs_to_del.update(Set(numpy.where(corr_array_bool[i,(i+1):n_features] > 0)[0] + i))
    idxs_to_del_array=numpy.array(list(idxs_to_del))
    train_data = numpy.delete(train_data, idxs_to_del_array,1)
    valid_data = numpy.delete(valid_data, idxs_to_del_array,1)
    test_data = numpy.delete(test_data, idxs_to_del_array,1)
    print("train_data shape after deletin correlated features %d %d"%train_data.shape)
    return (train_data,valid_data,test_data)
