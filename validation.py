from numpy import *
from scipy import *
from sklearn import cross_validation
import copy

def cross_validate(classifier, data_features, data_classes, k, learning_rate, num_epochs):
    kfolds = cross_validation.KFold(data_classes.size, n_folds=k)
    accuracy = 0.0
    #c = copy.deepcopy(classifier)
    c = classifier
    for train_index, test_index in kfolds:
        train_data = data_features[:, train_index]
        train_classes = data_classes[train_index]

        test_data = data_features[:, test_index]
        test_classes = data_classes[test_index]

        c.train(train_data, train_classes, learning_rate, num_epochs)

        # TODO Check if returned hyp are n_out x1 or num_samples*n_out*1
        hypotheses = c.classify(test_data)
        isEqual = hypotheses == test_classes[:,0]
        num_right = sum([1 if guess else 0 for guess in isEqual])
        print(float(num_right)/isEqual.shape[0])
        accuracy += float(num_right)/isEqual.shape[0]
    return accuracy/k
