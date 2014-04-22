import numpy  as np
from scipy import *
from sklearn import preprocessing
import neural_network as nw
import validation
import util
import scipy.io

data = scipy.io.loadmat('data/train_small.mat')


# data['train'][0][i] = dataset_i
# data['train'][0][0][0][0][0] = data_features
# data['train'][0][0][0][0][0][:,:, i] = ith data_sample
# data['train'][0][0][0][0][1] = data_classes
 
dataset = data['train'][0][1][0][0]
data_classes = dataset[1]
data_features = dataset[0].reshape(28*28,data_classes.size)
data_features = data_features.astype(np.float)
data_features = preprocessing.normalize(data_features)
learning_rate = 10**-5


#network = nw.NeuralNetwork(2, data_features.shape[0], 10)
#network.train(data_features, data_classes, learning_rate)

#accuracy = validation.cross_validate(network, data_features, data_classes, 5, learning_rate, 100)

network = nw.NeuralNetwork(2, 3, 10)
data_feature = array(([1,2,3]))
