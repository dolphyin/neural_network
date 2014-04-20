from numpy import *
from scipy import *
from util import * 

class NeuralNetwork(object):

    def __init__(self, num_layers, num_features, num_outputs):
        self.num_layers = num_layers
        self.layers = [];
        for i in range(num_layers)-1:
            if i == num_layers -1:
                self.layers[i] = NeuralNetworkLayer(i, num_features, num_outputs)
            else:
                self.layers[i] = NeuralNetworkLayer(i, num_features, num_features) 

    def train(self, train_data, train_classes, learning_rate, num_epochs=10):
        for i in range(num_epochs):
            mini_batches = self.getMiniBatches(train_data, train_classes)
            new_weights = self.weights
            for j in range(len(mini_batches)):
                for data_point in mini_batches[j]:
                    gradient = self.get_gradient(data_point)
                    new_weights -= learning_rate * gradient 
            self.weights = new_weights
        return None

    def classify(self, test_data): 
        return None

    # Returns list of randomized batches of data of size
    def getMiniBatches(self ,train_data, train_classes, size=200):
        all_data = hstack((train_data, train_classes))
        all_data = random.shuffle(all_data)
        return vsplit(all_data, train_data.shape(1)/200)
    
    def getGradient(self, data_point):
        return None

class NeuralNetworkLayer(object):

    def __init__(self, layer_num, prev_num_nodes, num_nodes):
        self.layer_num = layer_num
        self.num_nodes = num_nodes
        self.x_ij = zero(num_nodes)
        self.weights = random.rand(prev_num_nodes, num_nodes)

    # Calculates the current layer's X values based on layer's weights
    # and previous layer's X values.
    # @params: 
    def calculate_x(self, prev_x):

