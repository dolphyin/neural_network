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

    # Trains using the back propogation algorithm
    def train(self, train_data, train_classes, learning_rate, num_epochs=10):
        for i in range(num_epochs):
            mini_batches = self.getMiniBatches(train_data, train_classes)
            new_weights = self.weights
            for j in range(len(mini_batches)):
                for data_point in mini_batches[j]:
                    data_feature = data_point[:, data_point.shape(1)-2]
                    data_class = data_point[:, -1]
                    
                    forward_pass(data_feature)
                    backward_pass(data_class, util.get_mean_squared_delta)

                    update_weights(learnig_rate)

    def forward_pass(self, data_feature):
        prev_x = None 
        for i in range(self.num_layers):
            layer = self.layers[i]
            if i == 0:
                layer.set_x(data_point)
                prev_x = data_point
            else:
                curr_x = layer.calculate_x(prev_X)
                layer.set_x(curr_x)
                prev_x = curr_x
    
    def backward_pass(self, data_class, error_delta_method):
        for l in range(self.num_layers-1, 0, -1):
            layer = self.layers[l]
            prev_weights = None
            prev_deltas = None
            if l == self.num_layers-1:
                hypotheses = layer.get_x() 
                hypotheses = [x == max(hypotheses) for x in hypotheses]
                truth = [ i == data_class for i in range(10)] 
                prev_deltas =  error_delta_method(hypotheses, truth)
                prev_weights = layer.weights
                layer.set_deltas(prev_deltas)
            else:
                new_deltas = layer.calculate_deltas(prev_deltas,layer.get_x(), prev_weights) 
                layer.set_deltas(new_deltas)
                prev_deltas = new_deltas
                prev_weights = layer.weights

    def update_weights(self, learning_rate):
        for l in range(self.num_layers):
            layer = self.layers[i]
            if l == 0:
                prev_x = layer.get_x()
                prev_deltas = layer.deltas
            else:
                for i in range(self.weights.shape[1]):
                    self.weights[i,:] -= learning_rate * prev_x[i] * prev_deltas
                prev_deltas = layer.deltas


    # Returns list of randomized batches of data of size
    def getMiniBatches(self ,train_data, train_classes, size=200):
        all_data = hstack((train_data, train_classes))
        all_data = random.shuffle(all_data)
        return vsplit(all_data, train_data.shape(1)/200)
    
class NeuralNetworkLayer(object):

    def __init__(self, layer_num, prev_num_nodes, num_nodes):
        self.layer_num = layer_num
        self.num_nodes = num_nodes
        self.x = zero(num_nodes)
        self.weights = random.rand(prev_num_nodes, num_nodes)

    # Calculates the current layer's X values based on layer's weights
    # and previous layer's X values.
    # @params: 
    def calculate_x(self, prev_x):
        sums = prev_x * self.weights[:,j]
        return [util.sigmoid(s) for s in sums]

    def get_x(self):
        return self.x
    
    def calculate_deltas(self, prev_deltas, curr_x, weights):
        sig_vect = 1 - curr_x**2
        for j in range(len(sig_vect)):
            sig_vect[j] *= sum(prev_deltas * weights[:, j])
        return sig_vect
