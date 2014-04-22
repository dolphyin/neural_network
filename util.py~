from numpy import *
from scipy import *
from math import *

def sigmoid(x):
    return 1.0/(1 + e**(-x)) 

def sigmoid_deriv(x):
    val = sigmoid(x)
    return val - val**2

def mean_squared_error(hypotheses, truth):
    return 0.5*sum((truth - hypotheses)**2)

def get_mean_squared_delta(hypothesis, truth) :
    return (truth - hypothesis) * (hypothesis**2 - hypothesis)

def cross_entropy_error(hypotheses, truth):
    return - sum(truth*log(hypotheses) + (1 - truth)*log(1 - hypotheses))

def get_cross_entropy_delta(hypotheses, truth):
    return - (truth * (hypothesis - hypothesis**2)/hypothesis + \
            (1 - truth)*(hypothesis**2 - hypothesis)/(1 - hypothesis))
