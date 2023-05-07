# -*- coding: utf-8 -*-
import numpy as np
 
def tanh(x):
    return np.tanh(x)
 
def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)
 
def logistic(x):
    return 1 / (1 + np.exp(-x))
 
def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    d = np.array(x, copy=True)
    d[x < 0] = 0 # 元素为负的导数为 0
    d[x >= 0] = 1 # 元素为正的导数为 1
    return d