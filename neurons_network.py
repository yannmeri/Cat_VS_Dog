import sys
from sklearn.datasets import make_blobs
import numpy as np
import numpy.random as random
from sklearn.metrics import accuracy_score
from utilities import *
import math
from tqdm import tqdm

#initializing function for my program

def init_function(n0, n1, n2):#n0 number of input, n1 number of neurons for the first layer and n2 number of neurons for the second layer
    W1 = random.randn(n1, n0)
    b1 = random.randn(n1, 1)
    W2 = random.randn(n2, n1)
    b2 = random.randn(n2, 1)

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return (params)

#mathematic funtion: f(x) = w1*x1 + w2 * x2 + ... + wm * xm and activate the neuron

def activate_function(X, params):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = 1 / (1 + np.exp(-Z1))

    Z2 = np.dot(A1, W2) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    activate = {
        "A1": A1,
        "A2": A2
    }

    return (activate)

#linear_funtion + sigmoid_function produce the result f(x) and log_loss_function behind will compute error about our model.

def log_loss_function(y, activate):
    A2 = activate["A2"]
    epsi = math.exp(-10)
    var = (1 / len(y)) * (-1)
    log_a = np.log(A2 + epsi)
    log_dif_a = np.log(1 - A2 + epsi)
    part_1 = y * log_a
    part_2 = (1 - y) * log_dif_a
    add_part = part_1 + part_2
    log_loss = var * np.sum(add_part)
    return (log_loss)

#we correct values for W and b with these formula: W = W - learning_rate * dW and  b = b - learning_rate * db

def gradients_function(y, activate, X, params):
    W2 = params["W2"]

    A1 = activate["A1"]
    A2 = activate["A2"]
    
    var = 1 / len(y)
    dZ2 = A2 - y
    
    dW1 = var * np.dot((1 - A1), dZ2) * A1.T * W2.T * X
    db1 = var * np.sum((1 - A1) * dZ2 * A1.T * W2)
    dW2 = var * np.dot(dZ2, A1.T)
    db2 = var * np.sum(A2 - y)

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return (gradients)

def update_value_w_b(gradients, params, learning_rate):

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return (params)

def gradients_descent_function(X, y, params, learning_rate):

    for i in tqdm(range(1000)):
        activate = activate_function(X, params)
        log_loss = log_loss_function(y, activate)
        # print(log_loss)
        gradients = gradients_function(y, activate, X, params)
        params = update_value_w_b(gradients, params, learning_rate)

    return (params)

#normalize my datas to obtain datas between [0,1]

def normalize_min_max_method(X_train, X_test):
    X_train_norm = X_train / X_train.max()
    X_test_norm = X_test / X_test.max()
    return (X_train_norm, X_test_norm)

def flatten_train_test(X_train, X_test):
    train_reshape = X_train.reshape(X_train.shape[0], -1)
    test_reshape = X_test.reshape(X_test.shape[0], -1)
    return (train_reshape, test_reshape)

#with the function behind we try to predict the result of y and check the model's accuracy.

def predict_function(X, params):
    activate = activate_function(X, params)
    A2 = activate["A2"]

    return (A2 >= 0.5)

if __name__=="__main__":

    X_train, y_train, X_test, y_test = load_data()
    X_train, X_test = flatten_train_test(X_train, X_test)
    X_train, X_test = normalize_min_max_method(X_train, X_test)
    #print(X_train.shape)
    params = init_function(X_train.shape[0], 2, y_train.shape[0])
    params = gradients_descent_function(X_train, y_train, params, 0.01)
    y_pred = predict_function(X_test, params)
    print(accuracy_score(y_test, y_pred))
