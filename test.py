import sys
from sklearn.datasets import make_blobs
import numpy as np
import numpy.random as random
from sklearn.metrics import accuracy_score
from utilities import *
import math
from tqdm import tqdm

#initializing function for my program

def init_function(X):
    W = random.randn(X.shape[1], 1)
    b = random.randn(1)

    return (W, b)

#mathematic funtion: f(x) = w1*x1 + w2 * x2 + ... + wm * xm

def linear_function(X, W, b):
    Z = np.dot(X, W) + b
    return (Z)

#a funtion that activate the neuron

def sigmoid_function(Z):
    denum = 1 + np.exp(-Z)
    A = 1 / denum
    return (A)


#linear_funtion + sigmoid_function produce the result f(x) and log_loss_function behind will compute error about our model.

def log_loss_function(y, A):
    add_var = 0
    epsi = math.exp(-10)
    var = (1 / len(y)) * (-1)
    log_a = np.log(A + epsi)
    log_dif_a = np.log(1 - A + epsi)
    part_1 = y * log_a
    part_2 = (1 - y) * log_dif_a
    add_part = part_1 + part_2
    log_loss = var * np.sum(add_part)
    return (log_loss)

#we correct values for W and b with these formula: W = W - learning_rate * dW and  b = b - learning_rate * db

def gradients_function(y, A, X):
    var = (1 / len(y)) * (-1)
    dW = var * np.dot(X.T, (y - A))
    db = var * np.sum(y - A)
    return (dW, db)

def update_value_w_b(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)

def gradients_descent_function(X, y, W, b, learning_rate):
    for i in tqdm(range(1000)):
        Z = linear_function(X, W, b)
        A = sigmoid_function(Z)
        log_loss = log_loss_function(y, A)
        # print(log_loss)
        dW, db = gradients_function(y, A, X)
        W, b = update_value_w_b(dW, db, W, b, 0.1)
    return (W, b)

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

def predict_function(X, W, b):
    new_Z = linear_function(X, W, b)
    new_A = sigmoid_function(new_Z)
    return (new_A >= 0.5)

if __name__=="__main__":
    # X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    # y = y.reshape(X.shape[0],1)
    # W = random.randn(X.shape[1], 1)
    # b = random.randn(1)
    # print(X.shape)
    # print(b.shape)
    # print(y_train.shape)
    # print(W.shape)
    # print(new_W, new_b)
    # print(y)
    # print(y_pred)

    X_train, y_train, X_test, y_test = load_data()
    print(X_train)
    X_train, X_test = flatten_train_test(X_train, X_test)
    print(X_train)
    X_train, X_test = normalize_min_max_method(X_train, X_test)
    W, b = init_function(X_train)
    new_W, new_b = gradients_descent_function(X_train, y_train, W, b, 0.01)
    y_pred = predict_function(X_test, new_W, new_b)
    print(accuracy_score(y_test, y_pred))


