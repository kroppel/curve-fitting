import numpy as np
import matplotlib.pyplot as plt
import time

"""Generate ``number_dp`` datapoints (x,y) with x being a vector 
containing ``dim_x`` random elements drawn from the interval [start, end) 
and y being a scalar obtained by evaluating the expression func(x) 
and adding gaussion distributed noise with parameters mean = 0 and 
std = ``noise``
"""
def generate_datapoints(number_dp, dim_x, start=0, end=1, func=(lambda x: x), noise=1):
    rng = np.random.default_rng(int(time.time()))

    X = rng.random((number_dp, dim_x))*(end-start)+start
    Y = np.apply_along_axis(func, 1, X)
    Y = Y.reshape((number_dp,1))
    Y = Y + rng.normal(0, noise, (number_dp, 1))

    return X, Y

"""Perform linear regression using least squares method
"""
def LSRegressionLinear(X, y):
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    W = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y))

    return W

"""Perform quadratic regression using least squares method
"""
def LSRegressionQuadratic(X, y):
    X_mod = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    # compute outer product between data vectors in X
    X_outer = np.einsum('...i,...j->...ij',X,X)
    # choose upper right part of pairwise products of the components of x
    # (including diagonal) to concatenate to vector x
    for dim in np.arange(1, X.shape[1]+1):
        X_mod = np.concatenate((X_outer[:,-dim,-dim:],X_mod), axis=1)
    W = np.dot(np.linalg.inv(np.dot(X_mod.transpose(), X_mod)), np.dot(X_mod.transpose(), y))

    return W

"""Performs the outer product for each row vector in matrix X
with itself and then adds the upper right triangle matrix of the result
including the diagonal flattened from the left onto X itself.
The intent of this function is to add all combinations of X's
(the independent variable's) components to X itself to
make the evaluation of a quadratic function possible by
simply performing a dot product between vectors.
"""
def add_X_outer(X):
    if len(X.shape)==1:
        # compute outer product between data vectors in X
        X_outer = np.einsum('...i,...j->...ij',X,X)
        # choose upper right part of pairwise products of the components of x
        # (including diagonal) to concatenate to vector x
        for dim in np.arange(1, X.shape[0]+1):
            X = np.concatenate((X_outer[-dim,-dim:],X), axis=0)
    else:   
        # compute outer product between data vectors in X
        X_outer = np.einsum('...i,...j->...ij',X,X)
        # choose upper right part of pairwise products of the components of x
        # (including diagonal) to concatenate to vector x
        for dim in np.arange(1, X.shape[1]+1):
            X = np.concatenate((X_outer[:,-dim,-dim:],X), axis=1)
    return X

"""Returns a function that represents a linear function f with the following properties:

        f(X) -> X*weights + bias 

Bias must be passed as last element of the weights array
"""
def model_linear(weights):
    return lambda x: np.dot(x,weights[0:-1])+weights[-1]

"""Returns a function that represents a quadratic function f with the following properties:

        f(X) -> (xi*xj for every combination xi,xj in X)*weights_quadratic + X*weights_linear + bias 
        
Weights and bias must all be part of the weights array following the order [weights_quadratic, weights_linear, bias]
The function add_X_outer enables the evaluation of the function at the input X by simply performing a dot product
"""
def model_quadratic(weights):
    return lambda x: np.dot(add_X_outer(x),weights[0:-1])+weights[-1]