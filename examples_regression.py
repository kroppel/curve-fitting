import matplotlib.pyplot as plt
import numpy as np
from src.regression import generate_datapoints, LSRegressionLinear, LSRegressionQuadratic, model_linear, model_quadratic, add_X_outer

save_to_file = False

"""Plot data points given by (x,y)
"""
def show_datapoints(x, y):
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    axs.spines['bottom'].set_position('zero')
    axs.spines['left'].set_position('zero')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.scatter(x, y)
    if not save_to_file:
        plt.show()
    else:
        plt.savefig("data.png")

"""Plot sets of data points given by (x,y) and
the regression model func with its parameters w
"""
def show_datapoints_and_regression(x, y, func, w):
    # sort datapoints (important for line plot)
    sorted_indices = np.argsort(x, axis=0)
    x = np.take_along_axis(x, sorted_indices, axis=0)
    y = np.take_along_axis(y, sorted_indices, axis=0)
    
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    axs.spines['bottom'].set_position('zero')
    axs.spines['left'].set_position('zero')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.scatter(x, y)
    axs.plot(x, (func(w))(x))
    if not save_to_file:
        plt.show()
    else:
        plt.savefig("data_regression.png")

def linear_regression_univariate_example():
    # Generate linear data
    X, Y = generate_datapoints(number_dp=100, dim_x=1, start=-10, end=10, func=model_linear(np.asarray([2, 10])), noise=0.8)
    show_datapoints(X, Y)
    # Perform linear regression
    weights = LSRegressionLinear(X, Y)
    predictions = model_linear(weights)(X)
    show_datapoints_and_regression(X, Y, model_linear, weights)
    
    mse = np.sum(np.power(predictions-Y, 2))/len(Y)
    print("MSE: "+str(mse))
    print(weights)

def linear_regression_multivariate_example():
    # Generate linear data
    X, Y = generate_datapoints(number_dp=100, dim_x=3, start=-10, end=10, func=model_linear(np.asarray([2, -3, 4, -5])), noise=1)
    # Perform linear regression
    weights = LSRegressionLinear(X, Y)
    predictions = model_linear(weights)(X)

    mse = np.sum(np.power(predictions-Y, 2))/len(Y)
    print("MSE: "+str(mse))
    print(weights)

def quadratic_regression_univariate_example():
    # Generate quadratic data
    X, Y = generate_datapoints(number_dp=100, dim_x=1, start=-10, end=10, func=model_quadratic(np.asarray([0.8, 5, -5])), noise=3)
    show_datapoints(X, Y)
    # Perform quadratic regression
    weights = LSRegressionQuadratic(X, Y)
    predictions = model_quadratic(weights)(X)
    show_datapoints_and_regression(X, Y, model_quadratic, weights)

    mse = np.sum(np.power(predictions-Y, 2))/len(Y)
    print("MSE: "+str(mse))
    print(weights)

def quadratic_regression_multivariate_example():
    # Generate quadratic data (#parameters = sum([1,2,...,dim_x, dim_x+1]))
    X, Y = generate_datapoints(number_dp=100, dim_x=2, start=-10, end=10, func=model_quadratic(np.asarray([0.8, 4, 3, 2, 2, -5])), noise=3)
    # Perform quadratic regression
    weights = LSRegressionQuadratic(X, Y)
    predictions = model_quadratic(weights)(X)

    mse = np.sum(np.power(predictions-Y, 2))/len(Y)
    print("MSE: "+str(mse))
    print(weights)

def main():
    linear_regression_univariate_example()
    linear_regression_multivariate_example()
    quadratic_regression_univariate_example()
    quadratic_regression_multivariate_example()


if __name__ == "__main__":
    main()