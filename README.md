# Linear and Quadratic Regression
## Module providing functions to perform linear and quadratic regression on multivariate data.

### src/regression.py:

Provides the functions for regression as well as data generation

__generate_datapoints(number_dp, dim_x, start=0, end=1, func=(lambda x: x), noise=1):__

Generate ``number_dp`` datapoints (x,y) with x being a vector containing ``dim_x`` random elements drawn from the interval [start, end) and y being a scalar obtained by evaluating the expression func(x) and adding gaussion distributed noise with parameters mean = 0 and std = ``noise``

__LSRegressionLinear(X, y):__

Performs linear regression

__LSRegressionQuadratic(X, y):__

Performs quadratic regression

### examples_regression.py:

Contains examples in which data is generated using a linear / quadratic model and then regression is performed to retrieve the original function parameters.

##### linear regression univariate

First we create a dataset containing 100 samples from the interval [-10, 10] that follow the linear model f(x)=2*x+10

["linear_data"](images/data_linear.png)



