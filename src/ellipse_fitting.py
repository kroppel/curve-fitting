import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.linalg.lapack import zggev

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
    plt.show()

"""Plot data points given by (x,y)
"""
def show_datapoints_with_ellipse1(x, y, e):
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    axs.spines['bottom'].set_position('zero')
    axs.spines['left'].set_position('zero')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.scatter(x, y)

    # Plot the least squares ellipse
    x_coord = np.linspace(x[0]-100,x[0]+100,1000)
    y_coord = np.linspace(y[0]-100,y[0]+100,1000)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = e[0] * X_coord ** 2 + e[1] * X_coord * Y_coord + e[2] * Y_coord**2 + e[3] * X_coord + e[4] * Y_coord
    axs.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)

    plt.show()

"""Plot data points given by (x,y)
"""
def show_datapoints_with_ellipse3(x, y, e):
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    axs.spines['bottom'].set_position('zero')
    axs.spines['left'].set_position('zero')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.scatter(x, y)

    # Plot the least squares ellipse
    x_coord = np.linspace(x[0]-100,x[0]+100,1000)
    y_coord = np.linspace(y[0]-100,y[0]+100,1000)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = e[0] * X_coord ** 2 + e[1] * X_coord * Y_coord + e[2] * Y_coord**2 + e[3] * X_coord + e[4] * Y_coord + e[5]
    axs.contour(X_coord, Y_coord, Z_coord, levels=[0], colors=('r'), linewidths=2)

    plt.show()

def fit_ellipse1(points):
    X = np.asarray(points[:,0:1])
    Y = np.asarray(points[:,1:])
    A = np.hstack([X**2, X*Y, Y**2, X, Y])

    # solve least squares 
    e = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(np.ones_like(X)))
    show_datapoints_with_ellipse1(X,Y,e)
    
    return(e)

def fit_ellipse3(points):
    X = np.asarray(points[:,0:1])
    Y = np.asarray(points[:,1:])
    A = np.hstack([X**2, X*Y, Y**2, X, Y, np.ones_like(X)])
    SM = A.T.dot(A)
    CM = np.zeros((6,6))
    CM[0,2] = -2
    CM[2,0] = -2
    CM[1,1] = 1

    AA, BB, Q, Z = scipy.linalg.qz(SM, CM)
    vals = np.diag(AA)[np.diag(BB)!=0]/np.diag(BB)[np.diag(BB)!=0]
    vals_all = np.zeros((np.min([AA.shape[0], BB.shape[0]])))
    vals_all[np.diag(BB)!=0] = vals
    _, _, V, W, _, _ = zggev(SM, CM)

    index = np.nonzero(vals_all<0)[0]
    e = (V[:,index]*4).real
    print(e)
    show_datapoints_with_ellipse3(X,Y,e)

    return(e)

def main():
    N = 100
    DIM = 2

    np.random.seed(2)

    # Generate random points on the unit circle by sampling uniform angles
    theta = np.random.uniform(0, 2*np.pi, (N,1))
    eps_noise = 0.1 * np.random.normal(size=[N,1])
    circle = np.hstack([np.cos(theta), np.sin(theta)])

    # Stretch and rotate circle to an ellipse with random linear tranformation
    B = np.random.randint(-3, 3, (DIM, DIM))
    noisy_ellipse = circle.dot(B) + eps_noise
    # add translation
    t = [5,5]
    noisy_ellipse = noisy_ellipse + t
    show_datapoints(noisy_ellipse[:,0], noisy_ellipse[:,1])

    fit_ellipse3(noisy_ellipse)

if __name__ == "__main__":
    main()








