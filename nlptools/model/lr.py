import numpy as np

def sigmoid(z):
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    return 1 / (1 + np.exp(-z))

def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    '''
    m = x.shape[0]
    for _ in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = -1 / m * np.sum(np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
        theta = theta - (alpha / m) * np.dot(x.T, h - y)
    J = float(J)
    return J, theta
