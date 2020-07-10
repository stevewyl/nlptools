import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()

def frobenius_norm(A):
    return np.sqrt(np.sum(np.square(A)))
