import numpy as np

def eculidean_dist(a, b):
    return np.naling.norm(a - b)

def cosine_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
