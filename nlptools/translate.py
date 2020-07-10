import os

import numpy as np
from gensim.models import KeyedVectors

from nlptools.metrics.distance import cosine_dist

class Translator:
    def __init__(self, source_vec_fn, target_vec_fn):
        self.source_vec_fn = source_vec_fn
        self.target_vec_fn = self.target_vec_fn
        self.source_embeddings = self.load_vec(self.source_vec_fn)
        self.target_embeddings = self.load_vec(self.target_vec_fn)
        print("source and target embeddings loaded")

        self.R = None

    def load_vec(self, vec_fn, binary=False):
        if vec_fn.endswith("bin"):
            binary = True
        return  KeyedVectors.load_word2vec_format(vec_fn, binary=binary)

    def load_data(self, data_fn):
        data = {}
        for line in open(data_fn, "r"):
            source_word, target_word = line.strip().split()
            data[source_word] = target_word
        return data

    # TODO: fix source_embeddings_subset
    def get_matrices(self, ds):
        X, Y = [], []
        source_set = set(self.source_embeddings.keys())
        target_set = set(self.target_embeddings.keys())
        for source_word, target_word in ds.items():
            if source_word in source_set and target_word in target_set:
                X.append(self.source_embeddings[source_word])
                Y.append(self.target_embeddings[target_word])
        X, Y = np.vstack(X), np.vstack(Y)
        return X, Y
    
    def compute_loss(self, X, Y, R):
        m = X.shape[0]
        diff = np.dot(X, R) - Y
        diff = np.sum(np.square(diff))
        loss = -1 / m * diff
        return loss
    
    def compute_gradient(self, X, Y, R):
        m = X.shape[0]
        gradient = np.dot(X.T, np.dot(X, R) - Y) * 2 / m
        return gradient
    
    def align_embeddings(self, X, Y, train_steps=100, learning_rate=0.0003):
        np.random.seed(129)

        R = np.random.rand(X.shape[1], X.shape[1])

        for i in range(train_steps):
            if i % 25 == 0:
                print(f"loss at iteration {i} is: {self.compute_loss(X, Y, R):.4f}")
            gradient = self.compute_gradient(X, Y, R)
            R -= learning_rate * gradient
        return R

    def nearest_neighbor(self, v, candidates, k=1):
        sim_l = []
        for row in candidates:
            cos_sim = 1 - cosine_dist(row, v)
            sim_l.append(cos_sim)
        sorted_ids = np.argsort(sim_l)
        k_idx = sorted_ids[-k:]
        return k_idx

    def train(self, train_fn, train_steps=400, learning_rate=0.8):
        ds = self.load_data(train_fn)
        X_train, Y_train = self.get_matrices(ds)
        self.R = self.align_embeddings(
            X_train, Y_train, train_steps=train_steps, learning_rate=learning_rate)
    
    def test(self, test_fn):
        ds = self.load_data(test_fn)
        X_test, Y_test = self.get_matrices(ds)
        pred = np.dot(X_test, self.R)
        num_correct = 0
        for i in range(len(pred)):
            pred_idx = self.nearest_neighbor(pred[i], Y_test)
            if pred_idx == i:
                num_correct += 1
        acc = num_correct / len(X_test)
        return acc
