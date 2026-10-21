"""
ANN-PSO — the paper's best model.

Same NumPy neural network as ann.py, but its weights are found by Particle Swarm
Optimization (pyswarms) instead of back-propagation. A "swarm" of candidate weight
vectors flies through weight-space; each particle is scored by the training MSE its
weights produce, and the swarm converges toward the weights that minimise that error.

Kept to one small hidden layer so the search space (number of weights) stays
tractable for PSO — high-dimensional PSO is slow and unreliable.
"""
import sys
from pathlib import Path

import numpy as np
import pyswarms as ps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ann import NumpyANN

NAME = "ANN-PSO"


class ANNPSO:
    def __init__(self, hidden=(8,), n_particles=40, iters=150, seed=42):
        self.hidden, self.n_particles, self.iters, self.seed = hidden, n_particles, iters, seed
        self.net = None

    def fit(self, X, y):
        np.random.seed(self.seed)
        net = NumpyANN(X.shape[1], hidden=self.hidden, seed=self.seed)
        dim = net.n_params
        y = y.ravel()

        def loss(weights):
            # weights: (n_particles, dim) -> one MSE per particle
            costs = np.empty(len(weights))
            for i, w in enumerate(weights):
                net.set_flat(w)
                pred = net.predict(X)
                costs[i] = np.mean((pred - y) ** 2)
            return costs

        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}   # cognitive, social, inertia
        bounds = (np.full(dim, -3.0), np.full(dim, 3.0))
        opt = ps.single.GlobalBestPSO(self.n_particles, dim, options=options, bounds=bounds)
        _, best = opt.optimize(loss, iters=self.iters, verbose=False)
        net.set_flat(best)
        self.net = net
        return self

    def predict(self, X):
        return self.net.predict(X)


def make():
    return ANNPSO()
