"""
Artificial Neural Network — pure NumPy feed-forward back-propagation network.

Built to match the reference paper (Kumar et al., 2026), 3-layer architecture from
its Fig. 7(a): Input layer -> Hidden layer(s) -> Output layer, sigmoid neurons.
Implemented in NumPy (not Keras/TensorFlow, which doesn't install on this Mac's Python),
using the paper's own equations:

  Forward pass (paper Eq. 4):   y_hat = f(W . X + b)
      - each neuron: z = (weights . inputs) + bias
      - activation f = sigmoid:  f(z) = 1 / (1 + e^-z)          -> output in [0,1]
  Module temperature feeding the inputs uses the paper's NOCT model
      (see preprocess.py):      T_module = T_ambient + (G/800)(NOCT - 20)
  Training = error back-propagation (gradient descent + momentum): weights are nudged
      down the error gradient until the mean-squared error stops falling.

All inputs/outputs are MinMax-scaled to [0,1] by dataset.py (the paper's normalisation
step), so the sigmoid output range matches the target. ANN-PSO reuses this exact network
and forward pass, only swapping back-propagation for Particle Swarm Optimization.
"""
import numpy as np

NAME = "ANN"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class NumpyANN:
    def __init__(self, n_in, hidden=(10, 10), seed=42):
        self.sizes = [n_in, *hidden, 1]
        rng = np.random.default_rng(seed)
        self.W = [rng.uniform(-0.5, 0.5, (self.sizes[i + 1], self.sizes[i])) for i in range(len(self.sizes) - 1)]
        self.b = [rng.uniform(-0.5, 0.5, self.sizes[i + 1]) for i in range(len(self.sizes) - 1)]

    # -- weight packing, used by ANN-PSO ------------------------------------
    @property
    def n_params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

    def set_flat(self, vec):
        i = 0
        for k in range(len(self.W)):
            n = self.W[k].size
            self.W[k] = vec[i:i + n].reshape(self.W[k].shape); i += n
        for k in range(len(self.b)):
            n = self.b[k].size
            self.b[k] = vec[i:i + n]; i += n

    # -- forward / predict ---------------------------------------------------
    def _forward(self, X):
        acts = [X]
        for W, b in zip(self.W, self.b):
            X = sigmoid(X @ W.T + b)
            acts.append(X)
        return acts

    def predict(self, X):
        return self._forward(X)[-1].ravel()

    # -- training (back-propagation + momentum) ------------------------------
    def fit(self, X, y, epochs=2000, lr=0.3, momentum=0.5, seed=42):
        y = y.reshape(-1, 1)
        vW = [np.zeros_like(w) for w in self.W]
        vb = [np.zeros_like(b) for b in self.b]
        for _ in range(epochs):
            acts = self._forward(X)
            out = acts[-1]
            delta = (out - y) * out * (1 - out)
            for l in range(len(self.W) - 1, -1, -1):
                gW = delta.T @ acts[l]
                gb = delta.sum(axis=0)
                if l > 0:
                    delta = (delta @ self.W[l]) * acts[l] * (1 - acts[l])
                vW[l] = momentum * vW[l] - lr * gW / len(X)
                vb[l] = momentum * vb[l] - lr * gb / len(X)
                self.W[l] += vW[l]
                self.b[l] += vb[l]
        return self


def make():
    # 2 hidden layers of 10 sigmoid neurons each, as in the paper's ANN.
    return _SklearnLikeANN(hidden=(10, 10), epochs=3000, lr=0.5)


class _SklearnLikeANN:
    """Thin fit/predict wrapper so run.py treats every model the same way."""
    def __init__(self, hidden, epochs, lr):
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        self.net = None

    def fit(self, X, y):
        self.net = NumpyANN(X.shape[1], hidden=self.hidden).fit(X, y, epochs=self.epochs, lr=self.lr)
        return self

    def predict(self, X):
        return self.net.predict(X)
