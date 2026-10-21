"""Gaussian Process Regression with a Rational Quadratic kernel, as in the paper."""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic, WhiteKernel

NAME = "GPR"


def make():
    kernel = ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(1e-2)
    return GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False, random_state=42)
