"""Baseline: multiple linear regression (sanity check)."""
from sklearn.linear_model import LinearRegression

NAME = "Linear Regression"


def make():
    return LinearRegression()
