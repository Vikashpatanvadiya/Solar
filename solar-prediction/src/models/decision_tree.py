"""Decision Tree regressor (CART), as in the paper."""
from sklearn.tree import DecisionTreeRegressor

NAME = "Decision Tree"


def make():
    # min_samples_leaf keeps the tree from memorising single days (overfit guard)
    return DecisionTreeRegressor(min_samples_leaf=5, random_state=42)
