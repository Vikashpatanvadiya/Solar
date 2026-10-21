"""Support Vector Machine regressor with a Gaussian (RBF) kernel, as in the paper."""
from sklearn.svm import SVR

NAME = "SVM"


def make():
    # RBF = the paper's "Gaussian" kernel; C/epsilon tuned for scaled [0,1] data.
    return SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.01)
