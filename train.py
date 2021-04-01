import numpy as np
import sklearn

class train:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = X.shape(0)
        self.D = X.shape(1)
        self.X_train, self.X_test, self.Y_train, self.Y_test

    def load_data(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = X.shape(0)
        self.D = X.shape(1)

    def traintestsplit(self, test_ratio=.3):
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
                sklearn.model_selection.train_test_split(X, Y, test_size=test_ratio)

