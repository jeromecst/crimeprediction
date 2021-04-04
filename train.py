import numpy as np
import sklearn

from sklearn.naive_bayes import GaussianNB
from sklearn import tree

class train:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.X_train = 0
        self.X_test = 0
        self.Y_train = 0
        self.Y_test = 0

    def load_data(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = X.shape[0]
        self.D = X.shape[1]

    def traintestsplit(self, test_ratio=.3):
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
                sklearn.model_selection.train_test_split(self.X, self.Y, test_size=test_ratio)

    def fit_GaussNB(self):
        '''
        Entraînement de notre modèle
        '''
        gnb = GaussianNB()
        gnb.fit(self.X_train, self.Y_train)
        y_pred = gnb.predict(self.X_test)
        #print("Ratio de bien placé pour le Bayessien naif = ", ((self.Y_test == y_pred).sum() / self.X_test.shape[0]))
        return ((self.Y_test == y_pred).sum() / self.X_test.shape[0])

        
        
    def fit_DecisionTree(self):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(self.X_train, self.Y_train)
        y_pred = clf.predict(self.X_test)
        #print("Ratio de bien placé pour le Decision tree = ", ((self.Y_test == y_pred).sum() / self.X_test.shape[0]))
        return ((self.Y_test == y_pred).sum() / self.X_test.shape[0])
