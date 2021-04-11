import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

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
        self.clf = 0

    def load_data(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.clf = 0

    def traintestsplit(self, test_ratio=.3):
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
                sklearn.model_selection.train_test_split(self.X, self.Y, test_size=test_ratio)

    def fit_GaussNB(self, display=False):
        '''
        Entraînement de notre modèle
        '''
        gnb = GaussianNB()
        gnb.fit(self.X_train, self.Y_train)
        if(display):
            print("Score GaussNB : ", gnb.score(self.X_test, self.Y_test))
        return gnb.score(self.X_test, self.Y_test)

        
    def fit_DecisionTree(self, display=False, min_samples_split = 130, min_samples_leaf = 60, max_features = None, min_impurity_decrease = 0.0):
        clf = tree.DecisionTreeClassifier(min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, max_features = max_features, min_impurity_decrease = min_impurity_decrease)
        clf = clf.fit(self.X_train, self.Y_train)
        if(display):
            print("Score DecisionTree : ", clf.score(self.X_test, self.Y_test))
        self.clf = clf
        return clf.score(self.X_test, self.Y_test)

    def predict_DecisionTree(self):
        return self.clf.predict(self.X_test), self.Y_test
    
    def model_DecisionTree(self):
        return self.clf
    
    def DecisionTree_feature_importances(self, feature_description):
        fig, ax = plt.subplots(1, 1, figsize=(20,10))
        n = feature_description.size
        b = ax.bar(range(n), self.clf.feature_importances_)
        ax.bar_label(b, labels=feature_description)
        ax.set_xlabel("feature description")
        ax.set_ylabel("feature importance")
        plt.savefig("feature_importance")


    def fit_RandomForestClassifier(self, display=False, min_samples_split = 130, min_samples_leaf = 60, max_features = None, min_impurity_decrease = 0.0):
        rfc = RandomForestClassifier(n_jobs = 4, n_estimators = 100, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, max_features = max_features, min_impurity_decrease = min_impurity_decrease)
        rfc  = rfc.fit(self.X_train, self.Y_train)
        if(display):
            print("Score RandomForestClassifier : ", rfc.score(self.X_test, self.Y_test))
        return rfc.score(self.X_test, self.Y_test)
