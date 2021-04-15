import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
import preprocessing

class vizualisation:
    
    def __init__(self, prepro):
        self.cm = 0
        self.prepro = prepro

    def matrice_confusion(self, y_pred, Y_test):
        self.cm = confusion_matrix(Y_test, y_pred)
        self.cm = self.cm.astype('float') / self.cm.sum(axis=1)[:,np.newaxis]
        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(self.cm, interpolation='nearest',cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(self.cm.shape[1]),yticks=np.arange(self.cm.shape[0]),xticklabels=["Arrested","Non_Arrested"],yticklabels=["Arrested","Non_Arrested"],title="Confusion matrix pour Y_test",xlabel="Predicted label",ylabel="True label")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fmt = '.2f'
        thresh = self.cm.max()/2.
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                ax.text(j, i, format(self.cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if self.cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig("Images/Matrice de confusion")
        
        
    def DecisionTree_plot(self, clf):
        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(clf, 
                   feature_names=self.prepro.features_description,  
                   class_names=["Arrested", "Non Arrested"],
                   filled=True)
        plt.savefig("Images/Arbre de décision")
        