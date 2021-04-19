import numpy as np
import numpy.random as rd
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
import preprocessing

class vizualisation:
    
    def __init__(self, prepro):
        self.cm = 0
        self.prepro = prepro

    def matrice_confusion(self, title, y_pred, Y_test):
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
        plt.savefig(f"images/{title}")
        
        
    def DecisionTree_plot(self, clf):
        fig = plt.figure(figsize=(35,15))
        _ = tree.plot_tree(clf, 
                   feature_names=self.prepro.features_description,  
                   fontsize=22,
                   class_names=["Arrested", "Non Arrested"],
                   rounded=True,
                   filled=True)
        plt.savefig("images/treeExample")
        
        
        
    def affichage_BAR_Primary_Type(self):
        Primary_Type = np.unique(self.prepro.data_panda['Primary Type'].to_numpy())
        Proportion_Primary_Type = np.empty(len(Primary_Type))
        Proportion_arrest = np.empty(len(Primary_Type))
        nb_tot = self.prepro.data.shape[0]
        for k, prim_type in enumerate(Primary_Type):
            nb_crime = np.sum(np.where(self.prepro.data_panda['Primary Type']==prim_type,1,0))
            Proportion_Primary_Type[k] = nb_crime/nb_tot
            nb_arrest = np.sum(np.where(((self.prepro.data_panda['Primary Type']==prim_type) & (self.prepro.data_panda['Arrest']==1)),1,0))
            Proportion_arrest[k] = nb_arrest/nb_crime
        pourcentage = .03
        A = Primary_Type[Proportion_Primary_Type>pourcentage]
        B = Proportion_Primary_Type[Proportion_Primary_Type>pourcentage]
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        b0 = ax.bar(range(len(A)), B, width=.95)
        ax.bar_label(b0, labels=A)
        ax.set_xlabel("Primary_Type")
        ax.set_ylabel("Proportion Primary_Type/nb_donnees")
        plt.savefig("images/PourcentagePrimaryTypes")
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        b1 = ax.bar(range(len(Primary_Type)), Proportion_arrest, width=.95)
        ax.bar_label(b1, labels=Primary_Type)
        ax.set_xlabel("Primary_Type")
        ax.set_ylabel("Proportion Primary_Type/nb_donnees")
        plt.savefig("images/NombreArrestationParCrime")


    def crimeexample(self, prepro, train, n = 10):
        train.traintestsplit(0.3)
        encodage = prepro.encodage
        ids = np.array(rd.rand(n)*train.X_test.shape[0], dtype=int)
        train.fit_DecisionTree(display=False)
        pred, test, proba = train.predict_DecisionTree()

        print(prepro.features_description)
        print()
        for i in ids:
            s = "\033[0;32mbien\033[0m" if pred[i] == test[i] else "\033[0;31mmal\033[0m"
            pc = proba[i][1] 
            q = "arrestation" if(pc > .5) else "pas d'arrestation"
            pc = max(pc, 1-pc)
            ptype = prepro.encodage[1][train.X_test[i, 1]]
            desc = prepro.encodage[2][train.X_test[i, 2]]
            part = prepro.encodage[10][train.X_test[i, 10]]
            ldesc = prepro.encodage[3][train.X_test[i, 3]]
            #month = prepro.encodage[14][train.X_test[i, 14]]
            year = prepro.encodage[11][train.X_test[i, 11]]
            print(f"Le crime \033[0;34m{ptype}: {desc} à {part}h {year} dans {ldesc}\033[0m\n est {s} classé: {100*pc:.0f}% {q}")
        print()
