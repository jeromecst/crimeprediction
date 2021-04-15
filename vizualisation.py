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
        
        
        
    def affichage_BAR_Primary_Type(self):
        Primary_Type = np.unique(self.prepro.data_panda['Primary Type'].to_numpy())
        Proportion_Primary_Type = np.empty(len(Primary_Type))
        nb_tot = 1000000
        for k, prim_type in enumerate(Primary_Type):
            nb = np.sum(np.where(self.prepro.data_panda['Primary Type']==prim_type,1,0))
            Proportion_Primary_Type[k] = nb/nb_tot

        A = Primary_Type[Proportion_Primary_Type>0.01]
        B = Proportion_Primary_Type[Proportion_Primary_Type>0.01]
        fig, ax = plt.subplots(1, 1)
        b = ax.bar(A, B, 1.0)
        ax.bar_label(b, labels=A)
        ax.set_xlabel("Primary_Type")
        ax.set_ylabel("Proportion Primary_Type/1M")
        plt.savefig("Images/BAR_Primary_Type")
        
        
        Primary_Type = np.unique(data['Primary Type'].to_numpy())
print(Primary_Type)
Proportion_crimes = np.empty(len(Primary_Type))
Proportion_arrest = np.empty(len(Primary_Type))

nb_tot = 1000000
for k, prim_type in enumerate(Primary_Type):
    
    nb_crime = np.sum(np.where(data['Primary Type']==prim_type,1,0))
    nb_arrest = np.sum(np.where(((data['Primary Type']==prim_type) & (data['Arrest']==1)),1,0))
    Proportion_arrest[k] = nb_arrest/nb_crime
    #nb_non_arrest = np.sum(np.where(data['Primary Type']==prim_type and data['Arrest']==0,1,0))
    Proportion_crimes[k] = nb_crime/nb_tot
    

A = Primary_Type[Proportion_crimes>0.01]
B = Proportion_crimes[Proportion_crimes>0.01]

fig, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].bar(A, B, 1.0)
ax[1].bar(Primary_Type, Proportion_arrest, 1.0)