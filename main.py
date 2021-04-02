import numpy as np
import preprocessing
import train

def affichage_encodage():
    for i in range(len(prepro.features_description)):
        for j in range(2):
            print("encodage du",prepro.features_description[i],j,prepro.encodage[i][j])

prepro = preprocessing.preprocessing("Crimes100K.csv")
prepro.ajout_dates()
prepro.Kmeans_coordinate(50)
prepro.ajout_Kmeans_coordinate()
prepro.affiche_Kmeans_coordinate()
data_encodée = prepro.encodage_features()
X, Y = prepro.XYsplit(data_encodée)
