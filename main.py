import numpy as np
import preprocessing
import train
import os.path
from os import path

file = "Crimes100K"

def affichage_encodage():
    for i in range(len(prepro.features_description)):
        for j in range(2):
            print("encodage du",prepro.features_description[i],j,prepro.encodage[i][j])

ignorePreprocessedFile = False
if path.isfile(f"{file}_prepro.csv") and not ignorePreprocessedFile:
    print("file found!")
    prepro = preprocessing.preprocessing(file, already_preprocessed=True)
    data_encodée = prepro.data
else:
    prepro = preprocessing.preprocessing(file)
    prepro.ajout_dates()
    prepro.Kmeans_coordinate(50)
    prepro.ajout_Kmeans_coordinate()
    #prepro.affiche_Kmeans_coordinate()
    data_encodée = prepro.encodage_features()
    prepro.save_to_csv(file)

X, Y = prepro.XYsplit(data_encodée)
print(X.shape, Y.shape)
