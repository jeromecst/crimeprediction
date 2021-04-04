import numpy as np
import preprocessing
import train
import time
import os.path
from os import path

file = "Crimes100K"

def affichage_encodage():
    for i in range(len(prepro.features_description)):
        for j in range(2):
            print("encodage du",prepro.features_description[i],j,prepro.encodage[i][j])

ignorePreprocessedFile = False
print("\n----------------------Début du preprocessing-----------------------\n") 
time_before = time.time()
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

time_after = time.time()
data_encodée = prepro.encodage_features()
print("Temps pour le preprocessing : ", (time_after-time_before), " secondes\n")
X, Y = prepro.XYsplit(data_encodée)
print("----------------------Début du training-----------------------\n")      
train = train.train(X,Y)
train.traintestsplit(0.3)
print("\n----------Début du training_GaussienNB------------\n")      
train.fit_GaussNB()
print("\n----------Début du training_DecisionTree----------\n")
train.fit_DecisionTree()
