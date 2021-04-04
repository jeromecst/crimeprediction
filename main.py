import numpy as np
import pandas as pd
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

def comparaison_features():
    compar_date_lieu_Gauss = np.zeros((5,4))
    compar_date_lieu_Tree = np.zeros((5,4))

    dates = ['Part of the day', 'Weekday', 'Weekend', 'Month', 'Hour']
    lieux = ['Ward', 'Community Area', 'District', 'Cluster']

    for i, date_i in enumerate(dates):
        for j, lieu_i in enumerate(lieux):
            X_temp, Y_temp = X.copy(), Y.copy()
            prepro.extract_date(X_temp, date_i)
            prepro.extract_lieu(X_temp, lieu_i)
            #print(f"----------------------Début du training, avec date_i = {date_i} et lieu_i = {lieu_i}-----------------------\n")      
            tr = train.train(X_temp,Y_temp)
            tr.traintestsplit(0.3)
            #print("\n----------Début du training_GaussienNB------------\n")      
            compar_date_lieu_Gauss[i][j]=tr.fit_GaussNB()
            #print("\n----------Début du training_DecisionTree----------\n")
            compar_date_lieu_Tree[i][j]=tr.fit_DecisionTree()
        
    panda_Gauss = pd.DataFrame(compar_date_lieu_Gauss, columns=lieux, index=dates)
    print(panda_Gauss)
    print("\n-------------------------------\n")
    panda_Tree = pd.DataFrame(compar_date_lieu_Tree, columns=lieux, index=dates)
    print(panda_Tree)


ignorePreprocessedFile = False
print("\n----------------------Début du preprocessing-----------------------\n") 
if path.isfile(f"{file}_prepro.csv") and not ignorePreprocessedFile:
    print(f"file found, reusing {file}_prepro.csv !")
    prepro = preprocessing.preprocessing(file, already_preprocessed=True)
    data_encodée = prepro.data
else:
    time_before = time.time()
    prepro = preprocessing.preprocessing(file)
    prepro.ajout_dates()
    prepro.Kmeans_coordinate(50)
    prepro.ajout_Kmeans_coordinate()
    #prepro.affiche_Kmeans_coordinate()
    data_encodée = prepro.encodage_features()
    prepro.save_to_csv(file)
    data_encodée = prepro.encodage_features()
    time_after = time.time()
    print("Temps pour le preprocessing : ", (time_after-time_before), " secondes\n")

X, Y = prepro.XYsplit(data_encodée)

train = train.train(X,Y)
train.traintestsplit(0.3)
display = True
print("\n----------Début du training_GaussienNB------------\n")      
train.fit_GaussNB(display)
print("\n----------Début du training_DecisionTree----------\n")
train.fit_DecisionTree(display)
