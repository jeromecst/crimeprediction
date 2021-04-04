import numpy as np
import preprocessing
import train
import time

def affichage_encodage():
    for i in range(len(prepro.features_description)):
        for j in range(2):
            print("encodage du",prepro.features_description[i],j,prepro.encodage[i][j])

print("\n----------------------Début du preprocessing-----------------------\n") 
time_before = time.time()
prepro = preprocessing.preprocessing("Crimes100K.csv")
prepro.ajout_dates()
prepro.Kmeans_coordinate(50)
prepro.ajout_Kmeans_coordinate()
time_after = time.time()
print("Temps pour le preprocessing : ", (time_after-time_before), " secondes\n")
#prepro.affiche_Kmeans_coordinate()
data_encodée = prepro.encodage_features()
X, Y = prepro.XYsplit(data_encodée)
print("----------------------Début du training-----------------------\n")      
train = train.train(X,Y)
train.traintestsplit(0.3)
print("\n----------Début du training_GaussienNB------------\n")      
train.fit_GaussNB()
print("\n----------Début du training_DecisionTree----------\n")
train.fit_DecisionTree()



