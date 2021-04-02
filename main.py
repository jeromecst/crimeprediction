import numpy as np
import preprocessing
import train

prepro = preprocessing.preprocessing("Crimes100K.csv")
prepro.ajout_dates()
prepro.Kmeans_coordinate(10)
prepro.ajout_Kmeans_coordinate()
#prepro.affiche_Kmeans_coordinate()
data = prepro.encodage_features()
print("encodage du lieu 0 :",prepro.encodage[3][0])
print("encodage du lieu 4 :",prepro.encodage[3][4])
