import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Kmeans
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv('Crimes100K.csv', low_memory=False)
del data['ID']
del data['Case Number']
del data['Block']
del data['Updated On']
del data['Longitude']
del data['Latitude']
del data['Location']


X_coordinate = data['X Coordinate']
Y_coordinate = data['Y Coordinate']
Coordinates = np.c_[X_coordinate, Y_coordinate]
print("Taille de coordonnee", Coordinates.shape)
data = data.to_numpy() 

def split_date(X):
    '''On sépare la colonne date en 5 parties : 
    -date1 : Matin/journée/soir/nuit (6h-10h; 10h-18h; 18h-22h; 22h-6h)
    -date2 : jours de la semaine 
    -date3 : week-end/semaine 
    -date4 : mois 
    -date5 : heure
    '''
    date = X[:,0]
    size = X.shape[0]
    date1, date2, date3, date4, date5 = np.zeros(size, dtype=int), np.zeros(size, dtype=int),\
            np.zeros(size, dtype=int), np.zeros(size, dtype=int), np.zeros(size, dtype=int)
    for i, k in enumerate(date):
        heure = int(k[11:13]) + (k[-2:]=="PM")*12
        if(heure>=6 and heure<10):
            date1[i]=0
        elif(heure>=10 and heure<18):
            date1[i]=1
        elif(heure>=18 and heure<22):
            date1[i]=2
        else:
            date1[i]=3
            
        jour = k[3:5]
        mois = k[0:2]
        annee = k[6:10]

        
        date2[i] = datetime.fromisoformat(f'{annee}-{mois}-{jour}').weekday() # 0 lundi - 6 dimanche
        
        date3[i] = (date2[i]>4)# 0 semaine - 1 week-end
        
        date4[i] = int(mois)

        date5[i] = heure

    return date1, date2, date3, date4, date5    

date1, date2, date3, date4, date5 = split_date(data)
data = np.c_[data, date1, date2, date3, date4, date5]
data = data[:, 1:]
print(data.shape)

data = OrdinalEncoder().fit_transform(data)
print(data.shape)
print(data)


def affiche_coordonnee(X,Y):
    plt.figure(1)
    plt.scatter(X,Y, s=1, marker='.')
    print("Création image affichage_coordonnee")
    plt.savefig("affichage_coordonnee")

affiche_coordonnee(X_coordinate, Y_coordinate)
km = KMeans(50)
predicts = km.fit_predict(Coordinates)

fig, ax = plt.subplots(1, 1)
ax.scatter(X_coordinate, Y_coordinate, c=predicts, s=3, marker='.')
plt.savefig('affichage_Kmoyenne')

print(predicts)
data = np.c_[data, predicts]

pd.DataFrame(data).to_csv("Crimes100K_featured.csv")
