import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Kmeans
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder

class preprocessing:
    
    def __init__(self, fichierCSV): ## instanciation d'un objet du type de la classe.
        self.data = pd.read_csv(fichierCSV, low_memory=False)
        self.delete_usless_features()
        self.data_panda = self.data.copy()

        X_coordinate = self.data['X Coordinate']
        Y_coordinate = self.data['Y Coordinate']
        self.Coordinates = np.c_[X_coordinate, Y_coordinate]
        
        self.km_predicts = 0
        self.km_clusters = 0
        self.km = 0
        self.encodage = 0
        
        self.data = self.data.to_numpy()
    
    def delete_usless_features(self):
        '''
        Supprime les features inutiles
        '''
        del self.data['ID']
        del self.data['Case Number']
        del self.data['Block']
        del self.data['Updated On']
        del self.data['Longitude']
        del self.data['Latitude']
        del self.data['Location']
    
    
    def split_date(self):
        '''On sépare la colonne date en 5 parties : 
        -date1 : Matin/journée/soir/nuit (6h-10h; 10h-18h; 18h-22h; 22h-6h)
        -date2 : jours de la semaine 
        -date3 : week-end/semaine 
        -date4 : mois 
        -date5 : heure
        @Return date1, date2, date3, date4, date5
        '''
        date = self.data_panda['Date']
        size = self.data.shape[0]
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

            # 0 lundi - 6 dimanche
            date2[i] = datetime.fromisoformat(f'{annee}-{mois}-{jour}').weekday() 

            date3[i] = (date2[i]>4)# 0 semaine - 1 week-end

            date4[i] = int(mois)

            date5[i] = heure

        return date1, date2, date3, date4, date5 
    
        
    def Kmeans_coordinate(self, K):
        '''
        On applique l'algorithme des kmeans de la classe scikit_learn aux coordonnées des crimes.
        @Param : K -> int : Nombre de clusters
        '''
        self.km = KMeans(K)
        self.km_predicts = self.km.fit_predict(self.Coordinates)
        self.km_clusters = np.zeros(K, dtype=int)
        for i in self.km_predicts:
            self.km_clusters[i] += 1
    
    
    def affiche_coordonnee(self):
        '''
        Crée un fichier affichage_coordonnée.png des coordonnées des crimes sur une carte 
        '''
        plt.figure(1)
        plt.scatter(self.Coordinates[:,0],self.Coordinates[:,1], s=1, marker='.')
        print("Création image affichage_coordonnee")
        plt.savefig("affichage_coordonnee")

        
        
    def affiche_Kmeans_coordinate(self):
        '''
        Crée un fichier affichage_Kmoyenne.png des coodonnées des crimes colorié 
        selon les différents clusters
        '''
        ids = np.argsort(self.km_clusters)
        crimes_sorted = self.km_clusters[ids]
        point = self.km.cluster_centers_[ids]

        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.Coordinates[:,0], self.Coordinates[:,1], c=self.km_predicts, s=3, marker='.')
        for i in range(1, 6):
            plt.scatter(point[-i,0], point[-i,1], label=i+1, s = 400, c='red', marker='*')
            plt.text(point[-i,0], point[-i,1], i, c='black', ha="center", va="center", size=10)
        print("sauvegarde du fichier affichage_Kmoyenne")
        plt.savefig('affichage_Kmoyenne')

    
    def ajout_Kmeans_coordinate(self):
        '''
        Ajoute une nouvelle features de prediction des crimes grace aux kmeans
        '''
        self.data = np.c_[self.data, self.km_predicts]
    
    def ajout_dates(self):
        '''
        Ajoute 5 nouvelles features dates, correspondants aux nouvelles dates utilisables. 
        On supprime l'ancienne colonne date qui n'est pas utilisable.
        '''
        date1, date2, date3, date4, date5 = self.split_date()
        self.data = np.c_[self.data,date1, date2, date3, date4, date5]
        self.data = self.data[:, 1:]

    def encodage_features(self):
        OE = OrdinalEncoder()
        X = OE.fit_transform(self.data)
        self.encodage = OE.categories_
        return X

    def XYsplit(self):
        print(self.data)

    def save_to_csv(self):
        pd.DataFrame(self.data).to_csv("Crimes100K_featured.csv")