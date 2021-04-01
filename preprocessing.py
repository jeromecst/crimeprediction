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


            date2[i] = datetime.fromisoformat(f'{annee}-{mois}-{jour}').weekday() # 0 lundi - 6 dimanche

            date3[i] = (date2[i]>4)# 0 semaine - 1 week-end

            date4[i] = int(mois)

            date5[i] = heure

        return date1, date2, date3, date4, date5 
    
        
    def Kmeans_coordinate(self, K):

        print("Taille de coordonnee", self.Coordinates.shape)
        
        #Appelle de la classe Kmeans

        self.km = KMeans(K)
        self.km_predicts = self.km.fit_predict(self.Coordinates)
        self.km_clusters = np.zeros(K, dtype=int)
        for i in self.km_predicts:
            self.km_clusters[i] += 1
    
    
    def affiche_coordonnee(self):
        plt.figure(1)
        plt.scatter(self.Coordinates[:,0],self.Coordinates[:,1], s=1, marker='.')
        print("Taille coord[0] : ", self.Coordinates[0].size)
        print("Création image affichage_coordonnee")
        plt.savefig("affichage_coordonnee")

        
        
    def affiche_Kmeans_coordinate(self):
        max_crime = np.max(self.km_clusters)
        max_clus = np.argmax(self.km_clusters)
        point = self.km.cluster_centers_[np.argmax(self.km_clusters)]
        print("Le cluster où il y a le plus de crimes est le numéro", max_clus, "avec",
                max_crime, "cimes, de coordonnée", point)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.Coordinates[:,0], self.Coordinates[:,1], c=self.km_predicts, s=3, marker='.')
        ax.scatter(point[0], point[1], c='yellow', s=100, marker='*', label="dont go there")
        ax.legend()
        plt.savefig('affichage_Kmoyenne')

    
    def ajout_Kmeans_coordinate(self):
        self.data = np.c_[self.data, self.km_predicts]
    
    def ajout_dates(self):
        date1, date2, date3, date4, date5 = self.split_date()
        self.data = np.c_[self.data,date1, date2, date3, date4, date5]
        self.data = self.data[:, 1:]

    def encodage_features(self):
        OE = OrdinalEncoder()
        X = EO.fit_transform(self.data)
        self.encosage = EO.categories_
        return X


    def save_to_csv(self):
        pd.DataFrame(self.data).to_csv("Crimes100K_featured.csv")
        
        
prepro = preprocessing("Crimes100K.csv")        
prepro.ajout_dates()
prepro.affiche_coordonnee()
prepro.Kmeans_coordinate(50)
prepro.affiche_Kmeans_coordinate()
prepro.ajout_Kmeans_coordinate()



