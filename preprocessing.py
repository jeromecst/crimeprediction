import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder

class preprocessing:
    

        ## instanciation d'un objet du type de la classe.
    def __init__(self, fichierCSV, already_preprocessed=False): 
        self.km_predicts = 0
        self.km_clusters = 0
        self.km = 0
        self.encodage = 0
        if(already_preprocessed):
            self.data = pd.read_csv(f"{fichierCSV}_prepro.csv", low_memory=False)
            headers = self.data.columns
            self.data_panda = self.data.copy()
            X_coordinate = self.data['X Coordinate']
            Y_coordinate = self.data['Y Coordinate']
            self.Coordinates = np.c_[X_coordinate, Y_coordinate]
            self.features_description = headers.to_numpy(copy=True)
            
            self.data = self.data.to_numpy()
        else:
            self.data = pd.read_csv(f"{fichierCSV}.csv", low_memory=False)
            self.delete_usless_features()
            self.data_panda = self.data.copy()

            X_coordinate = self.data['X Coordinate']
            Y_coordinate = self.data['Y Coordinate']
            self.Coordinates = np.c_[X_coordinate, Y_coordinate]
            
            headers = self.data.columns
            self.features_description = headers.to_numpy(copy=True)
            
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
    
    def extract_date(self, X, date_i):
        dates = ['Part of the day', 'Weekday', 'Weekend', 'Month', 'Hour']
        for d in dates:
            if(date_i !=d):
                id_d = self.features_description == d
                np.delete(X, id_d)
    
    def extract_lieu(self, X, lieu_i):
        lieux = ['Ward', 'Community Area', 'District', 'Cluster']
        for l in lieux:
            if(l != lieux_i):
                id_l = self.features_description == l
                np.delete(X, id_l)
        return data_temp
        
    def Kmeans_coordinate(self, K):
        '''
        On applique l'algorithme des kmeans de la classe 
        scikit_learn aux coordonnées des crimes.
        @Param : K -> int : Nombre de clusters
        '''
        self.km = KMeans(K)
        self.km_predicts = self.km.fit_predict(self.Coordinates)
        self.km_clusters = np.zeros(K, dtype=int)
        for i in self.km_predicts:
            self.km_clusters[i] += 1
    
    
    def affiche_coordonnee(self):
        '''
        Crée un fichier affichage_coordonnée.png des coordonnée
        s des crimes sur une carte 
        '''
        plt.figure(1)
        plt.scatter(self.Coordinates[:,0],self.Coordinates[:,1], s=1, marker='.')
        print("Création image affichage_coordonnee")
        plt.savefig("affichage_coordonnee")

        
        
    def affiche_Kmeans_coordinate(self):
        '''
        Crée un fichier affichage_Kmoyenne.png des coodonnées 
        des crimes colorié 
        selon les différents clusters
        '''
        ids = np.argsort(self.km_clusters)
        crimes_sorted = self.km_clusters[ids]
        point = self.km.cluster_centers_[ids]

        predicts_colors = []
        maax = crimes_sorted[-1]
        for i in range(self.data.shape[0]):
            c = self.km_clusters[self.km_predicts[i]]/maax
            predicts_colors += [(c, c, 0)]


        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.scatter(self.Coordinates[:,0], self.Coordinates[:,1], c=predicts_colors, s=.5, marker='.')
        #for i in range(1, 6):
        #    plt.scatter(point[-i,0], point[-i,1], label=i+1, s = 400, c='red', marker='*')
        #    plt.text(point[-i,0], point[-i,1], i, c='black', ha="center", va="center", size=10)
        plt.axis('off')
        ax.set_title("Carte des crimes à Chicago")
        print("sauvegarde du fichier affichage_Kmoyenne")
        plt.savefig('affichage_Kmoyenne')

    
    def ajout_Kmeans_coordinate(self):
        '''
        Ajoute une nouvelle features de prediction des crimes grace aux kmeans
        '''
        self.data = np.c_[self.data, self.km_predicts]
        self.features_description = np.append(self.features_description, "Cluster")
    
    def ajout_dates(self):
        '''
        Ajoute 5 nouvelles features dates, correspondants aux nouvelles dates utilisables. 
        On supprime l'ancienne colonne date qui n'est pas utilisable.
        '''
        date1, date2, date3, date4, date5 = self.split_date()
        self.data = np.c_[self.data,date1, date2, date3, date4, date5]
        self.data = self.data[:, 1:]
        self.features_description = self.features_description[1:]
        self.features_description = np.append(self.features_description, ['Part of the day', 'Weekday', 'Weekend', 'Month', 'Hour'])

    def encodage_features(self):
        OE = OrdinalEncoder()
        X = OE.fit_transform(self.data)
        X = X.astype('int')
        self.encodage = OE.categories_
        self.data_encodée = X
        return X

    def XYsplit(self, data):
        id_arrest = self.features_description == 'Arrest'
        X = data[:, id_arrest!=True]
        Y = data[:, id_arrest]
        self.features_description = self.features_description[id_arrest != True]
        return X, Y.flatten()

    def save_to_csv(self, title):
        pd.DataFrame(self.data_encodée).to_csv(f"{title}_prepro.csv", header = self.features_description, index=False)
