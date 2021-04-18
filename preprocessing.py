import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
from matplotlib import cm
import matplotlib.colors as colors

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
            self.features_description_private = headers.to_numpy(copy=True)
            self.features_description = 0
            
            self.data = self.data.to_numpy()
        else:
            self.data = pd.read_csv(f"{fichierCSV}.csv", low_memory=False)
            self.delete_usless_features()
            self.data_panda = self.data.copy()

            X_coordinate = self.data['X Coordinate']
            Y_coordinate = self.data['Y Coordinate']
            self.Coordinates = np.c_[X_coordinate.copy(), Y_coordinate.copy()]
            
            headers = self.data.columns
            self.features_description_private = headers.to_numpy(copy=True)
            
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
        date1, date2, date3, date4, date5 = [None]*size,[None]*size,[None]*size,[None]*size,[None]*size
        for i, k in enumerate(date):
            heure = int(k[11:13]) + (k[-2:]=="PM")*12
            if(heure>=2 and heure<6):
                date1[i]="night"
            elif(heure>=5 and heure<7):
                date1[i]="early morning"
            elif(heure>=7 and heure<10):
                date1[i]="morning"
            elif(heure>=10 and heure<12):
                date1[i]="late morning"
            elif(heure>=12 and heure<16):
                date1[i]="early afternoon"
            elif(heure>=16 and heure<18):
                date1[i]="late afternoon"
            elif(heure>=18 and heure<21):
                date1[i]="evening"
            elif(heure>=21 and heure<23):
                date1[i]="late evening"
            else:
                date1[i]="midnight"

            jour = k[3:5]
            mois = k[0:2]
            annee = k[6:10]

            # 0 lundi - 6 dimanche
            date2[i] = datetime.fromisoformat(f'{annee}-{mois}-{jour}').weekday()

            date3[i] = "week-end" if (date2[i]>4) else "week"# 0 semaine - 1 week-end

            date4[i] = mois

            date5[i] = heure

        return date1, date2, date3, date4, date5 
    
    def extract_date(self, X, date_i):
        dates = ['Part of the day', 'Weekday', 'Weekend', 'Month', 'Hour']
        for d in dates:
            if(date_i !=d):
                id_d = self.features_description_private == d
                np.delete(X, id_d, 1)
    
    def extract_lieu(self, X, lieu_i):
        lieux = ['Ward', 'Community Area', 'District', 'Cluster']
        for l in lieux:
            if(l != lieu_i):
                id_l = self.features_description_private == l
                np.delete(X, id_l, 1)
        


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
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        plt.scatter(self.Coordinates[:,0],self.Coordinates[:,1], s=.5, marker='.')
        plt.axis('off')
        print("Création image affichage_coordonnee")
        plt.savefig("images/affichage_coordonnee")

    def affiche_Kmeans_coordinate(self):
        '''
        Crée un fichier affichage_Kmoyenne.png des coodonnées 
        des crimes colorié 
        selon les différents clusters
        '''
        ids = np.argsort(self.km_clusters)
        crimes_sorted = self.km_clusters[ids]
        point = self.km.cluster_centers_[ids]

        #predicts_colors = []
        predict_0_1 = []
        maax = crimes_sorted[-1]
        for i in range(self.data.shape[0]):
            c = self.km_clusters[self.km_predicts[i]]/maax
            #predicts_colors += [(c, c, 0)]
            predict_0_1 += [c]

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        affichage = ax.scatter(self.Coordinates[:,0], self.Coordinates[:,1],c=predict_0_1, s=.5, marker='.', cmap=plt.cm.Reds)
        fig.colorbar(affichage,ax=ax)

        #for i in range(1, 6):
        #    plt.scatter(point[-i,0], point[-i,1], label=i+1, s = 400, c='red', marker='*')
        #    plt.text(point[-i,0], point[-i,1], i, c='black', ha="center", va="center", size=10)
        plt.axis('off')
        ax.set_title("Carte des crimes à Chicago")
        print("sauvegarde du fichier affichage_Kmoyenne")
        plt.savefig('images/affichage_Kmoyenne')

    
    def remove_Kmeans_coordinate(self):
        ids = np.where(self.features_description_private == "Cluster")
        self.data = np.delete(self.data, ids, 1)
        self.features_description_private = np.delete(self.features_description_private, ids)
        
    def ajout_Kmeans_coordinate(self):
        '''
        Ajoute une nouvelle features de prediction des crimes grace aux kmeans
        '''
        self.data = np.c_[self.data, self.km_predicts]
        self.features_description_private = np.append(self.features_description_private, "Cluster")
        for str in ["X Coordinate", "Y Coordinate"]:
            todelete = np.where(self.features_description_private == str)
            self.data = np.delete(self.data, todelete, 1)
            self.features_description_private = np.delete(self.features_description_private, todelete)
    
    def ajout_dates(self):
        '''
        Ajoute 5 nouvelles features dates, correspondants aux nouvelles dates utilisables. 
        On supprime l'ancienne colonne date qui n'est pas utilisable.
        '''
        date1, date2, date3, date4, date5 = self.split_date()
        self.data = np.c_[self.data,date1, date2, date3, date4, date5]
        todelete = np.where(self.features_description_private == "Date")
        self.data = np.delete(self.data, todelete, 1)
        self.features_description_private = np.delete(self.features_description_private, todelete)
        self.features_description_private = np.append(self.features_description_private, ['Part of the day', 'Weekday', 'Weekend', 'Month', 'Hour'])

    def encodage_features(self):
        OE = OrdinalEncoder()
        X = OE.fit_transform(self.data)
        X = X.astype('int')
        self.encodage = OE.categories_
        self.data_encodée = X
        return X

    def XYsplit(self, data):
        id_arrest = self.features_description_private == 'Arrest'
        X = data[:, id_arrest!=True]
        Y = data[:, id_arrest]
        self.features_description = self.features_description_private[id_arrest != True]
        return X, Y.flatten()

    def save_to_csv(self, title):
        print(f"Saving file to {title}_prepro.csv...")
        pd.DataFrame(self.data_encodée).to_csv(f"{title}_prepro.csv", header = self.features_description_private, index=False)
