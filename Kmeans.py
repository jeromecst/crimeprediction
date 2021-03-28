import numpy as np
import matplotlib.pyplot as plt


class Kmoyennes:

    

    def __init__(self,K, IterationMax): ## instanciation d'un objet du type de la classe.
        self.K = K
        self.N = 0
        self.D = 0
        self.IterationMax = IterationMax
        self.affectations = np.zeros((self.N))
        self.representants = np.zeros((self.K,self.D))
    
    def fit(self,X):
        self.N = X.shape[0]
        self.D = X.shape[1]
        representants_initiaux = np.array(X[0:self.K])
        #representants_initiaux = np.random.random((self.K,self.D))
        representants = representants_initiaux    
        ## boucle (comme plus haut)
        for i in range(self.IterationMax):
            a = self.maj_affectations(X,representants)
            representants = self.maj_representants(X, a)
            print("Iteration n° :", i)
            
        ## on affecte le resultat dans les variables membres de la classe.
        self.representants = representants
        self.affectations = a
        print(a)
        return self.representants
    
    
    def barycentre(self,X):
        Nselection = X.shape[0]
        if Nselection != 0:
            resultat = X.sum(axis=0)/Nselection
        else:
            resultat = X.sum(axis=0)*0.0
        return resultat
    
    def distance (self,p1,p2):

        if (p1.shape[0] != p2.shape[0]):
            print("Erreur pas les mêmes dimensions")
            return None
        prod_scal = 0
        for i in range (self.D):
            prod_scal += (p2[i]-p1[i])**2
        return np.sqrt(prod_scal)
    
    def maj_affectations(self, X, r):
        '''
        X représente nos points, et r représente les centres
        N réprésente le nombre d'images
        K le nombre de centres
        D la dimension
        '''
        a = np.zeros((self.N))

        for i in range (self.N):
            dist_min = 10000
            r_proche = 0
            for j in range(self.K):
                dist = self.distance(X[i],r[j])
                if(dist<dist_min):
                    dist_min=dist
                    r_proche=j
            a[i]=r_proche         

        return a

    def maj_representants(self, X, affectations):
        representants = np.zeros((self.K,self.D))
        for k in range(self.K):
            index = X[affectations == k]
            representants[k] = self.barycentre(index)
        return representants
        
    def affichage(self, X):
        colors = ['b', 'r','y', 'navy', 'g', 'm']
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        for k in range(self.K):
            K1 = X[self.affectations==k]
            ax.scatter(K1[:,0], K1[:,1],s=4, label=f"cluster {k+1}",marker='+' , c=colors[k])
        ax.legend()
        print("Création de l'image : affichage_Kmoyenne")
        plt.savefig("affichage_Kmoyenne")