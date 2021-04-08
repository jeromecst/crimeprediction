import numpy as np
import pandas as pd
import preprocessing
import train
import vizualisation
import time
import os.path
import matplotlib.pyplot as plt
from os import path

file = "Crimes100KEq" # ne pas écrire .csv
ignorePreprocessedFile = False

def bestNumberOfClusters(prepro, train, n = 12):
    if ignorePreprocessedFile == False:
        print("file needs to be reprocessed")
        return
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    score_list = []
    cluster_list = range(1, 71, 5)
    for cluster in cluster_list:
        prepro.Kmeans_coordinate(25)
        prepro.remove_Kmeans_coordinate()
        prepro.ajout_Kmeans_coordinate()
        X, Y = prepro.XYsplit(data_encodée)
        train.load_data(X, Y)
        print(cluster)
        score = 0
        for _ in range(n):
            train.traintestsplit(0.3)
            score += train.fit_DecisionTree(True)
        score_list += [score/n]
    ax.plot(cluster_list, score_list)
    ax.set_xlabel("clusters")
    ax.set_ylabel("score")
    ax.set_title("best number of clusters")
    plt.savefig("bestNumClusters")


def bestParamDecisionTree(train, X, Y):
    """
    Peut prendre beaucoup de temps, à éxécuter seulement sur un petit
    ensemble de données ( < 100K )
    """
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    score = []
    sample_list = range(2, 300, 50)
    for min_sample_split in sample_list:
        score += [train.fit_DecisionTree(min_samples_split = min_sample_split)]
    ax[0][0].plot(sample_list, score)
    ax[0][0].set_xlabel("min_samples_split")
    ax[0][0].set_ylabel("score")

    score = []
    sample_list = range(1, 300, 50)
    for min_samples_leaf in sample_list:
        score += [train.fit_DecisionTree(min_samples_leaf = min_samples_leaf)]
    ax[1][0].plot(sample_list, score)
    ax[1][0].set_xlabel("min_samples_leaf")
    ax[1][0].set_ylabel("score")

    score = []
    feature_list = range(1, 18)
    for max_features in feature_list:
        score += [train.fit_DecisionTree(max_features = max_features)]
    ax[0][1].plot(feature_list, score)
    ax[0][1].set_xlabel("max_feature")
    ax[0][1].set_ylabel("score")

    score = []
    impurity = np.linspace(0, .2, 40)
    for im in impurity:
        score += [train.fit_DecisionTree(min_impurity_decrease = im)]
    ax[1][1].plot(impurity, score)
    ax[1][1].set_xlabel("min_impurity_decrease")
    ax[1][1].set_ylabel("score")

    plt.savefig("bestParamDecisionTree")

def affichage_encodage():
    for i in range(len(prepro.features_description)):
        for j in range(2):
            print("encodage du",prepro.features_description[i],j,prepro.encodage[i][j])

def comparaison_features(train, X, Y):
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
            train.load_data(X_temp,Y_temp)
            train.traintestsplit(0.3)
            #print("\n----------Début du training_GaussienNB------------\n")      
            compar_date_lieu_Gauss[i][j]=train.fit_GaussNB()
            #print("\n----------Début du training_DecisionTree----------\n")
            compar_date_lieu_Tree[i][j]=train.fit_DecisionTree()
        
    panda_Gauss = pd.DataFrame(compar_date_lieu_Gauss, columns=lieux, index=dates)
    print(panda_Gauss)
    print("\n-------------------------------\n")
    panda_Tree = pd.DataFrame(compar_date_lieu_Tree, columns=lieux, index=dates)
    print(panda_Tree)

print("\n----------------------Début du preprocessing-----------------------\n") 
if path.isfile(f"{file}_prepro.csv") and not ignorePreprocessedFile:
    print(f"file found, re-using {file}_prepro.csv !")
    prepro = preprocessing.preprocessing(file, already_preprocessed=True)
    data_encodée = prepro.data
else:
    time_before = time.time()
    prepro = preprocessing.preprocessing(file)
    prepro.ajout_dates()
    prepro.Kmeans_coordinate(40)
    prepro.ajout_Kmeans_coordinate()
    #prepro.affiche_Kmeans_coordinate()
    data_encodée = prepro.encodage_features()
    prepro.save_to_csv(file)
    time_after = time.time()
    print("Temps pour le preprocessing : ", (time_after-time_before), " secondes\n")

X, Y = prepro.XYsplit(data_encodée)
train = train.train(X,Y)
train.traintestsplit(0.3)

#bestParamDecisionTree(train, X, Y)
#comparaison_features(train, X, Y)
#bestNumberOfClusters(prepro, train)

train.traintestsplit(0.3)
display = True
print("\n----------Début du training_GaussienNB------------\n")      
train.fit_GaussNB(display)

print("\n----------Début du training_DecisionTree----------\n")
train.fit_DecisionTree(display)
#train.DecisionTree_feature_importances(prepro.features_description)

print("\n----------Début de la visualisation----------\n")

clf = train.model_DecisionTree()
y_pred_Decision_Tree, Y_test = train.predict_DecisionTree()
vizu = vizualisation.vizualisation(prepro)
vizu.matrice_confusion(y_pred_Decision_Tree, Y_test)
vizu.DecisionTree_plot(clf)

#print("\n----------Début du training_RandomForestClassifier----------\n")
#train.fit_RandomForestClassifier(display)
