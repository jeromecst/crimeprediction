import numpy as np
import pandas as pd
import preprocessing
import train
import vizualisation
import sys
import time
import matplotlib.pyplot as plt
from os import path

crossvalidation=False # active toutes les fonctions qui font de la cross validation
 
if(len(sys.argv) < 2):
    print("Specify a file\n Example : python main.py Crimes100KEq.csv")
    exit(1)
else:
    file = str(sys.argv[1][:-4])
    print(file)
if(len(sys.argv) > 2):
    ignorePreprocessedFile = False
else:
    ignorePreprocessedFile = True

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
    plt.savefig("images/bestNumClusters")


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

    plt.savefig("images/bestParamDecisionTree")

def bestNumberData(train):
    X, Y = prepro.XYsplit(data_encodée)
    score = []
    N = X.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    the_range = np.linspace(N, 1000, 20, dtype=int)
    for n in the_range:
        Xsub = X[:int(n)].copy()
        Ysub = Y[:int(n)].copy()
        train.load_data(Xsub, Ysub)
        train.traintestsplit(0.3)
        score += [train.fit_DecisionTree(True)]
    ax.plot(the_range, score)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Score")
    ax.legend()
    plt.savefig("images/bestNumberData")

def affichage_encodage():
    for i in range(len(prepro.features_description)):
        for j in range(2):
            print("encodage du",prepro.features_description[i],j,prepro.encodage[i][j])

print("\nDébut du preprocessing\n") 
display = False
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
    if(display):
        prepro.affiche_Kmeans_coordinate()
        prepro.affiche_coordonnee()
    data_encodée = prepro.encodage_features()
    prepro.save_to_csv(file)
    time_after = time.time()
    print("Temps pour le preprocessing : ", (time_after-time_before), " secondes\n")

X, Y = prepro.XYsplit(data_encodée)
train = train.train(X,Y)
train.traintestsplit(0.3)

if(crossvalidation):
    bestParamDecisionTree(train, X, Y)
    bestNumberOfClusters(prepro, train)
    bestNumberData(train)

#X, Y = prepro.XYsplit(data_encodée)
#train.load_data(X,Y)
#train.traintestsplit(0.3)
display = True
print("\nDébut du training_GaussienNB\n")      
train.fit_GaussNB(display)

print("\nDébut du training_DecisionTree\n")
train.fit_DecisionTree(display)
train.DecisionTree_feature_importances(prepro.features_description)

print("\nDébut du training_RandomForestClassifier\n")
train.fit_RandomForestClassifier(100, display=display) 


print("\nDébut de la visualisation\n")
y_pred_Decision_Tree, Y_test, _ = train.predict_DecisionTree()
y_pred_Gauss, Y_test, _ = train.predict_Gauss()
vizu = vizualisation.vizualisation(prepro)
vizu.matrice_confusion("MatriceConfusionDecisionTree", y_pred_Decision_Tree, Y_test)
vizu.matrice_confusion("MatriceConfusionGauss", y_pred_Gauss, Y_test)
if(ignorePreprocessedFile): vizu.crimeexample(prepro, train, n = 20)

vizu.affichage_BAR_Primary_Type()

train.fit_DecisionTree(display=True, max_depth = 3)
clf = train.model_DecisionTree()
vizu.DecisionTree_plot(clf)
