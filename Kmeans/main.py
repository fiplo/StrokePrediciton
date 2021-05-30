import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state"
}

def returnKmeanClusters(data, n):
    kmeans = KMeans(n_clusters=n, **kmeans_kwargs)
    kmeans.fit(data)
    return kmeans

def kmeansIne(data, numarr):
    cs = []
    for i in numarr:
        cs.append(returnKmeanClusters(data, i).inertia_)
    return cs

def kmeanSil(data, numarr):
    sc = []
    for i in numarr:
        score = metrics.silhouette_score(data, returnKmeanClusters(data, i).labels_)
        sc.append(score)
    return sc

def plotKmeans(twoDimensionalData, labels, ax):
    u_labels = np.unique(labels)
    for i in u_labels:
        ax.scatter(twoDimensionalData[labels == i][twoDimensionalData.columns[0][0]] ,
                twoDimensionalData[labels == i][twoDimensionalData.columns[1][0]] , label = i)
    ax.set(xlabel=twoDimensionalData.columns[0][0], ylabel=twoDimensionalData.columns[1][0])
    ax.set_title(f"{len(u_labels)} clusters")

def SilhouetteKmeansPlot(twoDimensionalData, n, ax):
    kmeans = returnKmeanClusters(twoDimensionalData, n)
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax)
    visualizer.fit(twoDimensionalData)

