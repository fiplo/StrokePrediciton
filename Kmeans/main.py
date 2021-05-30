import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

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


