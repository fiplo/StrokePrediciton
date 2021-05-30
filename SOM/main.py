from sklearn_som.som import SOM
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score


def train_som(data, amount_of_clusters):
    som_shape = (amount_of_clusters, 1)
    som = SOM(m=som_shape[0], n=som_shape[1], dim=2)
    som.fit(data)

    return som


def plot_som(data, amount_of_clusters, ax):
    som = train_som(data, amount_of_clusters)

    predictions = som.predict(data)
    x = data[:, 0]
    y = data[:, 1]

    colors = ['red', 'green', 'blue']
    ax.scatter(x, y, c=predictions, cmap=ListedColormap(colors))


def smo_inertia(data, numarr):
    inertia_points = []
    for i in numarr:
        inertia_points.append(train_som(data, i).inertia_)
    return inertia_points


def smo_silhouette_score(data, numarr):
    silhouette_scores = []

    for i in numarr:
        smo = train_som(data, i)
        predictions = smo.predict(data)
        silhouette_scores.append(silhouette_score(data, predictions))

    return silhouette_scores
