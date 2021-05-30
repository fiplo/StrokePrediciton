from sklearn_som.som import SOM
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def train_som(data, amount_of_clusters):
    som_shape = (amount_of_clusters, 1)
    som = SOM(m=som_shape[0], n=som_shape[1], dim=2)
    som.fit(data)

    return som


def plot_som(data, amount_of_clusters):
    som = train_som(data, amount_of_clusters)

    predictions = som.predict(data)
    x = data[:, 0]
    y = data[:, 1]

    colors = ['red', 'green', 'blue']
    plt.scatter(x, y, c=predictions, cmap=ListedColormap(colors))
    plt.show()


def smo_inertia(data, numarr):
    cs = []
    for i in numarr:
        cs.append(train_som(data, i).inertia_)
    return cs
