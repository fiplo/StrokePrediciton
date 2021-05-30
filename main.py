from shared import *
from SOM.main import *
from Kmeans.main import *
import matplotlib.pyplot as plt


def main():
    data = read_file('./data/healthcare-dataset-stroke-data.csv')
    normalized_data = normalize(data)
    two_dimensional_data = normalized_data[['age', 'avg_glucose_level']]

    clusters_amount = [2,3,4,5]
    plt.rcParams["figure.figsize"] = (20, 10)

    for cluster in clusters_amount:
        labels = returnKmeanClusters(two_dimensional_data, cluster).fit_predict(two_dimensional_data)
        fig, axs = plt.subplots(nrows=1, ncols=2)
        kmeans = returnKmeanClusters(two_dimensional_data, cluster)
        plotKmeans(two_dimensional_data, labels, axs[0])
        som, predictions = plot_som(two_dimensional_data, cluster, axs[1])
        fig.suptitle(f'{cluster} clusters')
        plt.savefig(f'graph_{cluster}_clusters.png', dpi=100)
        input_data = two_dimensional_data.to_numpy()
        print(
            f"{cluster} clusters: \n   Kmeans:\n     Innertia: {kmeans.inertia_}\n     Silhouette: {metrics.silhouette_score(two_dimensional_data, kmeans.labels_)}\n   SOM:\n     Innertia: {som.inertia_}\n     Silhouette: {metrics.silhouette_score(input_data, predictions)}")

    inertia = smo_inertia(input_data, clusters_amount)
    silhouettes = smo_silhouette_score(input_data, clusters_amount)


if __name__ == "__main__":
    main()
