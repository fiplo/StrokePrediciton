from shared import *
from SOM.main import *
from Kmeans.main import *
import matplotlib.pyplot as plt


def main():
    data = read_file('./data/healthcare-dataset-stroke-data.csv')
    normalized_data = normalize(data)
    two_dimensional_data = normalized_data[['avg_glucose_level', 'bmi']]

    plot_initial_data(two_dimensional_data)

    clusters_amount = range(2,6)
    plt.rcParams["figure.figsize"] = (20, 10)
    input_data = two_dimensional_data.to_numpy()
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

    fig, ax = plt.subplots(2, 2)
    SilhouetteKmeansPlot(two_dimensional_data, 2, ax[0][0])
    SilhouetteKmeansPlot(two_dimensional_data, 3, ax[0][1])
    SilhouetteKmeansPlot(two_dimensional_data, 4, ax[1][0])
    SilhouetteKmeansPlot(two_dimensional_data, 5, ax[1][1])
    fig.set_size_inches(20, 17)
    plt.savefig(f'kmeans-siluetai.png')


    clusters_amount = range(2, 10)
    smo_inertia_list = smo_inertia(input_data, clusters_amount)
    kmeans_inertia_list = kmeansIne(two_dimensional_data, clusters_amount)


    plot_comparison(smo_inertia_list, kmeans_inertia_list, clusters_amount, 'inertia_comparison.png', 'Inertia values comparison', 'Inertia')

    smo_silhouettes_list = smo_silhouette_score(input_data, clusters_amount)
    kmeans_silhouettes_list = kmeanSil(two_dimensional_data, clusters_amount)

    plot_comparison(smo_silhouettes_list, kmeans_silhouettes_list, clusters_amount, 'silhouette_comparison.png', 'Silhouette coefficient values comparison', 'Silhouette coefficient')


if __name__ == "__main__":
    main()
