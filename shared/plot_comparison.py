import matplotlib.pyplot as plt


def plot_comparison(smo_arr, kmean_arr, clusters, file_name, title, ylabel):
    fig, ax = plt.subplots()
    ax.plot(clusters, smo_arr, label="SOM")
    ax.plot(clusters, kmean_arr, label="K-means")
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel(ylabel)
    plt.savefig(file_name)
