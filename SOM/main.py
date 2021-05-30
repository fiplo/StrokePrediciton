from minisom import MiniSom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_som(data):
    input_data = pd.DataFrame(data).drop(['stroke'], axis=1).to_numpy()
    som_shape = (2, 1)
    som = MiniSom(som_shape[0], som_shape[1], input_data.shape[1], sigma=0.5, learning_rate=.5,
                  neighborhood_function='gaussian', random_seed=0)
    som.pca_weights_init(input_data)
    som.train(input_data, 1000)

    winner_coordinates = np.array([som.winner(x) for x in input_data]).T
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

    for c in np.unique(cluster_index):
        plt.scatter(input_data[cluster_index == c, data.columns.get_loc("avg_glucose_level")],
                    input_data[cluster_index == c, data.columns.get_loc("bmi")], label='cluster=' + str(c), alpha=.7)

    plt.legend()
    plt.show()
