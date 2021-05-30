from shared import *
from SOM.main import *


def main():
    data = read_file('./data/healthcare-dataset-stroke-data.csv')
    normalized_data = normalize(data)
    input_data = normalized_data[['age', 'avg_glucose_level']].to_numpy()

    amount_clusters = [2, 3, 4]

    inertia = smo_inertia(input_data, amount_clusters)
    silhouettes = smo_silhouette_score(input_data, amount_clusters)

if __name__ == "__main__":
    main()
