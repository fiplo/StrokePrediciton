from shared import *
from SOM.main import *


def main():
    data = read_file('./data/healthcare-dataset-stroke-data.csv')
    normalized_data = normalize(data)
    input_data = normalized_data[['age', 'avg_glucose_level']].to_numpy()

    intertia = smo_inertia(input_data, [2, 3, 4])
    plot_som(input_data, 3)
    print(intertia)


if __name__ == "__main__":
    main()
