import matplotlib.pyplot as plt


def plot_initial_data(two_dimensional_data):
    input_data = two_dimensional_data.to_numpy()
    x = input_data[:, 0]
    y = input_data[:, 1]
    plt.scatter(x, y)
    plt.xlabel(two_dimensional_data.columns[0])
    plt.ylabel(two_dimensional_data.columns[1])
    plt.savefig('initial_data.png')
