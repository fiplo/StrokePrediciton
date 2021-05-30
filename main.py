from shared import *
from SOM.main import train_som


def main():
    data = read_file('./data/healthcare-dataset-stroke-data.csv')
    train_som(data)


if __name__ == "__main__":
    main()
