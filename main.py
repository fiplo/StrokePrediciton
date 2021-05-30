from shared import *
from SOM.main import SOM


def main():
    data = read_file('./data/healthcare-dataset-stroke-data.csv')
    SOM(data)


if __name__ == "__main__":
    main()
