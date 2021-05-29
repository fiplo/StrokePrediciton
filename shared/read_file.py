import pandas as pd


def read_file(file_name):
    strokeData = pd.read_csv(file_name)
    strokeData = strokeData.reset_index().drop('index', axis=1)
    del strokeData['id']
    strokeData = strokeData.dropna()
    strokeData = strokeData[strokeData.smoking_status != 'Unknown']
    return strokeData
