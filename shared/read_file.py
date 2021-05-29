import pandas as pd
from sklearn import preprocessing

def read_file(file_name):
    strokeData = pd.read_csv(file_name)
    strokeData = strokeData.reset_index().drop('index', axis=1)
    del strokeData['id']
    strokeData = strokeData.dropna()
    strokeData = strokeData[strokeData.smoking_status != 'Unknown']
    strokeData = strokeData.replace(to_replace=['Yes', 'No'], value=[1, 0])
    strokeData = strokeData.replace(to_replace=['Male', 'Female', 'Other'], value=[1, 0, 0.5])
    return strokeData

def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    column_names_to_normalize = ['age', 'avg_glucose_level', 'bmi']
    x = df[column_names_to_normalize].values
    x_scaled = min_max_scaler.fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df.index)
    df[column_names_to_normalize] = df_temp
    return df

