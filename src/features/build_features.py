import pandas as pd

def preprocess_data(df):
    df['Admit_Chance'] = (df['Admit_Chance'] >= 0.8).astype(int)
    df = df.drop(['Serial_No'], axis=1)
    df = pd.get_dummies(df, columns=['University_Rating', 'Research'])
    return df

def prepare_data(df):
    x = df.drop(['Admit_Chance'], axis=1)
    y = df['Admit_Chance']
    return x, y
