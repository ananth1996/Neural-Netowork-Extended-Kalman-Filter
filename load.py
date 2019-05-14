import pandas as pd


def load_abalone_data():
    columns = ['sex','length','diameter','height','weight','meat_weight','viscera_weight','shell_weight','rings']
    df = pd.read_csv('Data/abalone.data',header=None,names=columns)
    df['sex']=df.sex.astype('category').cat.codes
    y = df.pop('rings').values
    X = df.values
    return X,y

def load_bikes_data():
    df = pd.read_csv('Data/Bikes_hour.csv',index_col='instant')
    df = df.drop(columns='dteday')
    y = df.pop('cnt').values
    X = df.values
    return X,y

def load_wine_data():
    df = pd.read_csv('Data/winequality-white.csv',sep=';')
    y = df.pop('quality').values
    X = df.values
    return X,y