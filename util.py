from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np


def load_dataset(name):
    if name == 'iris':
        data = datasets.load_iris()
        X, y = data.data, data.target

    elif name == 'breast_cancer':
        data = datasets.load_breast_cancer()
        X, y = data.data, data.target

    elif name == 'wine_quality':
        df = pd.read_csv('winequality-red.csv', sep=';')
        X = df.drop('quality', axis=1).values
        y = df['quality'].values
    else:
        raise ValueError(f"Dataset '{name}' is not supported.")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y