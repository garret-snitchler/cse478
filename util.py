from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from ucimlrepo import fetch_ucirepo
import numpy as np


def load_dataset(name):
    if name == 'iris':
        data = datasets.load_iris()
        X, y = data.data, data.target

    elif name == 'breast_cancer':
        data = datasets.load_breast_cancer()
        X, y = data.data, data.target

    elif name == 'wine_quality':
        wine_quality = fetch_ucirepo(id=186)
        X = wine_quality.data.features.values 
        y = wine_quality.data.targets.values
        y = np.where(y >= 6, 1, 0) # splits into good or bad , 6+ meaning good
    else:
        raise ValueError(f"Dataset '{name}' is not supported.")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y


def evaluate_model(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_scores = []
    y_true = []
    y_pred = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_scores.append(model.score(X_test, y_test))
        y_true.extend(y_test)
        y_pred.extend(y_pred)
    
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
    return np.array(all_scores), conf_matrix