import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):  # Custom MAPE function
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100

def train_knn_model(X: pd.DataFrame, y: pd.DataFrame, n_neighbors=5) -> dict:
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize MultiOutput KNN
    knn = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=n_neighbors))
    knn.fit(X_train, y_train)

    # Predict
    y_pred = knn.predict(X_test)

    # Compute metrics
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5)), axis=0) * 100


    return {
        'model': knn,
        'metrics': {
            'R2': r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        },
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }
