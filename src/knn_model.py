import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100

def train_knn_model(X: pd.DataFrame, y: pd.DataFrame, n_neighbors=5) -> dict:
    X_train, X_test, y_train, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=n_neighbors))
    knn.fit(X_train, y_train)

    y_pred_log = knn.predict(X_test)
    y_pred = np.expm1(np.clip(y_pred_log, 0, 20))
    y_test = np.expm1(np.clip(y_test_log.values, 0, 20))

    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    r = np.array([
        np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
        for i in range(y_test.shape[1])
    ])

    return {
        'model': knn,
        'metrics': {
            'R': r,
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
