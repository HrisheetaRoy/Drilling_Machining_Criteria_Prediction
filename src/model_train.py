import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-5))) * 100


def compute_r(y_true, y_pred):
    r = []
    for i in range(y_true.shape[1]):
        corr_matrix = np.corrcoef(y_true[:, i], y_pred[:, i])
        r.append(corr_matrix[0, 1])  # Pearson's r
    return np.array(r)


def train_models(X: pd.DataFrame, y_log: pd.DataFrame) -> dict:
    # Split log-transformed target data
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

   
    models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=30,
        n_jobs=1
    ),
    'AdaBoost': AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4),
        n_estimators=100,
        learning_rate=0.1,
        random_state=100
    ),
    'KNN': KNeighborsRegressor(
        n_neighbors=5,
        n_jobs=1
    )
}
#     models = {
#     'RandomForest': RandomForestRegressor(
#         n_estimators=300, max_depth=None, min_samples_split=2,
#         random_state=30, n_jobs=1
#     ),
#     'XGBoost': XGBRegressor(
#         objective='reg:squarederror', n_estimators=50, random_state=50,
#         n_jobs=1, verbosity=0
#     ),
#     'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=1)
# } 

    trained_models = {}
    evaluation_results = {}

    for name, model in models.items():
        wrapped_model = MultiOutputRegressor(model)
        wrapped_model.fit(X_train, y_train_log)

        # Predict in log scale
        y_pred_log = wrapped_model.predict(X_test)

        # Inverse transform to original scale with clipping to prevent overflow
        y_pred = np.expm1(np.clip(y_pred_log, 0, 20))
        y_test = np.expm1(np.clip(y_test_log.values, 0, 20))

        # Compute evaluation metrics on original scale
        r = compute_r(y_test, y_pred)
        r2 = r2_score(y_test, y_pred, multioutput='raw_values')
        mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        trained_models[name] = wrapped_model
        evaluation_results[name] = {
            'R': r,
            'R2': r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }

    return {
        'models': trained_models,
        'metrics': evaluation_results,
        'X_test': X_test,
        'y_test': y_test_log  # returned in log-scale (raw), for downstream plotting
}
    
    