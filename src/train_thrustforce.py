import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-10, None))) * 100

def train_thrustforce_models(X, y_flank):
    y_log = np.log1p(y_flank)
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    y_test_actual = np.expm1(y_test)

    def evaluate(y_true, y_pred):
        return {
            'R': (pearsonr(y_true, y_pred)[0]),
            'R2': (r2_score(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred)
        }

    # 🔹 Random Forest (Tuned)
    rf_model = RandomForestRegressor(
        n_estimators=10,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=2,
        random_state=70,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_y_pred = np.expm1(rf_model.predict(X_test))
    rf_metrics = evaluate(y_test_actual, rf_y_pred)

    # 🔹 XGBoost (Tuned)
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_y_pred = np.expm1(xgb_model.predict(X_test))
    xgb_metrics = evaluate(y_test_actual, xgb_y_pred)

    # 🔹 AdaBoost (Tuned)
    ada_model = AdaBoostRegressor(
        n_estimators=300,
        learning_rate=0.5,
        random_state=42
    )
    ada_model.fit(X_train, y_train)
    ada_y_pred = np.expm1(ada_model.predict(X_test))
    ada_metrics = evaluate(y_test_actual, ada_y_pred)

    return {
        'RandomForest': {
            'model': rf_model,
            'metrics': rf_metrics
        },
        'XGBoost': {
            'model': xgb_model,
            'metrics': xgb_metrics
        },
        'AdaBoost': {
            'model': ada_model,
            'metrics': ada_metrics
        }
    }
