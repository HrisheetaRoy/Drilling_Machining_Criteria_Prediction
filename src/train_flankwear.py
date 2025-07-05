import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def train_flankwear_model(X, y_flank):
    # Log transform
    y_log = np.log1p(y_flank)

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }

    model = GridSearchCV(
        XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=params,
        scoring='r2',
        cv=3,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    return {
        'model': model.best_estimator_,
        'params': model.best_params_,
        'metrics': {
            'R2': r2_score(y_test_actual, y_pred),
            'MAE': mean_absolute_error(y_test_actual, y_pred),
            'MSE': mean_squared_error(y_test_actual, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_actual, y_pred))

        }
    }
