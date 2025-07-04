#RF, DT, XGB

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_models(X: pd.DataFrame, y: pd.DataFrame) -> dict:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models that support multi-target regression
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    }

    trained_models = {}
    evaluation_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate metrics for each target
        r2 = r2_score(y_test, y_pred, multioutput='raw_values')
        mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')

        # Store evaluation in dictionary
        evaluation_results[name] = {
            'R2': r2,
            'MAE': mae,
            'MSE': mse
        }

    return {'models': trained_models, 'metrics': evaluation_results}
