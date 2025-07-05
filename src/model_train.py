import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_models(X: pd.DataFrame, y: pd.DataFrame) -> dict:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    }

    trained_models = {}
    evaluation_results = {}

    ensemble_model = VotingRegressor([
        ('rf', models['RandomForest']),
        ('xgb', models['XGBoost'])
    ])
    ensemble_model.fit(X_train, y_train)
    y_pred_ensemble = ensemble_model.predict(X_test)

    r2 = r2_score(y_test, y_pred_ensemble, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred_ensemble, multioutput='raw_values')
    mse = mean_squared_error(y_test, y_pred_ensemble, multioutput='raw_values')

    trained_models['Ensemble'] = ensemble_model
    evaluation_results['Ensemble'] = {
        'R2': r2,
        'MAE': mae,
        'MSE': mse
        }







    return {
        'models': trained_models,
        'metrics': evaluation_results,
        'X_test': X_test,
        'y_test': y_test
    }
