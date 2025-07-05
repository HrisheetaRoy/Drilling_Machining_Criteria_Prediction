from sklearn.model_selection import GridSearchCV

def tune_random_forest(X, y):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
    }

    grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_, grid.cv_results_
