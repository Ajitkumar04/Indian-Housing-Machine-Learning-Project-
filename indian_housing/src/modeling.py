from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "mse": mse, "rmse": rmse, "r2": float(r2)}


def train_baselines(X_train, X_test, y_train, y_test) -> Tuple[Dict, Dict, object]:
    """Train baseline models and return their metrics and best model by RMSE.

    Returns: (metrics_dict, trained_models, best_model)
    """
    results = {}
    models = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_lr = lr.predict(X_test)
    results['linear_regression'] = evaluate_model(y_test, y_lr)
    models['linear_regression'] = lr

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)
    results['random_forest'] = evaluate_model(y_test, y_rf)
    models['random_forest'] = rf

    # choose best by RMSE
    best_key = min(results.keys(), key=lambda k: results[k]['rmse'])
    best_model = models[best_key]
    return results, models, best_model


def save_model(model, path: str = "models/best_model.pkl") -> str:
    joblib.dump(model, path)
    return path


__all__ = ["evaluate_model", "train_baselines", "save_model"]
