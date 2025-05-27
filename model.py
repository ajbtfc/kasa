import logging
from logging.handlers import RotatingFileHandler
import joblib

from statistics import LinearRegression

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from main import MODEL_LOG_FILE

log_handler = RotatingFileHandler(
    MODEL_LOG_FILE, maxBytes=1024 * 1024, backupCount=5  # 1MB max, keep 5 backups
)
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

def process_data():
    df = pd.read_csv('kasa/sump_regression_dataset.csv')

    # Compute row-to-row difference in duration
    df['duration_diff'] = df['duration_to_next_run_min'].diff()

    # Keep only rows where the change is <= 400, or the first row (which has NaN diff)
    df_filtered = df[(df['duration_diff'].isna()) | (df['duration_diff'] <= 400)].copy()

    # Optionally drop the helper column
    df_filtered.drop(columns='duration_diff', inplace=True)
    df = df_filtered

    moisture_cols = [col for col in df.columns if 'moisture' in col]

    X = df[moisture_cols[-2:]]

    y = df['duration_to_next_run_min']
    return X,y

def train_model(X, y):
    cv = LeaveOneOut()

    # ----Linear Regression----
    # # Model and cross-validation strategy
    # model = LinearRegression()
    #
    # # Get cross-validated predictions
    # y_pred = cross_val_predict(model, X, y, cv=cv)
    #
    # # Calculate metrics
    # mae = mean_absolute_error(y, y_pred)
    # rmse = np.sqrt(mean_squared_error(y, y_pred))
    #
    # # Print results
    # print(model)
    # print("MAE:", mae)
    # print("RMSE:", rmse)
    # model.fit(X, y)
    # for col, coef in zip(moisture_cols[-2:], model.coef_):
    #     print(f"{col}: {coef:.2f}")

    param_grid = {'alpha': np.logspace(-3, 2, 30)}

    # -----RIDGE-----
    ridge = Ridge()

    ridge_cv = GridSearchCV(ridge, param_grid, scoring='neg_mean_absolute_error', cv=cv)
    ridge_cv.fit(X, y)
    logger.info("Best Ridge alpha:", ridge_cv.best_params_['alpha'])
    print("Best Ridge MAE:", -ridge_cv.best_score_)

    # Ridge regression with alpha (regularization strength)
    model = Ridge(alpha=ridge_cv.best_params_['alpha'])  # alpha can be tuned

    # Cross-validated predictions
    y_pred_ridge = cross_val_predict(model, X, y, cv=cv)

    logger.info("Ridge Regression")
    logger.info("MAE:", mean_absolute_error(y, y_pred_ridge))
    logger.info("RMSE:", np.sqrt(mean_squared_error(y, y_pred_ridge)))

    # Access coefficients fitted on full data (fit separately)
    model.fit(X, y)
    logger.info("Coefficients:", model.coef_)
    # -----LASSO-----
    # lasso = Lasso()
    #
    # lasso_cv = GridSearchCV(lasso, param_grid, scoring='neg_mean_absolute_error', cv=cv)
    # lasso_cv.fit(X, y)
    # print("Best Lasso alpha:", lasso_cv.best_params_['alpha'])
    # print("Best Lasso MAE:", -lasso_cv.best_score_)
    #
    # # Similarly for Lasso (L1 regularization)
    # lasso_model = Lasso(alpha=lasso_cv.best_params_['alpha'])  # alpha can be tuned
    #
    # y_pred_lasso = cross_val_predict(lasso_model, X, y, cv=cv)
    #
    # print("\nLasso Regression")
    # print("MAE:", mean_absolute_error(y, y_pred_lasso))
    # print("RMSE:", np.sqrt(mean_squared_error(y, y_pred_lasso)))
    #
    # lasso_model.fit(X, y)
    # print("Coefficients:", lasso_model.coef_)

    # -----RANDOM FOREST-----
    # # Initialize model
    # rf = RandomForestRegressor(random_state=42)
    #
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [None, 3, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }
    #
    # # MAE as the scoring metric
    # scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    #
    # # Leave-One-Out Cross-Validation
    # cv = LeaveOneOut()
    #
    # # Grid Search
    # grid_search = GridSearchCV(rf, param_grid, scoring=scorer, cv=cv, n_jobs=-1)
    # grid_search.fit(X, y)
    #
    # # Best model and results
    # best_rf = grid_search.best_estimator_
    # print("Best Parameters:", grid_search.best_params_)
    #
    # # Predict with best model using LOO CV
    # from sklearn.model_selection import cross_val_predict
    # from sklearn.metrics import mean_squared_error
    #
    # y_pred_rf = cross_val_predict(best_rf, X, y, cv=cv)
    # mae = mean_absolute_error(y, y_pred_rf)
    # rmse = np.sqrt(mean_squared_error(y, y_pred_rf))
    #
    # print("Best Random Forest")
    # print("MAE:", mae)
    # print("RMSE:", rmse)

    # -----GRADIENT BOOSTING -----
    #
    # gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    # y_pred_gb = cross_val_predict(gb, X, y, cv=LeaveOneOut())
    # print("MAE:", mean_absolute_error(y, y_pred_gb))
    # print("RMSE:", np.sqrt(mean_squared_error(y, y_pred_gb)))
    return model


def save_model(model, filePath):
    """
    Save a trained model to disk.

    Parameters:
    - model: the trained model object
    - filePath: path to save the model (e.g., 'models/linear_model.pkl')
    """
    joblib.dump(model, filePath)
    print(f"Model saved to {filePath}")

def load_model(filePath):
    """
    Load a model from disk.

    Parameters:
    - filePath: path to the saved model file

    Returns:
    - Loaded model object
    """
    model = joblib.load(filePath)
    print(f"Model loaded from {filePath}")
    return model

# -----PLOT PREDS-----
# # Collect predictions
# predictions = {
#     'Linear': y_pred,
#     'Ridge': y_pred_ridge,
#     'Lasso': y_pred_lasso,
#     'GradientBoosting': y_pred_gb,
#     'Random Forest': y_pred_rf  # this is already from best_rf with cross_val_predict
# }
#
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
# axs = axs.flatten()
#
# for ax, (name, preds) in zip(axs, predictions.items()):
#     ax.scatter(y, preds, alpha=0.6)
#     ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
#     ax.set_title(name)
#     ax.set_xlabel("Actual Duration")
#     ax.set_ylabel("Predicted Duration")
#     ax.grid(True)
#
# # Hide any unused subplot (e.g., the 6th subplot in a 2x3 grid)
# for i in range(len(predictions), len(axs)):
#     axs[i].axis('off')
#
# plt.tight_layout()
# plt.show()
#
