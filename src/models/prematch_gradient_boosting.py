import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

# Project folders (this file lives in src/models/, so the project root is 2 levels up)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"

df = pd.read_csv(DATA / "EPL_prematch_stats.csv")

## Preparing Data
# Save Features: Two Options
# Option 1: Save all features
X_home = df.drop(['Match ID', 'Team_A', 'Team_B', 'Fulltime GF_A', 'Fulltime GF_B'], axis=1)
X_away = df.drop(['Match ID', 'Team_A', 'Team_B', 'Fulltime GF_A', 'Fulltime GF_B'], axis=1)

# Option 2: Save select features
# NOTE: Option 2 runs second, so it overwrites Option 1. This is the one the model uses.
X_home = df.drop(['Match ID', 'Team_A', 'Team_B', 'Fulltime GF_A', 'Fulltime GF_B', 'xGA_A', 'Tot GA_A', 'Avg GA_A', 'Rank_B', 'Points Before Match_B', 'xG_B', 'Tot GF_B', 'Avg GF_B', 'Recent Record_B'], axis=1)
X_away = df.drop(['Match ID', 'Team_A', 'Team_B', 'Fulltime GF_A', 'Fulltime GF_B', 'xGA_B', 'Tot GA_B', 'Avg GA_B', 'Rank_A', 'Points Before Match_A', 'xG_A', 'Tot GF_A', 'Avg GF_A', 'Recent Record_A'], axis=1)

# Save labels (fulltime goals) for home and away
y_home = df["Fulltime GF_A"]
y_away = df["Fulltime GF_B"]

# Split the data into training and test sets
X_home_train, X_home_test, y_home_train, y_home_test = train_test_split(X_home, y_home, test_size=0.2, random_state=42)
X_away_train, X_away_test, y_away_train, y_away_test = train_test_split(X_away, y_away, test_size=0.2, random_state=42)


## Select best hyperparameter combos using GridSearch
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Model will minimize MSE
xgb_home = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search_home = GridSearchCV(
    estimator=xgb_home,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',   # minimize MSE
    cv=3,                               # 3-fold cross-validation
    verbose=1,                          # shows progress
    n_jobs=-1                           # use all CPU cores
)
grid_search_home.fit(X_home_train, y_home_train)

# Same for away model
xgb_away = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search_away = GridSearchCV(
    estimator=xgb_away,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)
grid_search_away.fit(X_away_train, y_away_train)


## Use the best model from each grid search
# GridSearchCV already retrains the best settings on the full training set (refit=True by default),
# so best_estimator_ is the final model. It keeps random_state=42, so results repeat run to run.
best_model_home = grid_search_home.best_estimator_
best_model_away = grid_search_away.best_estimator_
print("Best home settings:", grid_search_home.best_params_)
print("Best away settings:", grid_search_away.best_params_)


## Make final predictions
y_home_pred = best_model_home.predict(X_home_test)
y_away_pred = best_model_away.predict(X_away_test)


## Evaluate models performance
def rmse(y_true, y_pred):
    # Square root of MSE puts the error back in goals
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Baseline: always predict the average goals from the training set
home_base = np.full(len(y_home_test), y_home_train.mean())
away_base = np.full(len(y_away_test), y_away_train.mean())

print("Pre-match gradient boosting (XGBoost)")
print(f"Home RMSE: {rmse(y_home_test, y_home_pred):.2f}  (baseline {rmse(y_home_test, home_base):.2f})")
print(f"Away RMSE: {rmse(y_away_test, y_away_pred):.2f}  (baseline {rmse(y_away_test, away_base):.2f})")
print(f"Home MAE: {mean_absolute_error(y_home_test, y_home_pred):.2f}")
print(f"Away MAE: {mean_absolute_error(y_away_test, y_away_pred):.2f}")
print(f"Home R²: {r2_score(y_home_test, y_home_pred):.2f}")
print(f"Away R²: {r2_score(y_away_test, y_away_pred):.2f}")