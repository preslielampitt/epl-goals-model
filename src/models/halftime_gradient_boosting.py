import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

# Project folders (this file lives in src/models/, so the project root is 2 levels up)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
IMAGES = ROOT / "docs" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "EPL_halftime_2024-25.csv")

## Engineered features (computed from first-half stats; +1 avoids dividing by zero)
# Keep in sync with the other halftime script
df['Home Shot Quality'] = (df['Home Shots on Goal']) / (df['Home Total Shots'] + 1)
df['Away Shot Quality'] = (df['Away Shots on Goal']) / (df['Away Total Shots'] + 1)
df['Home Opp Defense'] = (df['Home Halftime Goals'] + 1) / (df['Home Shots on Goal'] + 1)
df['Away Opp Defense'] = (df['Away Halftime Goals'] + 1) / (df['Away Shots on Goal'] + 1)
df['Home Chance Creation'] = (df['Home Ball Possession'] / 100) * (df['Home Shots on Goal'] + 1)
df['Away Chance Creation'] = (df['Away Ball Possession'] / 100) * (df['Away Shots on Goal'] + 1)

## Preparing Data
# Save Features
# Common columns to drop
common_drop = ['Fixture ID', 'Home Team', 'Away Team', 'Home Fulltime Goals', 'Away Fulltime Goals']

# Additional columns to drop for selected features
home_extra_drop = [
    'Home Yellow Cards', 'Home Red Cards', 'Home Advantage',
    'Home Total Shots', 'Home Ball Possession',
    'Away Shots on Goal', 'Away Total Shots', 'Away Red Cards',
    'Away Passes %', 'Away_Home Advantage', 'Away Yellow Cards',
    'Away Ball Possession', 'Away expected_goals', 'Away Opp Defense'
]

away_extra_drop = [
    'Away Yellow Cards', 'Away Red Cards', 'Away_Home Advantage',
    'Away Total Shots', 'Away Ball Possession',
    'Home Shots on Goal', 'Home Total Shots', 'Home Red Cards',
    'Home Passes %', 'Home Advantage', 'Home Yellow Cards',
    'Home Ball Possession', 'Home expected_goals', 'Home Opp Defense'
]

# Option 1: Save most features for both home and away
def save_most_features(df):
    return df.drop(columns=common_drop)

# Option 2: Save select features each for home and away
def save_select_home_features(df):
    return df.drop(columns=common_drop + home_extra_drop)

def save_select_away_features(df):
    return df.drop(columns=common_drop + away_extra_drop)

# Save variables (this model uses Option 2)
X_home = save_select_home_features(df)
X_away = save_select_away_features(df)

# Save feature column lists (the UI uses these to build inputs in the right order)
with open(MODELS / "feature_home_columns.json", "w") as f:
    json.dump(X_home.columns.tolist(), f)
with open(MODELS / "feature_away_columns.json", "w") as f:
    json.dump(X_away.columns.tolist(), f)

# Save labels (fulltime goals) for home and away
y_home = df["Home Fulltime Goals"]
y_away = df["Away Fulltime Goals"]

# Split the data into training and test sets
X_home_train, X_home_test, y_home_train, y_home_test = train_test_split(X_home, y_home, test_size=0.2, random_state=42)
X_away_train, X_away_test, y_away_train, y_away_test = train_test_split(X_away, y_away, test_size=0.2, random_state=42)


## Select best hyperparameter combos using GridSearch
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.05, 0.1, 0.2],
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

# Save both models (the UI loads these)
joblib.dump(best_model_home, MODELS / "best_model_home.pkl")
joblib.dump(best_model_away, MODELS / "best_model_away.pkl")


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

# Fairer baseline: halftime goals + average second-half goals from the training set
home_2nd_half = (y_home_train - X_home_train["Home Halftime Goals"]).mean()
away_2nd_half = (y_away_train - X_away_train["Away Halftime Goals"]).mean()
home_ht_base = X_home_test["Home Halftime Goals"] + home_2nd_half
away_ht_base = X_away_test["Away Halftime Goals"] + away_2nd_half

print("Halftime gradient boosting (XGBoost)")
print(f"Home RMSE: {rmse(y_home_test, y_home_pred):.2f}  (baseline {rmse(y_home_test, home_base):.2f}, halftime baseline {rmse(y_home_test, home_ht_base):.2f})")
print(f"Away RMSE: {rmse(y_away_test, y_away_pred):.2f}  (baseline {rmse(y_away_test, away_base):.2f}, halftime baseline {rmse(y_away_test, away_ht_base):.2f})")
print(f"Home MAE: {mean_absolute_error(y_home_test, y_home_pred):.2f}")
print(f"Away MAE: {mean_absolute_error(y_away_test, y_away_pred):.2f}")
print(f"Home R²: {r2_score(y_home_test, y_home_pred):.2f}")
print(f"Away R²: {r2_score(y_away_test, y_away_pred):.2f}")


## SHAP waterfall: how the app's models predicted the demo match
# The first test match is Ipswich vs Nottingham Forest, the same match as data/sample_halftime_input.csv
for side, model, X_test in [("home", best_model_home, X_home_test), ("away", best_model_away, X_away_test)]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test.iloc[[0]], check_additivity=False)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig(IMAGES / f"shap_halftime_xgb_{side}_demo_match.png", dpi=200, bbox_inches="tight")
    plt.close()

print(f"Demo-match SHAP plots saved to {IMAGES}")
