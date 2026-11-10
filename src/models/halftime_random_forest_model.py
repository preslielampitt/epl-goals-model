import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Project folders (this file lives in src/models/, so the project root is 2 levels up)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
IMAGES = ROOT / "docs" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

# Read CSV with statistics
df = pd.read_csv(DATA / "EPL_halftime_2024-25.csv")

## Preparing Data
# Save Features: Two Options
# Option 1: Save all features (better home results, slightly worse away results)
X_home = df.drop(['Fixture ID', 'Home Team', 'Away Team', 'Home Fulltime Goals', 'Away Fulltime Goals'], axis=1)
X_away = df.drop(['Fixture ID', 'Home Team', 'Away Team', 'Home Fulltime Goals', 'Away Fulltime Goals'], axis=1)

# OR Option 2: Save select features (worse home results, slightly better away results)
# NOTE: Option 2 runs second, so it overwrites Option 1. This is the one the model uses.
X_home = df.drop(['Fixture ID', 'Home Team', 'Away Team', 'Home Fulltime Goals', 'Away Fulltime Goals', 'Home Yellow Cards', 'Home Red Cards', 'Home Advantage', 'Away Shots on Goal', 'Away Total Shots', 'Away Red Cards', 'Away Passes %', 'Away expected_goals', 'Away_Home Advantage'], axis=1)
X_away = df.drop(['Fixture ID', 'Home Team', 'Away Team', 'Home Fulltime Goals', 'Away Fulltime Goals', 'Away Yellow Cards', 'Away Red Cards', 'Away_Home Advantage', 'Away Shots on Goal', 'Home Total Shots', 'Home Red Cards', 'Home Passes %', 'Home expected_goals', 'Home Advantage'], axis=1)

# Save labels (fulltime goals) for home and away
y_home = df["Home Fulltime Goals"]
y_away = df["Away Fulltime Goals"]

# Split the data into training and test sets
X_home_train, X_home_test, y_home_train, y_home_test = train_test_split(X_home, y_home, test_size=0.2, random_state=42)
X_away_train, X_away_test, y_away_train, y_away_test = train_test_split(X_away, y_away, test_size=0.2, random_state=42)


## Train the random forest model
home_model = RandomForestRegressor(n_estimators=100, random_state=42)
away_model = RandomForestRegressor(n_estimators=100, random_state=42)
home_model.fit(X_home_train, y_home_train)
away_model.fit(X_away_train, y_away_train)


## Make predictions
home_preds = home_model.predict(X_home_test)
away_preds = away_model.predict(X_away_test)


## Evaluate models performance
def rmse(y_true, y_pred):
    # Square root of MSE puts the error back in goals
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Baseline: always predict the average goals from the training set
home_base = np.full(len(y_home_test), y_home_train.mean())
away_base = np.full(len(y_away_test), y_away_train.mean())

print("Halftime random forest")
print(f"Home RMSE: {rmse(y_home_test, home_preds):.2f}  (baseline {rmse(y_home_test, home_base):.2f})")
print(f"Away RMSE: {rmse(y_away_test, away_preds):.2f}  (baseline {rmse(y_away_test, away_base):.2f})")
print(f"Home MAE: {mean_absolute_error(y_home_test, home_preds):.2f}")
print(f"Away MAE: {mean_absolute_error(y_away_test, away_preds):.2f}")
print(f"Home R²: {r2_score(y_home_test, home_preds):.2f}")
print(f"Away R²: {r2_score(y_away_test, away_preds):.2f}")

# Visualize goal predictions vs. actual (rounded for display only)
results = pd.DataFrame({
    "Predicted Home Goals": home_preds.round(),
    "Actual Home Goals": y_home_test.values,
    "Predicted Away Goals": away_preds.round(),
    "Actual Away Goals": y_away_test.values
})
print(results.head(10))


## Use SHAP to determine importance of features
# Red = high value of that feature; blue = low value of that feature
# Positive SHAP value pushes the predicted score up; negative SHAP value pushes it down
# Zero SHAP = neutral impact for that prediction
for side, model, X_test in [("home", home_model, X_home_test), ("away", away_model, X_away_test)]:
    # TreeExplainer is built for tree models like random forests: fast and exact
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test, check_additivity=False)

    # Summary plot: overall impact of each feature
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(f"Halftime random forest: {side} goals")
    plt.savefig(IMAGES / f"shap_halftime_rf_{side}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Waterfall plot: why the model predicted what it did for one sample game
    sample_index = 0
    shap.plots.waterfall(shap_values[sample_index], show=False)
    plt.savefig(IMAGES / f"shap_halftime_rf_{side}_example_match.png", dpi=200, bbox_inches="tight")
    plt.close()

print(f"SHAP plots saved to {IMAGES}")