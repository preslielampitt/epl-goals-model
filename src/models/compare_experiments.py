# Experiments: does the feature set or the amount of data change accuracy?
# Every version of a model is scored on the SAME test matches, so the RMSEs can be compared.
# This script only reports results; it doesn't save models or change anything the app uses.

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

# Project folders (this file lives in src/models/, so the project root is 2 levels up)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


## Helpers
def rmse(y_true, y_pred):
    # Square root of MSE puts the error back in goals
    return np.sqrt(mean_squared_error(y_true, y_pred))

def make_rf():
    # Same settings as the random forest scripts
    return RandomForestRegressor(n_estimators=100, random_state=42)

def make_xgb():
    # Same grid search as the XGBoost scripts (pre-match learning rates)
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    return GridSearchCV(XGBRegressor(objective='reg:squarederror', random_state=42),
                        param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)

def score(label, make_model, X_train, y_train, X_test, y_test):
    # Train one model and compare it to the full-match average baseline on the test matches
    model = make_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    base = np.full(len(y_test), y_train.mean())
    return {**label,
            "Train matches": len(X_train),
            "RMSE": round(rmse(y_test, pred), 3),
            "Baseline: full-match average": round(rmse(y_test, base), 3),
            "R²": round(r2_score(y_test, pred), 3)}


## Experiment 1: halftime random forest, Option 1 (all features) vs Option 2 (selected features)
ht = pd.read_csv(DATA / "EPL_halftime_2024-25.csv")
ht_drop = ['Fixture ID', 'Home Team', 'Away Team', 'Home Fulltime Goals', 'Away Fulltime Goals']

# Copied from halftime_random_forest_model.py
ht_options = {
    "Option 1 (all)": (ht_drop, ht_drop),
    "Option 2 (selected)": (
        ht_drop + ['Home Yellow Cards', 'Home Red Cards', 'Home Advantage', 'Away Shots on Goal', 'Away Total Shots', 'Away Red Cards', 'Away Passes %', 'Away expected_goals', 'Away_Home Advantage'],
        ht_drop + ['Away Yellow Cards', 'Away Red Cards', 'Away_Home Advantage', 'Away Shots on Goal', 'Home Total Shots', 'Home Red Cards', 'Home Passes %', 'Home expected_goals', 'Home Advantage'],
    ),
}

# Same split as the training scripts (same row count + random_state=42 -> same test matches)
ht_train, ht_test = train_test_split(ht, test_size=0.2, random_state=42)

halftime_results = []
for option, (home_drop, away_drop) in ht_options.items():
    for side, drop in [("Home", home_drop), ("Away", away_drop)]:
        target = f"{side} Fulltime Goals"
        halftime_results.append(score(
            {"Features": option, "Side": side},
            make_rf,
            ht_train.drop(columns=drop), ht_train[target],
            ht_test.drop(columns=drop), ht_test[target],
        ))


## Experiment 2: pre-match models, 1 season vs 3 seasons, Option 1 vs Option 2
one = pd.read_csv(DATA / "EPL_prematch_stats.csv")
three = pd.read_csv(DATA / "EPL_prematch_stats_3_seasons.csv")

# The 3-season file adds a season suffix to team names (e.g. "Fulham_24"); split it off
for col in ["Team_A", "Team_B"]:
    three["Season"] = three[col].str[-2:]
    three[col] = three[col].str[:-3]

# Test matches = the 2024-25 test matches the pre-match scripts use
one_train, one_test = train_test_split(one, test_size=0.2, random_state=42)
test_keys = set(zip(one_test["Team_A"], one_test["Team_B"]))

# Find those same matches in the 3-season file; everything else becomes training data
in_test_keys = pd.Series([k in test_keys for k in zip(three["Team_A"], three["Team_B"])], index=three.index)
is_test = (three["Season"] == "24") & in_test_keys
assert is_test.sum() == len(one_test), "Some test matches weren't found in the 3-season file"
three_test = three[is_test].set_index(["Team_A", "Team_B"]).loc[list(zip(one_test["Team_A"], one_test["Team_B"]))].reset_index()
three_train = three[~is_test]

pm_drop = ['Match ID', 'Team_A', 'Team_B', 'Fulltime GF_A', 'Fulltime GF_B', 'Season']

# Copied from prematch_random_forest_model.py
pm_options = {
    "Option 1 (all)": (pm_drop, pm_drop),
    "Option 2 (selected)": (
        pm_drop + ['xGA_A', 'Tot GA_A', 'Avg GA_A', 'Rank_B', 'Points Before Match_B', 'xG_B', 'Tot GF_B', 'Avg GF_B', 'Recent Record_B'],
        pm_drop + ['xGA_B', 'Tot GA_B', 'Avg GA_B', 'Rank_A', 'Points Before Match_A', 'xG_A', 'Tot GF_A', 'Avg GF_A', 'Recent Record_A'],
    ),
}

prematch_results = []
for data_label, train, test in [("1 season", one_train, one_test), ("3 seasons", three_train, three_test)]:
    for option, (home_drop, away_drop) in pm_options.items():
        for side, drop, target in [("Home", home_drop, "Fulltime GF_A"), ("Away", away_drop, "Fulltime GF_B")]:
            for model_name, make_model in [("Random forest", make_rf), ("XGBoost", make_xgb)]:
                prematch_results.append(score(
                    {"Data": data_label, "Features": option, "Side": side, "Model": model_name},
                    make_model,
                    train.drop(columns=drop, errors="ignore"), train[target],
                    test.drop(columns=drop, errors="ignore"), test[target],
                ))


## Print results
pd.set_option("display.width", 200)
print("\nHalftime random forest: Option 1 vs Option 2 (same 68 test matches)")
print(pd.DataFrame(halftime_results).to_string(index=False))
print("\nPre-match: 1 season vs 3 seasons (same 2024-25 test matches)")
print(pd.DataFrame(prematch_results).sort_values(["Side", "Model", "Data", "Features"]).to_string(index=False))
