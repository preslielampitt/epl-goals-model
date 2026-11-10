import json
import math
import joblib
import pandas as pd
import gradio as gr
from pathlib import Path

# Project folders (this file lives in app/, so the project root is 1 level up)
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

## Load the ML home and away models + feature columns
# Load models (created by src/models/halftime_gradient_boosting.py)
best_model_home = joblib.load(MODELS / "best_model_home.pkl")
best_model_away = joblib.load(MODELS / "best_model_away.pkl")

# Load saved feature columns
with open(MODELS / "feature_home_columns.json", "r") as f:
    feature_home_columns = json.load(f)
with open(MODELS / "feature_away_columns.json", "r") as f:
    feature_away_columns = json.load(f)

## Turn predicted goals into win / draw / loss probabilities
def poisson_pmf(k, lam):
    # Chance of exactly k goals when lam goals are expected
    return math.exp(-lam) * lam**k / math.factorial(k)

def outcome_probabilities(home_goals, away_goals, home_ht, away_ht, max_goals=10):
    # Expected second-half goals (floor at 0.05 in case the model predicts below the halftime score)
    lam_home = max(home_goals - home_ht, 0.05)
    lam_away = max(away_goals - away_ht, 0.05)
    home_win = draw = away_win = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_pmf(h, lam_home) * poisson_pmf(a, lam_away)
            if home_ht + h > away_ht + a:
                home_win += p
            elif home_ht + h == away_ht + a:
                draw += p
            else:
                away_win += p
    return home_win, draw, away_win

def predict_scores_from_csv(file):
    try:
        df = pd.read_csv(file.name)

        # Validate required columns
        missing_home = [col for col in feature_home_columns if col not in df.columns]
        missing_away = [col for col in feature_away_columns if col not in df.columns]
        if missing_home:
            return f"Missing home columns: {', '.join(missing_home)}"
        if missing_away:
            return f"Missing away columns: {', '.join(missing_away)}"
        if 'Home Team' not in df.columns or 'Away Team' not in df.columns:
            return "CSV must include 'Home Team' and 'Away Team' columns."

        # Save first row
        row = df.iloc[0]
        # Prepare features for prediction
        X_home_input = df[feature_home_columns].iloc[[0]]
        X_away_input = df[feature_away_columns].iloc[[0]]

        # Predict
        home_goals = best_model_home.predict(X_home_input)[0]
        away_goals = best_model_away.predict(X_away_input)[0]

        # Round predictions for display
        home_goals_rounded = round(home_goals)
        away_goals_rounded = round(away_goals)

        # Estimated win / draw / loss probabilities (Poisson on second-half goals)
        home_win, draw, away_win = outcome_probabilities(
            home_goals, away_goals, row['Home Halftime Goals'], row['Away Halftime Goals']
        )

        # Most likely outcome, from the probabilities (not the rounded score)
        outcomes = {
            f"{row['Home Team']} win": home_win,
            "Draw": draw,
            f"{row['Away Team']} win": away_win,
        }
        result = max(outcomes, key=outcomes.get)

        # Return detailed result
        output = (
            f"🏟️ Match: {row['Home Team']} vs {row['Away Team']}\n\n"
            f"🔢 Predicted Score:\n"
            f"   - {row['Home Team']}: {home_goals_rounded}\n"
            f"   - {row['Away Team']}: {away_goals_rounded}\n"
            f"   (Expected goals: {row['Home Team']} {home_goals:.1f}, {row['Away Team']} {away_goals:.1f})\n\n"
            f"📊 Win Probabilities (estimated):\n"
            f"   - {row['Home Team']}: {home_win:.0%}\n"
            f"   - Draw: {draw:.0%}\n"
            f"   - {row['Away Team']}: {away_win:.0%}\n\n"
            f"📣 Most likely result: {result} ({outcomes[result]:.0%})"
        )
        return output

    except Exception as e:
        return f"❌ Error: {str(e)}"


## Simple Gradio Interface to display predictions
gr.Interface(
    fn=predict_scores_from_csv,
    inputs=gr.File(label="Upload CSV with Halftime Stats"),
    outputs=gr.Textbox(label="Predicted Outcome"),
    title="⚽ Soccer Engine",
    description="Upload a CSV file containing halftime stats to predict the final score and win/draw/loss probabilities.",
    theme="soft"
).launch()  # opens locally; share=True would create a public link