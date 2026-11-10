import json
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

        # Calculate win probabilities
        total_pred = home_goals_rounded + away_goals_rounded
        if total_pred == 0:
            home_prob = away_prob = 0
        else:
            home_prob = round((home_goals_rounded / total_pred) * 100)
            away_prob = round((away_goals_rounded / total_pred) * 100)

        # Determine match result
        if home_goals_rounded > away_goals_rounded:
            result = f"{row['Home Team']} is predicted to win!"
        elif away_goals_rounded > home_goals_rounded:
            result = f"{row['Away Team']} is predicted to win!"
        else:
            result = "It's predicted to be a tie!"

        # Return detailed result
        output = (
            f"🏟️ Match: {row['Home Team']} vs {row['Away Team']}\n\n"
            f"🔢 Predicted Score:\n"
            f"   - {row['Home Team']}: {home_goals_rounded}\n"
            f"   - {row['Away Team']}: {away_goals_rounded}\n\n"
            f"📊 Win Probabilities:\n"
            f"   - {row['Home Team']}: {home_prob}%\n"
            f"   - {row['Away Team']}: {away_prob}%\n\n"
            f"📣 Result: {result}"
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
    description="Upload a CSV file containing halftime stats to predict final score, win probability, and match outcome.",
    theme="soft"
).launch()  # opens locally; share=True would create a public link