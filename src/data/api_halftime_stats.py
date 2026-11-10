### The following code creates a CSV file with statistics 
### from the 2024-25 English Premier League season 
### for games that have already occured

import http.client
import json
import pandas as pd
import datetime
import os
from dotenv import load_dotenv
from pathlib import Path

# Project folders (this file lives in src/data/, so the project root is 2 levels up)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

load_dotenv()
API_KEY = os.environ.get("API_FOOTBALL_KEY")
if not API_KEY:
    raise SystemExit("Add your key to .env: API_FOOTBALL_KEY=your_api_key_here")

conn = http.client.HTTPSConnection("v3.football.api-sports.io")
headers = {
    'x-rapidapi-host': "v3.football.api-sports.io",
    'x-rapidapi-key': API_KEY
    }

# Request fixture data for EPL 2024 season
conn.request("GET", "/fixtures?league=39&season=2024", headers=headers)
res = conn.getresponse()
data = res.read()
fixtures_json = json.loads(data.decode("utf-8"))

# Extract fixture IDs for all matches in EPL 2024 season
fixture_ids = [fixture["fixture"]["id"] for fixture in fixtures_json["response"]]

# Ensure there are 380 fixtures ids
# as there are 380 matches in one EPL season
print(f"Fixtures found: {len(fixture_ids)}")

# Save fixture IDs to a local file
with open(DATA / "fixtures_2024.json", "w") as f:
    json.dump(fixture_ids, f)

## Extract statisitics, halftime scores, and fulltime scores for each match that has already occured
# Define the stats to extract
desired_stats = {
    "Shots on Goal", "Total Shots", "Fouls", "Ball Possession",
    "Yellow Cards", "Red Cards", "Passes %", "xG"
}

fixtures_data = []

# Loop through each fixture
for fixture_id in fixture_ids:
    print(f"Processing fixture: {fixture_id}")
    fixture_entry = {"Fixture ID": fixture_id}

    conn = http.client.HTTPSConnection("v3.football.api-sports.io")

    headers = {
        'x-rapidapi-host': "v3.football.api-sports.io",
        'x-rapidapi-key': API_KEY
        } 

    # Get team statistics
    conn.request("GET", f"/fixtures/statistics?fixture={fixture_id}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    stats_json = json.loads(data.decode("utf-8"))

    # Get home and away teams from fixture info
    conn.request("GET", f"/fixtures?id={fixture_id}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    fixture_json = json.loads(data.decode("utf-8"))

    if fixture_json.get("response"): 
        # Excludes future matches from dataframe
        date = fixture_json["response"][0]["fixture"]["date"]
        date = pd.to_datetime(date).tz_localize(None) # Make 'date' tz-naive

        today = pd.to_datetime("today").tz_localize(None) # Make 'today' tz-naive
        if date > today:
            break
          
        fixture_data = fixture_json["response"][0]
        home_team = fixture_data["teams"]["home"]["name"]
        away_team = fixture_data["teams"]["away"]["name"]

        # Store stats for each team
        for team_data in stats_json.get("response", []):
            team_name = team_data["team"]["name"]

            # Determine if the team is home or away
            if team_name == home_team:
                team_side = "Home"
            elif team_name == away_team:
                team_side = "Away"

            # Store stats for the team
            fixture_entry[f"{team_side} Team"] = team_name
            for stat in team_data["statistics"]:
                if stat["type"] in desired_stats:
                    value = stat["value"]

                    # Convert percentages to numeric values
                    if isinstance(value, str) and "%" in value:
                        value = float(value.replace("%", ""))

                    fixture_entry[f"{team_side} {stat['type']}"] = value

        # Get halftime score
        halftime = fixture_data["score"]["halftime"]
        fixture_entry["Home Halftime Goals"] = halftime["home"]
        fixture_entry["Away Halftime Goals"] = halftime["away"]

        # Get fulltime score
        fulltime = fixture_data["score"]["fulltime"]
        fixture_entry["Home Fulltime Goals"] = fulltime["home"]
        fixture_entry["Away Fulltime Goals"] = fulltime["away"]

        # Add to list
        fixtures_data.append(fixture_entry)
    else:
        # Handle cases where 'response' is empty or missing
        # This should be games that haven't occurred yet
        print(f"Skipping fixture ID {fixture_id} due to missing or empty 'response' data.")

# Convert to DataFrame
df_fixtures = pd.DataFrame(fixtures_data)

# Replace NaN with 0
df_fixtures = df_fixtures.fillna(0)

# Checks the length of the dataframe to ensure its not duplicating
# Should be less than 380 as not all games have occured
print(f"Matches collected: {len(df_fixtures)}")

# If too many rows, delete duplicates
df_fixtures = df_fixtures.drop_duplicates(subset=["Fixture ID"])

# Save to CSV 
df_fixtures.to_csv(DATA / "EPL_2024_fulltime_stats.csv", index=False)

print("Script finished successfully.")