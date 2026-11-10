import pandas as pd

#Scrape match data from FBref
df_league = pd.read_html('https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures',
                  attrs={"id":"sched_2024-2025_9_1"})[0]

#Remove future matches
df_league['Date'] = pd.to_datetime(df_league['Date'], errors='coerce')
today = pd.to_datetime("today")  # Get the current date
df_league = df_league[df_league['Date'] <= today]  # Keep only past or current matches

#Deletes unnecessary columns permanately, only run once
df_league.drop(columns=["Time", "Attendance", "Venue", "Referee", "Match Report", "Notes", "Day"], inplace=True)

#Split score column into two separte columns
def split_score(df, score_column="score"):
  df["Score"] = df["Score"].str.replace('–', '-', regex=False)
  # Split the score column and handle non-numeric values
  df[['Home Score', 'Away Score']] = df[score_column].str.split('-', expand=True)
  # Convert to numeric, coerce invalid values to NaN, and then fill NaN with 0
  df[['Home Score', 'Away Score']] = df[['Home Score', 'Away Score']].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
  return df

df_league = split_score(df_league, "Score")
df_league = df_league.drop(columns=["Score"])

#Reorder columns
df_league["Match ID"] = df_league.index + 1
#Specifying order of columns
df_league = df_league[["Wk", "Match ID", "Date", "Home", "Home Score", "xG", "Away", "Away Score", "xG.1"]]
# Define column names
columns = ["Wk", "Match ID", "Date", "Home", "Home Score", "xG", "Away", "Away Score", "xG.1"]

##Get stats
#Get recent record from last five matches
def get_last_five_per_match(df):
  # Create separate DataFrames for home and away teams
  home_df = df[['Wk', 'Match ID', 'Date', 'Home', 'Home Score', 'Away Score', 'xG', 'xG.1']].copy()
  away_df = df[['Wk', 'Match ID', 'Date', 'Away', 'Away Score', 'Home Score', 'xG.1', 'xG']].copy()

  # Rename columns for consistency
  home_df.columns = ['Wk', 'Match ID', 'Date', 'Team', 'Goals For', 'Goals Against', 'xG', 'xGA']
  away_df.columns = ['Wk', 'Match ID', 'Date', 'Team', 'Goals For', 'Goals Against', 'xG', 'xGA']

  home_df["Home Adv"] = 1
  away_df["Home Adv"] = 0

  # Combine into a single table 
  combined_df = pd.concat([home_df, away_df])

  # Convert scores to integers
  combined_df['Goals For'] = combined_df['Goals For'].astype(int)
  combined_df['Goals Against'] = combined_df['Goals Against'].astype(int)

  # Function to encode match result
  def encode_result(row):
    if row['Goals For'] > row['Goals Against']:
      return 1  # Win
    elif row['Goals For'] < row['Goals Against']:
      return 0  # Loss
    else:
      return 0.5  # Draw

  # Apply result encoding
  combined_df['Outcome'] = combined_df.apply(encode_result, axis=1)

  # Sort by team and date
  combined_df = combined_df.sort_values(by=['Team', 'Date'])

  # Dictionary to store match history for each team
  team_history = {}

  # Store recent records for each match
  last_five_per_match = []

  for __, row in combined_df.iterrows():
    team = row['Team']  # Get team name

    # Get previous matches before this one
    previous_matches = team_history.get(team, [])[:]

    # Store the last 5 games before this match
    last_five = previous_matches[-5:]

    # Pad with 0 if fewer than 5 exist
    while len(last_five) < 5:
        last_five.insert(0, 0.0)

    # Append this match's recent record
    last_five_per_match.append(sum(last_five))

    # Add this match to the team's history **after** assigning the record
    team_history.setdefault(team, []).append(row['Outcome'])

  # Assign the calculated recent records to the DataFrame
  combined_df['Recent Record'] = last_five_per_match

  return combined_df

recent_record_df = get_last_five_per_match(df_league)

#Get goal and xG stats from before match
def get_more_stats(df):
  team_stats = {}

  # Store past stat for each match
  matches_played = []
  tot_goals_for = []
  tot_goals_against = []
  avg_goals_for = []
  avg_goals_against = []
  goal_differential = []
  average_xG = []
  average_xGA = []
  goal_differential = []

  for __, row in df.iterrows():
    team = row['Team']  # Get team name

    if team in team_stats:
      stats = team_stats[team]
    else:
      stats = {
          'matches_played': 0,
          'goals_for': 0,
          'goals_against': 0,
          'xG': 0,
          'xGA': 0,
      }

    # Compute averages (avoid division by zero)
    avg_goals_for_current = stats['goals_for'] / stats['matches_played'] if stats['matches_played'] > 0 else 0
    avg_goals_against_current = stats['goals_against'] / stats['matches_played'] if stats['matches_played'] > 0 else 0
    average_xG_current = stats['xG'] / stats['matches_played'] if stats['matches_played'] > 0 else 0
    average_xGA_current = stats['xGA'] / stats['matches_played'] if stats['matches_played'] > 0 else 0
    goal_differential_current = stats['goals_for'] - stats['goals_against']

    # Store the past stats for this match
    tot_goals_for.append(stats['goals_for'])
    tot_goals_against.append(stats['goals_against'])
    matches_played.append(stats['matches_played'])
    avg_goals_for.append(avg_goals_for_current)
    avg_goals_against.append(avg_goals_against_current)
    average_xG.append(average_xG_current)
    average_xGA.append(average_xGA_current)
    goal_differential.append(goal_differential_current)

    # Update team stats **after** storing previous stats
    stats['matches_played'] += 1
    stats['goals_for'] += row['Goals For']
    stats['goals_against'] += row['Goals Against']
    stats['xG'] += row['xG']
    stats['xGA'] += row['xGA']
    team_stats[team] = stats  # Save updated stats

  # Assign calculated columns to DataFrame
  df['MP'] = matches_played
  df['Tot GF'] = tot_goals_for
  df['Tot GA'] = tot_goals_against
  df['Avg GF'] = avg_goals_for
  df['Avg GA'] = avg_goals_against
  df['xG'] = average_xG
  df['xGA'] = average_xGA
  df['GD'] = goal_differential

  return df

more_stats_df = get_more_stats(recent_record_df)
more_stats_df = more_stats_df.drop(columns=["Date", "Goals Against"])

#Calculate points before each match
def calc_points_before_match(df):
  # Calculate points before each match
  team_points = {}
  points_before_match = []

  for index, row in df.iterrows():
      team = row['Team']
      if team not in team_points:
          team_points[team] = 0
      points_before_match.append(team_points[team])

      # Update points based on the match outcome
      if row["Outcome"] == 1:
          team_points[team] += 3
      elif row["Outcome"] == 0.5:
          team_points[team] += 1

  df["Points Before Match"] = points_before_match
  return df

points_df = calc_points_before_match(more_stats_df)

#Calculate table standings
def calculate_table_standing(df):
    # Create a copy to avoid modifying the original DataFrame
    df_with_rank = df.copy()
    
    # Group by match week and team, then sort for ranking
    table_standings = []
    for week in df_with_rank['Wk'].unique():
        week_df = df_with_rank[df_with_rank['Wk'] == week]

        # Calculate points for the current week
        team_points_week = {}
        for index, row in week_df.iterrows():
            team = row['Team']
            if team not in team_points_week:
                team_points_week[team] = row['Points Before Match']
            
        # Create a temporary DataFrame for ranking within the week
        rank_df = pd.DataFrame({'Team': list(team_points_week.keys()), 'Points': list(team_points_week.values())})
        
        # Add other ranking criteria (GD, GA)
        # Merge the other required columns from the original dataframe
        rank_df = pd.merge(rank_df, week_df[['Team', 'GD', 'Tot GA']], on='Team', how='left').drop_duplicates(subset=['Team'])

        # Sort by multiple columns
        rank_df = rank_df.sort_values(by=['Points', 'GD', 'Tot GA'], ascending=[False, False, True])

        # Assign ranks based on the combined sorting
        rank_df['Rank'] = range(1, len(rank_df) + 1)
        
        # Merge back the ranks to the original dataframe for that week
        week_df = pd.merge(week_df, rank_df[['Team','Rank']], on='Team', how='left')
        
        table_standings.append(week_df)

    # Concatenate all week dataframes back into a single dataframe
    final_table = pd.concat(table_standings)
    return final_table

rank_df = calculate_table_standing(points_df)
rank_df = rank_df.drop(columns=["Wk", "Outcome", "MP"])

#Specifying order of columns
rank_df = rank_df[["Match ID", "Rank", "Points Before Match", "Team", "Home Adv", "xG", "xGA", "Tot GF", "Tot GA", "Avg GF", "Avg GA", "GD", "Recent Record", "Goals For"]]
rank_df.rename(columns={'Goals For': 'Fulltime GF'}, inplace=True)

##Organize data into one dataframe
# Split the dataframe into two: one for home teams and one for away teams
df_home = rank_df[rank_df['Home Adv'] == 1].copy()  # Contains rows where Home Adv = 1
df_away = rank_df[rank_df['Home Adv'] == 0].copy()  # Contains rows where Home Adv = 0

# Merge the home and away dataframes
merged_df = pd.merge(df_home, df_away, on='Match ID', suffixes=('_A', '_B'))

# Select and rename columns
merged_df = merged_df[[
    'Match ID', 'Rank_A', 'Points Before Match_A', 'Team_A', 'xG_A', 'xGA_A',
    'Tot GF_A', 'Tot GA_A', 'Avg GF_A', 'Avg GA_A', 'GD_A', 'Recent Record_A', 'Fulltime GF_A',
    'Rank_B', 'Points Before Match_B', 'Team_B', 'xG_B', 'xGA_B', 'Tot GF_B',
    'Tot GA_B', 'Avg GF_B', 'Avg GA_B', 'GD_B', 'Recent Record_B', 'Fulltime GF_B'
]]

merged_df = merged_df.sort_values(by=["Match ID"])

# Save as CSV
merged_df.to_csv("EPL_prematch_stats.csv", index=False)