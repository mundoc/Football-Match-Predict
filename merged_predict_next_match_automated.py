# By Edmundo Cuadra

import pandas as pd # For data manipulation
import numpy as np # For data computation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import os
import requests
import math
from collections import defaultdict
from datetime import datetime
import streamlit as st
import plotly.graph_objs as go
import bs4
from bs4 import BeautifulSoup
import altair as alt


# Ignore all warnings
warnings.filterwarnings("ignore")

# Read the CSV file into a Pandas DataFrame.
matches_old = pd.read_csv('light_leagues_data.csv')

# Get data from new seasons
matches_new_liga = pd.read_csv('https://www.football-data.co.uk/mmz4281/2324/SP1.csv')
matches_new_prem = pd.read_csv('https://www.football-data.co.uk/mmz4281/2324/E0.csv')

# Concatenate the DataFrames one on top of the other
leagues = pd.concat([matches_new_liga, matches_new_prem], axis=0)
matches = pd.concat([matches_old, leagues], axis=0)

# Reset the index of the combined DataFrame, if needed
matches.reset_index(drop=True, inplace=True)

# Drop rows where 'Date' is NaN
matches = matches.dropna(subset=['Date'])

matches['Date'] = pd.to_datetime(matches['Date'], format = 'mixed')

# Create the 'Season' column
matches['Season'] = np.where(matches['Date'].dt.month >= 8,
                             matches['Date'].dt.year,
                             matches['Date'].dt.year - 1)

# Convert the 'Season' column to a string in the 'YYYY-YY' format
matches['Season'] = matches['Season'].astype(str) + '-' + (matches['Season'] + 1).astype(str).str[-2:]


# Create new rows (with the upcoming match)

def get_next_matches(url, league_id, season='2023-24', num_matches=10):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the relevant elements containing match data
    matches = soup.find_all('div', class_='fixres__item')
    
    # Extract the date from the h4 tag
    match_date_tag = soup.find('h4', class_='fixres__header2')
    if match_date_tag:
        match_date_str = match_date_tag.text.strip()

        # Use regex to extract the day and month part
        date_pattern = re.compile(r'\b(\d{1,2})(st|nd|rd|th)?\s+(\w+)\b')
        match = date_pattern.search(match_date_str)
        if match:
            day = match.group(1)
            month = match.group(3)
            match_date_str = f"{day} {month}"

            # Convert to datetime object
            try:
                match_date = datetime.strptime(match_date_str, "%d %B")
            except ValueError:
                print(f"Error parsing date: {match_date_str}")
                return []
        else:
            print("No valid date found")
            return []
    else:
        print("Date header not found")
        return []

    current_year = datetime.now().year

    next_matches_data = []

    # Counter for the number of matches found
    matches_found = 0

    for match in matches:  # Iterate over all matches
        try:
            # Extract data for each match
            home_team = match.find('span', class_='swap-text__target').text.strip()
            away_team = match.find_all('span', class_='swap-text__target')[1].text.strip()
            match_time_str = match.find('span', class_='matches__date').text.strip()

            # Extract and convert match time to a datetime object
            match_time = datetime.strptime(match_time_str, "%H:%M")

            # Construct the full match datetime
            match_datetime = datetime(current_year, match_date.month, match_date.day, match_time.hour, match_time.minute)

            match_data = {
                'Div': league_id.upper(),
                'Date': match_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'Season': season,
                'HomeTeam': home_team,
                'AwayTeam': away_team,
            }

            next_matches_data.append(match_data)
            matches_found += 1

            if matches_found >= num_matches:
                break  # Stop iteration after collecting the desired number of matches
        except Exception as e:
            print(f"Error processing match data: {e}")

    return next_matches_data

# Use the function to get data for different leagues
spanish_league_url = 'https://www.skysports.com/la-liga-fixtures'
epl_url = 'https://www.skysports.com/premier-league-fixtures'

next_matches_data = get_next_matches(spanish_league_url, 'soccer_spain_la_liga', num_matches=10)
next_matches_data_epl = get_next_matches(epl_url, 'soccer_epl', num_matches=10)

# Convert the list of dictionaries into a DataFrame
next_matches_df = pd.DataFrame(next_matches_data)
next_matches_df['Date'] = pd.to_datetime(next_matches_df['Date'], format='%Y-%m-%d %H:%M:%S')
next_matches_df['Div'] = next_matches_df['Div'].replace('SOCCER_SPAIN_LA_LIGA', 'SP1')
next_matches_pl_df = pd.DataFrame(next_matches_data_epl)
next_matches_pl_df['Date'] = pd.to_datetime(next_matches_pl_df['Date'], format='%Y-%m-%d %H:%M:%S')
next_matches_pl_df['Div'] = next_matches_pl_df['Div'].replace('SOCCER_EPL', 'E0')

# Append the DataFrame for multiple matches to the existing 'matches' DataFrame
matches = pd.concat([matches, next_matches_df], ignore_index=True)
matches = pd.concat([matches, next_matches_pl_df], ignore_index=True)

matches = matches.replace({
    'Villareal': 'Villarreal',
    'Celta Vigo': 'Celta',
    'CA Osasuna': 'Osasuna',
    'Atletico Madrid': 'Ath Madrid',
    'Athletic Bilbao': 'Ath Bilbao',
    'Almería': 'Almeria',
    'Rayo Vallecano': 'Vallecano',
    'Granada CF': 'Granada',
    'Cádiz CF': 'Cadiz',
    'Alavés':'Alaves',
    'Real Sociedad':'Sociedad',
    'Real Betis':'Betis',
    'Real Mallorca':'Mallorca',
    'Wolverhampton Wanderers':'Wolves',
    'Tottenham Hotspur':'Tottenham',
    'Brighton and Hove Albion':'Brighton',
    'Nottingham Forest':"Nott'm Forest",
    'Newcastle United':'Newcastle',
    'West Ham United':'West Ham',
    'Manchester United':'Man United',
    'Manchester City':'Man City',
    'Luton Town':'Luton'

})

# Convert 'Date' to datetime and sort the DataFrame
matches['Date'] = pd.to_datetime(matches['Date'], dayfirst=True)
matches.sort_values(by='Date', inplace=True)

# Function to assign points for a single match
def assign_points(row):
    if row['FTR'] == 'H':
        return (row['HomeTeam'], 3), (row['AwayTeam'], 0)
    elif row['FTR'] == 'A':
        return (row['HomeTeam'], 0), (row['AwayTeam'], 3)
    else:
        return (row['HomeTeam'], 1), (row['AwayTeam'], 1)

# Initialize dictionaries for points and goal difference
team_points_5 = defaultdict(lambda: np.zeros(5, dtype=int))  # A rolling window of the last 5 games for points
team_points_20 = defaultdict(lambda: np.zeros(20, dtype=int)) # A rolling window of the last 20 games for points
team_goal_diff = defaultdict(lambda: np.zeros(5, dtype=int))  # A rolling window of the last 5 games for goal differences

# Iterate through each row in the DataFrame
for index, row in matches.iterrows():
    home_team, away_team = row['HomeTeam'], row['AwayTeam']
    home_points, away_points = assign_points(row)
    home_goal_diff = row['FTHG'] - row['FTAG']  # Home team goal difference for this match
    away_goal_diff = row['FTAG'] - row['FTHG']  # Away team goal difference for this match

    # Store the sum of the last 5 games' points and goal differences (excluding the current game)
    matches.at[index, 'HomeTeam_RecentPoints'] = team_points_5[home_team].sum()
    matches.at[index, 'AwayTeam_RecentPoints'] = team_points_5[away_team].sum()
    matches.at[index, 'HomeTeam_RecentGoalDiff'] = team_goal_diff[home_team].sum()
    matches.at[index, 'AwayTeam_RecentGoalDiff'] = team_goal_diff[away_team].sum()

    # New: Store the sum of the last 20 games' points (excluding the current game)
    matches.at[index, 'HomeTeam_RecentPoints20'] = team_points_20[home_team].sum()
    matches.at[index, 'AwayTeam_RecentPoints20'] = team_points_20[away_team].sum()

    # Update the rolling window for the next match
    team_points_5[home_team] = np.roll(team_points_5[home_team], -1)
    team_points_5[away_team] = np.roll(team_points_5[away_team], -1)
    team_points_20[home_team] = np.roll(team_points_20[home_team], -1)
    team_points_20[away_team] = np.roll(team_points_20[away_team], -1)
    team_goal_diff[home_team] = np.roll(team_goal_diff[home_team], -1)
    team_goal_diff[away_team] = np.roll(team_goal_diff[away_team], -1)

    team_points_5[home_team][-1] = home_points[1]
    team_points_5[away_team][-1] = away_points[1]
    team_points_20[home_team][-1] = home_points[1]
    team_points_20[away_team][-1] = away_points[1]
    team_goal_diff[home_team][-1] = home_goal_diff if not math.isnan(home_goal_diff) else 0
    team_goal_diff[away_team][-1] = away_goal_diff if not math.isnan(away_goal_diff) else 0

# The default factory function returns 'NA' for any missing key
last_encounter_result = defaultdict(lambda: 'NA')

# Populate the dictionary with 'D' for all unique home-away pairs from matches
for home in matches['HomeTeam'].unique():
    for away in matches['AwayTeam'].unique():
        last_encounter_result[(home, away)] = 'D'

# Iterate through the DataFrame
for index, row in matches.iterrows():
    teams = (row['HomeTeam'], row['AwayTeam'])
    reverse_teams = (row['AwayTeam'], row['HomeTeam'])  # For checking away matches

    # Initialize columns
    matches.at[index, 'LastEncounterWon'] = 0
    matches.at[index, 'LastEncounterLost'] = 0

    # Check the last encounter result and update accordingly
    if last_encounter_result[teams] == 'H':
        matches.at[index, 'LastEncounterWon'] = 1
    elif last_encounter_result[teams] == 'A':
        matches.at[index, 'LastEncounterLost'] = 1

    # For reverse matches, invert the result
    elif last_encounter_result[reverse_teams] == 'A':
        matches.at[index, 'LastEncounterWon'] = 1
    elif last_encounter_result[reverse_teams] == 'H':
        matches.at[index, 'LastEncounterLost'] = 1

    # Update the last encounter result
    last_encounter_result[teams] = row['FTR']


# To create Europa competition variables:
# Create a new DataFrame to store positions per season
positions_per_season = pd.DataFrame(columns=['Season', 'Team', 'Wins', 'Draws', 'Points', 'Position'])

# Initialize dictionaries for points, wins, and draws
team_points_season = defaultdict(lambda: defaultdict(int))
team_wins_season = defaultdict(lambda: defaultdict(int))
team_draws_season = defaultdict(lambda: defaultdict(int))

# Iterate through each row in the DataFrame
for index, row in matches.iterrows():
    home_team, away_team = row['HomeTeam'], row['AwayTeam']
    home_points, away_points = assign_points(row)

    # Update points, wins, and draws for the current season
    team_points_season[row['Season']][home_team] += home_points[1]
    team_points_season[row['Season']][away_team] += away_points[1]

    if home_points[1] == 3:  # Home win
        team_wins_season[row['Season']][home_team] += 1
    elif away_points[1] == 3:  # Away win
        team_wins_season[row['Season']][away_team] += 1
    else:  # Draw
        team_draws_season[row['Season']][home_team] += 1
        team_draws_season[row['Season']][away_team] += 1

# Calculate positions per season
for season, team_points in team_points_season.items():
    season_data = {'Season': season, 'Team': [], 'Wins': [], 'Draws': [], 'Points': [], 'Position': [], 'Div': []}

    # Sort teams by points and assign positions
    sorted_teams = sorted(team_points.keys(), key=lambda x: (team_points[x], x), reverse=True)
    for position, team in enumerate(sorted_teams, start=1):
        # Retrieve the league division for the team
        team_div = matches[(matches['Season'] == season) & ((matches['HomeTeam'] == team) | (matches['AwayTeam'] == team))]['Div'].values[0]

        season_data['Team'].append(team)
        season_data['Wins'].append(team_wins_season[season][team])
        season_data['Draws'].append(team_draws_season[season][team])
        season_data['Points'].append(team_points[team])
        season_data['Position'].append(position)
        season_data['Div'].append(team_div)

    # Append season data to the positions_per_season DataFrame
    positions_per_season = positions_per_season._append(pd.DataFrame(season_data))

# Add columns indicating whether the home and away teams will play in the Champions League or Europa League the season afterward
matches['HomeChampionsLeague'] = 0
matches['AwayChampionsLeague'] = 0
matches['HomeEuropaLeague'] = 0
matches['AwayEuropaLeague'] = 0

# Manually set Champions League and Europa League status for the first season (2005-2006)
first_season = '2005-2006'
champions_league_teams = {"SP1": ["Barcelona", "Real Madrid", "Betis", "Villarreal"],
                           "E0": ["Chelsea", "Arsenal", "Man United", "Everton"]}
europa_league_teams = {"SP1": ["Espanol", "Sevilla"],
                       "E0": ["Liverpool", "Bolton"]}

for div, teams in champions_league_teams.items():
    matches.loc[(matches['Season'] == first_season) & (matches['Div'] == div), 'HomeChampionsLeague'] = matches['HomeTeam'].isin(teams).astype(int)
    matches.loc[(matches['Season'] == first_season) & (matches['Div'] == div), 'AwayChampionsLeague'] = matches['AwayTeam'].isin(teams).astype(int)

for div, teams in europa_league_teams.items():
    matches.loc[(matches['Season'] == first_season) & (matches['Div'] == div), 'HomeEuropaLeague'] = matches['HomeTeam'].isin(teams).astype(int)
    matches.loc[(matches['Season'] == first_season) & (matches['Div'] == div), 'AwayEuropaLeague'] = matches['AwayTeam'].isin(teams).astype(int)

# Update the 'ChampionsLeague' and 'EuropaLeague' columns based on the positions_per_season DataFrame for subsequent seasons
for index, row in matches.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    season = row['Season']
    team_div = row['Div']

    if season != first_season:
        # Check if the home team finished in the top 4 for Champions League or 5-6 for Europa League
        home_team_position = positions_per_season[(positions_per_season['Season'] == season) & (positions_per_season['Team'] == home_team) & (positions_per_season['Div'] == team_div)]['Position'].values
        if home_team_position.size > 0:
            home_team_position = home_team_position[0]
            matches.at[index, 'HomeChampionsLeague'] = 1 if 1 <= home_team_position <= 4 else 0
            matches.at[index, 'HomeEuropaLeague'] = 1 if 5 <= home_team_position <= 6 else 0

        # Check if the away team finished in the top 4 for Champions League or 5-6 for Europa League
        away_team_position = positions_per_season[(positions_per_season['Season'] == season) & (positions_per_season['Team'] == away_team) & (positions_per_season['Div'] == team_div)]['Position'].values
        if away_team_position.size > 0:
            away_team_position = away_team_position[0]
            matches.at[index, 'AwayChampionsLeague'] = 1 if 1 <= away_team_position <= 4 else 0
            matches.at[index, 'AwayEuropaLeague'] = 1 if 5 <= away_team_position <= 6 else 0

historical_rankings = [
    ("Barcelona", 1),
    ("Real Madrid", 2),
    ("Ath Madrid", 3),
    ("Sevilla", 4),
    ("Valencia", 5),
    ("Villarreal", 6),
    ("Ath Bilbao", 7),
    ("Sociedad", 8),
    ("Getafe", 9),
    ("Espanol", 10),
    ("Betis", 11),
    ("Osasuna", 12),
    ("Celta", 13),
    ("Levante", 14),
    ("Mallorca", 15),
    ("Malaga", 16),
    ("La Coruna", 17),
    ("Granada", 18),
    ("Valladolid", 19),
    ("Vallecano", 20),
    ("Zaragoza", 21),
    ("Santander", 22),
    ("Eibar", 23),
    ("Alaves", 24),
    ("Almeria", 25),
    ("Sp Gijon", 26),
    ("Elche", 27),
    ("Cadiz", 28),
    ("Leganes", 29),
    ("Girona", 30),
    ("Recreativo", 31),
    ("Las Palmas", 32),
    ("Huesca", 33),
    ("Tenerife", 34),
    ("Hercules", 35),
    ("Numancia", 36),
    ("Murcia", 37),
    ("Xerez", 38),
    ("Gimnastic", 39),
    ("Cordoba", 40),
    ("Man United", 1),
    ("Liverpool", 2),
    ("Man City", 3),
    ("Chelsea", 4),
    ("Arsenal", 5),
    ("Tottenham", 6),
    ("Everton", 7),
    ("West Ham", 8),
    ("Newcastle", 9),
    ("Aston Villa", 10),
    ("Fulham", 11),
    ("Southampton", 12),
    ("Leicester", 13),
    ("Crystal Palace", 14),
    ("Wolves", 15),
    ("Stoke", 16),
    ("Sunderland", 17),
    ("West Brom", 18),
    ("Blackburn", 19),
    ("Wigan", 20),
    ("Burnley", 21),
    ("Bolton", 22),
    ("Brighton", 23),
    ("Swansea", 24),
    ("Bournemouth", 25),
    ("Watford", 26),
    ("Portsmouth", 27),
    ("Norwich", 28),
    ("Middlesbrough", 29),
    ("Hull", 30),
    ("Birmingham", 31),
    ("Leeds", 32),
    ("Brentford", 33),
    ("Sheffield United", 34),
    ("Reading", 35),
    ("Cardiff", 36),
    ("QPR", 37),
    ("Charlton", 38),
    ("Nott'm Forest", 39),
    ("Blackpool", 40),
    ("Huddersfield", 41),
    ("Luton", 42),
    ("Derby", 43)
]

# Convert historical rankings list to a dictionary
historical_rankings_dict = dict(historical_rankings)

# Add historical rankings to the 'matches' DataFrame
matches['HomeRanking'] = matches['HomeTeam'].map(historical_rankings_dict)
matches['AwayRanking'] = matches['AwayTeam'].map(historical_rankings_dict)

# Add red card column, to see how many red cards where shown in the last game to the team

# Sort the DataFrame by Date
matches['Date'] = pd.to_datetime(matches['Date'])
matches = matches.sort_values(by='Date')

# Initialize a dictionary to store the last game reds for each team
last_game_reds = {}

# Initialize the new columns
matches['reds_last_game_home'] = 0
matches['reds_last_game_away'] = 0

# Iterate through the DataFrame
for index, row in matches.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

    # Get the last game reds for the current teams
    matches.at[index, 'reds_last_game_home'] = last_game_reds.get(home_team, 0)
    matches.at[index, 'reds_last_game_away'] = last_game_reds.get(away_team, 0)

    # Update the dictionary with the current game reds
    last_game_reds[home_team] = row['HR']
    last_game_reds[away_team] = row['AR']

# Add column for bookings, shots on target, woodwork, attendance, offside, free kicks, and corners for both home and away teams over their last 5 games (excluding the current game)"""

# Generalized function to create a rolling sum column
def create_rolling_sum(df, team_col, feature_col):
    # Create a temporary DataFrame to avoid modifying the original one
    temp_df = df.copy()
    # Sort by team and date
    temp_df = temp_df.sort_values(by=[team_col, 'Date'])
    # Create a new column with rolling sum of the feature
    rolling_col_name = f'Rolling_{feature_col}_{team_col}'
    temp_df[rolling_col_name] = temp_df.groupby(team_col)[feature_col].rolling(window=5, min_periods=1).sum().shift().reset_index(level=0, drop=True)
    return temp_df[rolling_col_name]

# Features to calculate rolling sums for
features = ['HY', 'HS', 'HST', 'HC',  # Home team features
            'AY', 'AS', 'AST', 'AC']  # Away team features

# Apply the function for each feature and merge back to the matches DataFrame
for feature in features:
    team_type = 'HomeTeam' if feature.startswith('H') else 'AwayTeam'
    rolling_col = create_rolling_sum(matches, team_type, feature)
    matches[f'Rolling_{feature}'] = rolling_col

# Replace NaN values with 0 (assuming no activity in the first few games)
matches.fillna(0, inplace=True)

#Assign numeric values to our target variable:

# Recode the target variable into three categories (2: Home Win, 1: Draw, 0: Away Win)
matches['MatchOutcome'] = matches['FTR'].map({'H': 2, 'D': 1, 'A': 0})

# Combine variables into 'difference' variables

# Combine Home and Away into one variable accounting for difference
matches['Diff_RecentGoalDiff'] = matches['HomeTeam_RecentGoalDiff'] - matches['AwayTeam_RecentGoalDiff']
matches['Diff_RecentPoints'] = matches['HomeTeam_RecentPoints'] - matches['AwayTeam_RecentPoints']
matches['Rank_diff'] = matches['HomeRanking'] - matches['AwayRanking']

# Create a win by home team column

matches['HomeGoalDiff'] = matches['FTHG'] - matches['FTAG']

# Since the first season contains many unknown values ('previous encounter', 'recent goal diff and 'recent points' cant be calculated for the first matches) we are going to drop all of the data for the first year (2005)."""

matches = matches[matches['Date'].dt.year != 2005]

# Create variables for points this season"""

# Sort the DataFrame by date
matches = matches.sort_values(by='Date')
# Create a dictionary to store current season points for each team
team_points_current_season = defaultdict(lambda: 0)

# Store the current season
current_season = None

# Iterate through each row in the DataFrame
for index, row in matches.iterrows():
    home_team, away_team = row['HomeTeam'], row['AwayTeam']
    home_points, away_points = assign_points(row)

    # Check if it's the start of a new season
    if current_season != row['Season']:
        current_season = row['Season']

        # Reset points for all teams to 0 at the start of a new season
        team_points_current_season = defaultdict(lambda: 0)

    # Update points for the current season
    team_points_current_season[home_team] += home_points[1]
    team_points_current_season[away_team] += away_points[1]

    # Store the current season points
    matches.at[index, 'HomeTeam_PointsThisSeason'] = team_points_current_season[home_team]
    matches.at[index, 'AwayTeam_PointsThisSeason'] = team_points_current_season[away_team]


# Convert div to binary"""

matches['SP1'] = (matches['Div'] == 'SP1').astype(int)
matches['E0'] = (matches['Div'] == 'E0').astype(int)

# Store model variables in two DFs: X for independent variables and y for dependent"""

# Store rows with NA values in the 'MatchOutcome' column in another DataFrame
last_rows_df = matches[matches['MatchOutcome'].isna()].copy()

# Drop rows with NA values in the 'MatchOutcome' column from the 'matches' DataFrame
matches = matches[matches['MatchOutcome'].notna()]


# Select the desired columns for X
X = matches[['LastEncounterLost', 'LastEncounterWon',
             'HomeChampionsLeague', 'AwayChampionsLeague', 'HomeEuropaLeague',
             'AwayEuropaLeague', 'reds_last_game_home', 'reds_last_game_away',
             'Rolling_HY', 'Rolling_HS', 'Rolling_HST', 'Rolling_HC',
             'Rolling_AY', 'Rolling_AS', 'Rolling_AST', 'Rolling_AC', 'HomeTeam_RecentGoalDiff',
             'AwayTeam_RecentGoalDiff', 'HomeTeam_RecentPoints', 'AwayTeam_RecentPoints',
             'HomeTeam_PointsThisSeason','AwayTeam_PointsThisSeason',  'HomeRanking', 'AwayRanking', 'SP1', 'E0']]

# Store variables to predict with just dependent variables columns
X_to_predict = last_rows_df[['LastEncounterLost', 'LastEncounterWon',
             'HomeChampionsLeague', 'AwayChampionsLeague', 'HomeEuropaLeague',
             'AwayEuropaLeague', 'reds_last_game_home', 'reds_last_game_away',
             'Rolling_HY', 'Rolling_HS', 'Rolling_HST', 'Rolling_HC',
             'Rolling_AY', 'Rolling_AS', 'Rolling_AST', 'Rolling_AC', 'HomeTeam_RecentGoalDiff',
             'AwayTeam_RecentGoalDiff', 'HomeTeam_RecentPoints', 'AwayTeam_RecentPoints',
             'HomeTeam_PointsThisSeason','AwayTeam_PointsThisSeason',  'HomeRanking', 'AwayRanking', 'SP1', 'E0']]


# Select and create the dependent variable
y = matches['MatchOutcome']


# Separate data into Training, Validation and Test

# Split the data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Make sure y_train is int type
y_train = y_train.astype(int)

# Scale the data"""

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Scale the df_to_predict using the scaler
X_to_predict_scaled = scaler.transform(X_to_predict)


# Set random seeds to make the neural network replicable
seed_value = 42
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Categrocial NN

# Load the saved model
NN_model_categorical = tf.keras.models.load_model('NN_model.h5')

# NN Categorical predictions

# Predict probabilities with the neural network model
predictions_prob_nn = NN_model_categorical.predict(X_to_predict_scaled)

# Extract the 'HomeTeam' and 'AwayTeam' columns
home_teams = last_rows_df['HomeTeam']
away_teams = last_rows_df['AwayTeam']

# Create a new DataFrame for the neural network predictions
# Order of output probabilities is [away win, draw, home win]
df_nn_predictions = pd.DataFrame({
    'Home': home_teams,
    'Away': away_teams,
    'Home Win Probability': predictions_prob_nn[:, 2],
    'Draw Probability': predictions_prob_nn[:, 1],
    'Away Win Probability': predictions_prob_nn[:, 0]
})

df_nn_predictions = df_nn_predictions.sort_index()


def plot_donut_chart(home_percentage, draw_percentage, away_percentage):

    # Desired order of outcomes
    outcome_labels = ["Away Win", "Draw", "Home Win"]

    # Assign colors to each outcome in the desired order: pale red, pale blue, and gray
    colors = ['#FFC0C0', '#808080', '#C0C0FF']  # Corresponding to "Away Win", "Draw", "Home Win"

    # Reorder percentages to match the desired outcome order
    percentages = [away_percentage, draw_percentage, home_percentage]

    # Create the donut chart
    fig = go.Figure(data=[go.Pie(labels=outcome_labels,
                                 values=percentages,
                                 hole=.4,
                                 marker_colors=colors,
                                 textinfo='text+percent',
                                 textposition='outside',
                                 insidetextorientation='radial',
                                 direction='clockwise',
                                 sort=False
                                 )])

    # Customize hover info and the look of the chart
    fig.update_traces(hoverinfo='label+percent', textfont_size=12, marker=dict(line=dict(color='#000000', width=0.4)))
    # Adjust the size of the entire figure
    fig.update_layout(width=325, height=350)

    # Make legend smaller
    fig.update_layout(legend=dict(font=dict(size=10)))

    return fig

# Set the title for the Streamlit app
st.title('La Liga Predictions')

# Iterate through each row in the DataFrame
for index, row in df_nn_predictions[:10].iterrows():
    # Get the data for the current match
    home_team = row['Home']
    away_team = row['Away']
    home_win_probability = row['Home Win Probability']
    draw_probability = row['Draw Probability']
    away_win_probability = row['Away Win Probability']

    # Plot the donut chart
    fig = plot_donut_chart(home_win_probability, draw_probability, away_win_probability)

    # Display the team names alongside the chart
    st.write(f"**{home_team}** vs **{away_team}**")
    
    # Display the chart in the Streamlit app
    st.plotly_chart(fig, use_container_width=True, width=200, height=100)
  
# Set the title for the Streamlit app
st.title('EPL Predictions')

for index, row in df_nn_predictions[10:].iterrows():
    # Get the data for the current match
    home_team = row['Home']
    away_team = row['Away']
    home_win_probability = row['Home Win Probability']
    draw_probability = row['Draw Probability']
    away_win_probability = row['Away Win Probability']

    # Plot the donut chart
    fig = plot_donut_chart(home_win_probability, draw_probability, away_win_probability)

    # Display the team names alongside the chart
    st.write(f"**{home_team}** vs **{away_team}**")

    # Display the chart in the Streamlit app
    st.plotly_chart(fig, use_container_width=True, width=200, height=100)



# Set the title for the Streamlit app
st.title('Accuracy distribution of the model for every 20 matches')

# Define the values for the histogram
hist_values = np.array([0., 0., 0., 0., 0., 2., 0., 3., 1., 8., 14., 17., 29., 21., 18., 14., 8., 2., 1., 0., 0.])

# Define the data frame for Altair
data = {'Num of Correct Predictions': range(len(hist_values)), 'Probability': hist_values}
df = pd.DataFrame(data)

# Plot the histogram
bars = alt.Chart(df).mark_bar().encode(
    x=alt.X('Num of Correct Predictions:O', title='Number of Correct Predictions', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('Probability:Q', title='Probability')  # Add percent formatting to y-axis
).properties(
    width=alt.Step(20)  # Adjust the width of each bar
)

st.altair_chart(bars, use_container_width=True)
st.write(f"Average number of correct predictions for every 20 matches: 12.32 matches")

"""👁⚫️✨"""
