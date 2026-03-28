# By Edmundo Cuadra — updated March 2026

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import math
import os
import requests
from collections import defaultdict
from datetime import datetime
import streamlit as st
import plotly.graph_objs as go
import altair as alt

warnings.filterwarnings("ignore")

# ─── Page configuration ─────────────────────────────────────
st.set_page_config(
    page_title="Football Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
#MainMenu, footer, header {visibility: hidden;}
.stApp {font-family: 'Inter', sans-serif;}

.hero {text-align:center; padding:1.8rem 1rem 0.5rem;}
.hero h1 {
    font-size:2.6rem; font-weight:800; margin-bottom:0.15rem;
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p {font-size:1rem; color:#999; margin-top:0;}

.match-card {
    background:linear-gradient(145deg,#1a1a2e 0%,#16213e 100%);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:16px; padding:1.2rem 1rem 0.6rem;
    margin-bottom:0.75rem;
    transition:transform 0.2s,box-shadow 0.2s;
}
.match-card:hover {
    transform:translateY(-2px);
    box-shadow:0 8px 30px rgba(0,0,0,0.35);
}
.match-teams {
    display:flex; justify-content:center; align-items:center;
    gap:0.6rem; margin-bottom:0.25rem;
}
.team-name {font-size:0.95rem;font-weight:700;color:#e0e0e0;}
.vs-badge {
    font-size:0.65rem; font-weight:600; color:#555;
    background:rgba(255,255,255,0.05); border-radius:6px;
    padding:2px 8px;
}
.pred-label {
    text-align:center; font-size:0.78rem; font-weight:600;
    margin-top:2px; margin-bottom:4px;
}
.pred-home {color:#3b82f6;} .pred-draw {color:#9ca3af;} .pred-away {color:#ef4444;}

.stTabs [data-baseweb="tab-list"] {gap:8px;justify-content:center;}
.stTabs [data-baseweb="tab"] {border-radius:10px;padding:8px 28px;font-weight:600;}

.section-divider {
    height:1px; margin:1.5rem 0;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);
}
.acc-header {text-align:center; margin:0.5rem 0;}
.acc-header h3 {font-weight:700; color:#e0e0e0;}
.acc-header p {color:#888; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────
CURRENT_SEASON = '2025-26'

ESPN_TO_FD = {
    'Rayo Vallecano': 'Vallecano', 'Atletico Madrid': 'Ath Madrid',
    'Atlético Madrid': 'Ath Madrid', 'Atlético de Madrid': 'Ath Madrid',
    'Athletic Club': 'Ath Bilbao', 'Athletic Bilbao': 'Ath Bilbao',
    'Real Sociedad': 'Sociedad', 'Real Betis': 'Betis',
    'Real Mallorca': 'Mallorca', 'RCD Mallorca': 'Mallorca',
    'Real Valladolid': 'Valladolid', 'Celta Vigo': 'Celta',
    'Celta de Vigo': 'Celta', 'CA Osasuna': 'Osasuna',
    'Alavés': 'Alaves', 'Deportivo Alavés': 'Alaves',
    'Leganés': 'Leganes', 'CD Leganés': 'Leganes',
    'RCD Espanyol': 'Espanol', 'Espanyol': 'Espanol',
    'Cádiz': 'Cadiz', 'Cádiz CF': 'Cadiz',
    'Almería': 'Almeria', 'UD Almería': 'Almeria',
    'Granada CF': 'Granada', 'Villarreal CF': 'Villarreal',
    'Real Oviedo': 'Oviedo',
    'Manchester City': 'Man City', 'Manchester United': 'Man United',
    'Tottenham Hotspur': 'Tottenham',
    'Wolverhampton Wanderers': 'Wolves',
    'Brighton & Hove Albion': 'Brighton',
    'Brighton and Hove Albion': 'Brighton',
    'Nottingham Forest': "Nott'm Forest",
    'Newcastle United': 'Newcastle', 'West Ham United': 'West Ham',
    'Leicester City': 'Leicester', 'Luton Town': 'Luton',
    'Sheffield Utd': 'Sheffield United',
    'AFC Bournemouth': 'Bournemouth',
}

FD_NAME_MAP = {
    'Villareal': 'Villarreal', 'Celta Vigo': 'Celta',
    'CA Osasuna': 'Osasuna', 'Atlético': 'Ath Madrid',
    'Athletic Club': 'Ath Bilbao', 'Almería': 'Almeria',
    'Rayo Vallecano': 'Vallecano', 'Granada CF': 'Granada',
    'Cádiz CF': 'Cadiz', 'Alavés': 'Alaves',
    'Leganés': 'Leganes', 'Real Sociedad': 'Sociedad',
    'Real Betis': 'Betis', 'Real Mallorca': 'Mallorca',
    'Real Valladolid': 'Valladolid', 'Espanyol': 'Espanol',
    'Wolverhampton Wanderers': 'Wolves',
    'Tottenham Hotspur': 'Tottenham',
    'Brighton and Hove Albion': 'Brighton',
    'Nottm Forest': "Nott'm Forest",
    'Newcastle United': 'Newcastle',
    'West Ham United': 'West Ham',
    'Man Utd': 'Man United', 'Luton Town': 'Luton',
    'Ipswich': 'Ipswich Town',
}

HISTORICAL_RANKINGS = {
    "Barcelona": 1, "Real Madrid": 2, "Ath Madrid": 3, "Sevilla": 4,
    "Valencia": 5, "Villarreal": 6, "Ath Bilbao": 7, "Sociedad": 8,
    "Getafe": 9, "Espanol": 10, "Betis": 11, "Osasuna": 12,
    "Celta": 13, "Levante": 14, "Mallorca": 15, "Malaga": 16,
    "La Coruna": 17, "Granada": 18, "Valladolid": 19, "Vallecano": 20,
    "Zaragoza": 21, "Santander": 22, "Eibar": 23, "Alaves": 24,
    "Almeria": 25, "Sp Gijon": 26, "Elche": 27, "Cadiz": 28,
    "Leganes": 29, "Girona": 30, "Recreativo": 31, "Las Palmas": 32,
    "Huesca": 33, "Tenerife": 34, "Hercules": 35, "Numancia": 36,
    "Murcia": 37, "Xerez": 38, "Gimnastic": 39, "Cordoba": 40,
    "Oviedo": 41,
    "Man United": 1, "Liverpool": 2, "Man City": 3, "Chelsea": 4,
    "Arsenal": 5, "Tottenham": 6, "Everton": 7, "West Ham": 8,
    "Newcastle": 9, "Aston Villa": 10, "Fulham": 11, "Southampton": 12,
    "Leicester": 13, "Crystal Palace": 14, "Wolves": 15, "Stoke": 16,
    "Sunderland": 17, "West Brom": 18, "Blackburn": 19, "Wigan": 20,
    "Burnley": 21, "Bolton": 22, "Brighton": 23, "Swansea": 24,
    "Bournemouth": 25, "Watford": 26, "Portsmouth": 27, "Norwich": 28,
    "Middlesbrough": 29, "Hull": 30, "Birmingham": 31, "Leeds": 32,
    "Brentford": 33, "Sheffield United": 34, "Reading": 35,
    "Cardiff": 36, "QPR": 37, "Charlton": 38, "Nott'm Forest": 39,
    "Blackpool": 40, "Huddersfield": 41, "Luton": 42, "Derby": 43,
    "Ipswich Town": 44,
}

FEATURE_COLS = [
    'LastEncounterLost', 'LastEncounterWon',
    'HomeChampionsLeague', 'AwayChampionsLeague',
    'HomeEuropaLeague', 'AwayEuropaLeague',
    'reds_last_game_home', 'reds_last_game_away',
    'Rolling_HY', 'Rolling_HS', 'Rolling_HST', 'Rolling_HC',
    'Rolling_AY', 'Rolling_AS', 'Rolling_AST', 'Rolling_AC',
    'HomeTeam_RecentGoalDiff', 'AwayTeam_RecentGoalDiff',
    'HomeTeam_RecentPoints', 'AwayTeam_RecentPoints',
    'HomeTeam_PointsThisSeason', 'AwayTeam_PointsThisSeason',
    'HomeRanking', 'AwayRanking', 'SP1', 'E0',
]


# ─── Helper functions ────────────────────────────────────────

def assign_points(row):
    if row['FTR'] == 'H':
        return (row['HomeTeam'], 3), (row['AwayTeam'], 0)
    elif row['FTR'] == 'A':
        return (row['HomeTeam'], 0), (row['AwayTeam'], 3)
    return (row['HomeTeam'], 1), (row['AwayTeam'], 1)


def create_rolling_sum(df, team_col, feature_col):
    temp = df.copy().sort_values(by=[team_col, 'Date'])
    col_name = f'Rolling_{feature_col}_{team_col}'
    temp[col_name] = (
        temp.groupby(team_col)[feature_col]
        .rolling(window=5, min_periods=1).sum()
        .shift().reset_index(level=0, drop=True)
    )
    return temp[col_name]


@st.cache_resource
def _load_weights():
    return np.load('model_weights.npz')


def nn_predict(X):
    """Forward pass through the trained neural network using only numpy.
    Architecture: Dense(128,relu)→BN→Dense(64,relu)→BN→Dense(32,relu)→BN→Dense(3,softmax)
    """
    w = _load_weights()
    eps = 1e-3

    def dense(x, layer):
        return x @ w[f'{layer}_kernel'] + w[f'{layer}_bias']

    def batch_norm(x, layer):
        gamma = w[f'{layer}_gamma']
        beta = w[f'{layer}_beta']
        mean = w[f'{layer}_moving_mean']
        var = w[f'{layer}_moving_variance']
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def relu(x):
        return np.maximum(0, x)

    def softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    x = relu(dense(X, 'dense'))
    x = batch_norm(x, 'batch_normalization')
    x = relu(dense(x, 'dense_1'))
    x = batch_norm(x, 'batch_normalization_1')
    x = relu(dense(x, 'dense_2'))
    x = batch_norm(x, 'batch_normalization_2')
    x = softmax(dense(x, 'dense_3'))
    return x


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_espn_fixtures(league_div, num_matches=10):
    """Pull upcoming scheduled fixtures from the public ESPN API."""
    league_map = {'SP1': 'esp.1', 'E0': 'eng.1'}
    espn_code = league_map.get(league_div)
    if not espn_code:
        return pd.DataFrame()

    now_str = datetime.now().strftime('%Y-%m-%dT')
    today_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    base = f'https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_code}/scoreboard'

    resp = requests.get(base, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    calendar = data.get('leagues', [{}])[0].get('calendar', [])
    upcoming = [d for d in calendar if d >= now_str][:10]

    rows = []
    for date_str in upcoming:
        dp = date_str[:10].replace('-', '')
        day = requests.get(f'{base}?dates={dp}', timeout=15).json()
        for ev in day.get('events', []):
            if ev.get('status', {}).get('type', {}).get('state') != 'pre':
                continue
            comps = ev['competitions'][0].get('competitors', [])
            home = next((c for c in comps if c['homeAway'] == 'home'), None)
            away = next((c for c in comps if c['homeAway'] == 'away'), None)
            if home and away:
                rows.append({
                    'Div': league_div,
                    'Date': today_ts,
                    'Season': CURRENT_SEASON,
                    'HomeTeam': ESPN_TO_FD.get(
                        home['team']['displayName'],
                        home['team']['displayName']),
                    'AwayTeam': ESPN_TO_FD.get(
                        away['team']['displayName'],
                        away['team']['displayName']),
                })
        if len(rows) >= num_matches:
            break
    return pd.DataFrame(rows[:num_matches]) if rows else pd.DataFrame()


# ─── Main data pipeline ─────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Crunching the numbers …")
def load_and_predict():
    """End-to-end: load data, engineer features, predict."""

    # 1 — Historical + current season data ────────────────────
    matches_old = pd.read_csv('light_leagues_data23_24.csv')
    season_dfs = [matches_old]
    for scode in ['2425', '2526']:
        for div in ['SP1', 'E0']:
            try:
                url = f'https://www.football-data.co.uk/mmz4281/{scode}/{div}.csv'
                df = pd.read_csv(url)
                if not df.empty:
                    season_dfs.append(df)
            except Exception:
                pass
    matches = pd.concat(season_dfs, axis=0, ignore_index=True)
    matches = matches.dropna(subset=['Date'])
    matches['Date'] = pd.to_datetime(matches['Date'], format='mixed', dayfirst=True)
    matches['Season'] = np.where(
        matches['Date'].dt.month >= 8,
        matches['Date'].dt.year,
        matches['Date'].dt.year - 1,
    )
    matches['Season'] = (
        matches['Season'].astype(str) + '-'
        + (matches['Season'] + 1).astype(str).str[-2:]
    )
    matches_df = matches.copy()

    # 2 — Upcoming fixtures from ESPN ─────────────────────────
    next_sp = fetch_espn_fixtures('SP1', 10)
    next_en = fetch_espn_fixtures('E0', 10)
    next_matches = pd.concat([next_sp, next_en], ignore_index=True)
    if next_matches.empty:
        return pd.DataFrame()
    matches = pd.concat([matches_df, next_matches], ignore_index=True)
    matches = matches.replace(FD_NAME_MAP)
    matches['Date'] = pd.to_datetime(matches['Date'], format='mixed', dayfirst=True)
    matches.sort_values(by='Date', inplace=True)

    # 3 — Feature: recent form (5 & 20 games) ────────────────
    tp5 = defaultdict(lambda: np.zeros(5, dtype=int))
    tp20 = defaultdict(lambda: np.zeros(20, dtype=int))
    tgd = defaultdict(lambda: np.zeros(5, dtype=int))

    for idx, row in matches.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        hp, ap = assign_points(row)
        hgd = row['FTHG'] - row['FTAG']
        agd = row['FTAG'] - row['FTHG']
        matches.at[idx, 'HomeTeam_RecentPoints'] = tp5[ht].sum()
        matches.at[idx, 'AwayTeam_RecentPoints'] = tp5[at].sum()
        matches.at[idx, 'HomeTeam_RecentGoalDiff'] = tgd[ht].sum()
        matches.at[idx, 'AwayTeam_RecentGoalDiff'] = tgd[at].sum()
        matches.at[idx, 'HomeTeam_RecentPoints20'] = tp20[ht].sum()
        matches.at[idx, 'AwayTeam_RecentPoints20'] = tp20[at].sum()
        for arr, team, val in [
            (tp5, ht, hp[1]), (tp5, at, ap[1]),
            (tp20, ht, hp[1]), (tp20, at, ap[1]),
        ]:
            arr[team] = np.roll(arr[team], -1); arr[team][-1] = val
        for team, val in [(ht, hgd), (at, agd)]:
            tgd[team] = np.roll(tgd[team], -1)
            tgd[team][-1] = val if not math.isnan(val) else 0

    # 4 — Feature: last encounter ─────────────────────────────
    ler = defaultdict(lambda: 'NA')
    for h in matches['HomeTeam'].unique():
        for a in matches['AwayTeam'].unique():
            ler[(h, a)] = 'D'
    for idx, row in matches.iterrows():
        t = (row['HomeTeam'], row['AwayTeam'])
        r = (row['AwayTeam'], row['HomeTeam'])
        matches.at[idx, 'LastEncounterWon'] = 0
        matches.at[idx, 'LastEncounterLost'] = 0
        if ler[t] == 'H':
            matches.at[idx, 'LastEncounterWon'] = 1
        elif ler[t] == 'A':
            matches.at[idx, 'LastEncounterLost'] = 1
        elif ler[r] == 'A':
            matches.at[idx, 'LastEncounterWon'] = 1
        elif ler[r] == 'H':
            matches.at[idx, 'LastEncounterLost'] = 1
        ler[t] = row['FTR']

    # 5 — Feature: European competition flags ─────────────────
    positions_per_season = pd.DataFrame(
        columns=['Season', 'Team', 'Wins', 'Draws', 'Points', 'Position'])
    tps = defaultdict(lambda: defaultdict(int))
    tws = defaultdict(lambda: defaultdict(int))
    tds = defaultdict(lambda: defaultdict(int))
    for _, row in matches.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        hp, ap = assign_points(row)
        tps[row['Season']][ht] += hp[1]
        tps[row['Season']][at] += ap[1]
        if hp[1] == 3:
            tws[row['Season']][ht] += 1
        elif ap[1] == 3:
            tws[row['Season']][at] += 1
        else:
            tds[row['Season']][ht] += 1
            tds[row['Season']][at] += 1
    for season, team_pts in tps.items():
        sd = {'Season': season, 'Team': [], 'Wins': [], 'Draws': [],
              'Points': [], 'Position': [], 'Div': []}
        sorted_teams = sorted(team_pts.keys(),
                              key=lambda x: (team_pts[x], x), reverse=True)
        for pos, team in enumerate(sorted_teams, 1):
            team_div = matches[
                (matches['Season'] == season)
                & ((matches['HomeTeam'] == team) | (matches['AwayTeam'] == team))
            ]['Div'].values[0]
            sd['Team'].append(team)
            sd['Wins'].append(tws[season][team])
            sd['Draws'].append(tds[season][team])
            sd['Points'].append(team_pts[team])
            sd['Position'].append(pos)
            sd['Div'].append(team_div)
        positions_per_season = pd.concat(
            [positions_per_season, pd.DataFrame(sd)], ignore_index=True)

    for col in ['HomeChampionsLeague', 'AwayChampionsLeague',
                'HomeEuropaLeague', 'AwayEuropaLeague']:
        matches[col] = 0

    first_season = '2005-2006'
    cl = {"SP1": ["Barcelona", "Real Madrid", "Betis", "Villarreal"],
          "E0": ["Chelsea", "Arsenal", "Man United", "Everton"]}
    el = {"SP1": ["Espanol", "Sevilla"], "E0": ["Liverpool", "Bolton"]}
    for div, teams in cl.items():
        mask = (matches['Season'] == first_season) & (matches['Div'] == div)
        matches.loc[mask, 'HomeChampionsLeague'] = matches['HomeTeam'].isin(teams).astype(int)
        matches.loc[mask, 'AwayChampionsLeague'] = matches['AwayTeam'].isin(teams).astype(int)
    for div, teams in el.items():
        mask = (matches['Season'] == first_season) & (matches['Div'] == div)
        matches.loc[mask, 'HomeEuropaLeague'] = matches['HomeTeam'].isin(teams).astype(int)
        matches.loc[mask, 'AwayEuropaLeague'] = matches['AwayTeam'].isin(teams).astype(int)

    def prev_season_str(s):
        year = int(s[:4])
        return f"{year - 1}-{str(year)[-2:]}"

    for idx, row in matches.iterrows():
        if row['Season'] == first_season:
            continue
        prev_s = prev_season_str(row['Season'])
        div = row['Div']
        for prefix, team_col in [('Home', 'HomeTeam'), ('Away', 'AwayTeam')]:
            pos = positions_per_season[
                (positions_per_season['Season'] == prev_s)
                & (positions_per_season['Team'] == row[team_col])
                & (positions_per_season['Div'] == div)
            ]['Position'].values
            if pos.size > 0:
                matches.at[idx, f'{prefix}ChampionsLeague'] = int(1 <= pos[0] <= 4)
                matches.at[idx, f'{prefix}EuropaLeague'] = int(5 <= pos[0] <= 6)

    # 6 — Feature: historical rankings ────────────────────────
    matches['HomeRanking'] = matches['HomeTeam'].map(HISTORICAL_RANKINGS)
    matches['AwayRanking'] = matches['AwayTeam'].map(HISTORICAL_RANKINGS)

    # 7 — Feature: red cards last game ────────────────────────
    matches = matches.sort_values(by='Date')
    lgr = {}
    matches['reds_last_game_home'] = 0
    matches['reds_last_game_away'] = 0
    for idx, row in matches.iterrows():
        matches.at[idx, 'reds_last_game_home'] = lgr.get(row['HomeTeam'], 0)
        matches.at[idx, 'reds_last_game_away'] = lgr.get(row['AwayTeam'], 0)
        lgr[row['HomeTeam']] = row.get('HR', 0)
        lgr[row['AwayTeam']] = row.get('AR', 0)

    # 8 — Feature: rolling match stats ────────────────────────
    for feat in ['HY', 'HS', 'HST', 'HC', 'AY', 'AS', 'AST', 'AC']:
        tt = 'HomeTeam' if feat.startswith('H') else 'AwayTeam'
        matches[f'Rolling_{feat}'] = create_rolling_sum(matches, tt, feat)
    matches.fillna(0, inplace=True)

    # 9 — Feature: points this season ─────────────────────────
    matches['MatchOutcome'] = matches['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    matches = matches[matches['Date'].dt.year != 2005]
    matches = matches.sort_values(by='Date')
    tpcs = defaultdict(int)
    cur_s = None
    for idx, row in matches.iterrows():
        if cur_s != row['Season']:
            cur_s = row['Season']
            tpcs = defaultdict(int)
        ht, at = row['HomeTeam'], row['AwayTeam']
        matches.at[idx, 'HomeTeam_PointsThisSeason'] = tpcs[ht]
        matches.at[idx, 'AwayTeam_PointsThisSeason'] = tpcs[at]
        hp, ap = assign_points(row)
        tpcs[ht] += hp[1]
        tpcs[at] += ap[1]

    # 10 — League dummies ─────────────────────────────────────
    matches['SP1'] = (matches['Div'] == 'SP1').astype(int)
    matches['E0'] = (matches['Div'] == 'E0').astype(int)

    # 11 — Split & scale ──────────────────────────────────────
    last_rows = matches[matches['MatchOutcome'].isna()].copy()
    train_data = matches[matches['MatchOutcome'].notna()]

    train_data = train_data.sort_values('Date')
    X = train_data[FEATURE_COLS]
    y = train_data['MatchOutcome']
    X_to_predict = last_rows[FEATURE_COLS]

    n = len(X)
    X_train = X.iloc[:int(n * 0.6)]
    y_train = y.iloc[:int(n * 0.6)].astype(int)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_pred_scaled = scaler.transform(X_to_predict)

    # 12 — Model prediction (pure numpy forward pass) ────────
    probs = nn_predict(X_pred_scaled)

    result = pd.DataFrame({
        'Home': last_rows['HomeTeam'].values,
        'Away': last_rows['AwayTeam'].values,
        'Div': last_rows['Div'].values,
        'Home Win Probability': probs[:, 2],
        'Draw Probability': probs[:, 1],
        'Away Win Probability': probs[:, 0],
    })
    return result.sort_index()


# ─── Chart helpers ───────────────────────────────────────────

COLOR_HOME = '#3b82f6'
COLOR_DRAW = '#6b7280'
COLOR_AWAY = '#ef4444'


def donut_chart(home_pct, draw_pct, away_pct):
    labels = ["Away Win", "Draw", "Home Win"]
    values = [away_pct, draw_pct, home_pct]
    colors = [COLOR_AWAY, COLOR_DRAW, COLOR_HOME]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.48,
        marker_colors=colors,
        textinfo='percent', textposition='outside',
        textfont=dict(size=11, color='#ccc'),
        direction='clockwise', sort=False,
        hoverinfo='label+percent',
    ))
    fig.update_traces(marker=dict(line=dict(color='#0e1117', width=2)))
    fig.update_layout(
        width=280, height=280,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            font=dict(size=10, color='#999'),
            orientation='h', y=-0.05, x=0.5, xanchor='center',
        ),
        showlegend=True,
    )
    return fig


def predicted_outcome(home_p, draw_p, away_p):
    mx = max(home_p, draw_p, away_p)
    if mx == home_p:
        return 'Home Win', 'pred-home'
    elif mx == away_p:
        return 'Away Win', 'pred-away'
    return 'Draw', 'pred-draw'


def render_match_card(row):
    hp = row['Home Win Probability']
    dp = row['Draw Probability']
    ap = row['Away Win Probability']
    label, css = predicted_outcome(hp, dp, ap)

    st.markdown(
        f'<div class="match-card">'
        f'<div class="match-teams">'
        f'<span class="team-name">{row["Home"]}</span>'
        f'<span class="vs-badge">vs</span>'
        f'<span class="team-name">{row["Away"]}</span>'
        f'</div>'
        f'<div class="pred-label {css}">Prediction: {label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(donut_chart(hp, dp, ap), use_container_width=True,
                    config={'displayModeBar': False})


# ─── Page layout ─────────────────────────────────────────────

st.markdown(
    '<div class="hero">'
    '<h1>⚽ Football Match Predictions</h1>'
    '<p>Neural-network predictions for upcoming La Liga &amp; Premier League fixtures</p>'
    '</div>',
    unsafe_allow_html=True,
)

try:
    preds = load_and_predict()
except Exception as exc:
    st.error(f"Something went wrong while loading data: {exc}")
    st.stop()

if preds.empty:
    st.info("No upcoming fixtures found right now. Check back closer to matchday!")
    st.stop()

la_liga = preds[preds['Div'] == 'SP1'].reset_index(drop=True)
epl = preds[preds['Div'] == 'E0'].reset_index(drop=True)

tab_liga, tab_epl = st.tabs(["🇪🇸  La Liga", "🏴󠁧󠁢󠁥󠁮󠁧󠁿  Premier League"])

for tab, df, empty_msg in [
    (tab_liga, la_liga, "No upcoming La Liga fixtures found."),
    (tab_epl, epl, "No upcoming Premier League fixtures found."),
]:
    with tab:
        if df.empty:
            st.info(empty_msg)
            continue
        cols = st.columns(2, gap="medium")
        for i, (_, row) in enumerate(df.iterrows()):
            with cols[i % 2]:
                render_match_card(row)

# ─── Model accuracy section ─────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="acc-header">'
    '<h3>📊 Model Accuracy Distribution</h3>'
    '<p>Simulated accuracy across batches of 20 matches</p>'
    '</div>',
    unsafe_allow_html=True,
)

hist_values = np.array([
    0, 0, 0, 0, 0, 2, 0, 3, 1, 8, 14, 17, 29, 21, 18, 14, 8, 2, 1, 0, 0
])
hist_df = pd.DataFrame({
    'Correct Predictions': range(len(hist_values)),
    'Frequency': hist_values,
})

bars = (
    alt.Chart(hist_df)
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color='#667eea')
    .encode(
        x=alt.X('Correct Predictions:O', title='Correct Predictions out of 20',
                 axis=alt.Axis(labelAngle=0, labelColor='#999', titleColor='#999')),
        y=alt.Y('Frequency:Q', title='Frequency',
                 axis=alt.Axis(labelColor='#999', titleColor='#999')),
        tooltip=['Correct Predictions', 'Frequency'],
    )
    .properties(height=300)
    .configure_view(strokeWidth=0)
    .configure_axis(grid=False)
)

st.altair_chart(bars, use_container_width=True)

st.markdown(
    '<p style="text-align:center;color:#888;font-size:0.9rem;">'
    'Average correct predictions per 20 matches: <b>12.32</b>'
    '</p>',
    unsafe_allow_html=True,
)
