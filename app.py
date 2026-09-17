import streamlit as st
import pandas as pd
import re
import requests

# ==========================================
# 1. HELPER FUNCTIONS & UI WIDGETS
# ==========================================

def clean_market_data(df):
    def fix_market_string(text):
        text = re.sub(r'(Over|Under)\s*(\d)(\d)', r'\1 \2.\3', str(text), flags=re.IGNORECASE)
        text = re.sub(r'(Corners|Goals|Cards)([A-Za-z])', r'\1 \2', text)
        text = text.replace("Over05", "Over 0.5").replace("Over85", "Over 8.5")
        return text
    if 'Market' in df.columns:
        df['Market'] = df['Market'].apply(fix_market_string)
    if 'Model Probability (%)' in df.columns:
        df['Model Probability (%)'] = df['Model Probability (%)'].apply(lambda x: f"{float(x):.2f}")
    return df

def render_engine_status(is_connected=True):
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        if is_connected:
            st.markdown(
                "<div style='background-color: #1e3a29; color: #4ade80; padding: 6px 12px; border-radius: 6px; border: 1px solid #22c55e; font-size: 13px; text-align: center; font-weight: 600;'>🟢 Data Engine Live</div>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background-color: #3b181b; color: #f87171; padding: 6px 12px; border-radius: 6px; border: 1px solid #ef4444; font-size: 13px; text-align: center; font-weight: 600; box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);'>🔴 Activate Engine</div>", 
                unsafe_allow_html=True
            )

def refresh_app():
    pass

# ==========================================
# 2. LIVE API DATA PIPELINE
# ==========================================

@st.cache_data(ttl=3600) # Caches data for 1 hour to keep app fast
def load_live_fpl_data():
    """Fetches real-time active player data from the Official FPL API."""
    try:
        fpl_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        response = requests.get(fpl_url)
        fpl_data = response.json()
        
        players_df = pd.DataFrame(fpl_data['elements'])
        teams_df = pd.DataFrame(fpl_data['teams'])
        return players_df, teams_df
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

players_df, teams_df = load_live_fpl_data()

def get_team_roster(team_name, players_dataframe, teams_dataframe):
    """Filters the live API data for the specific team's top active players."""
    if players_dataframe.empty or teams_dataframe.empty:
        return pd.DataFrame({"Notice": ["API Data Unavailable"]})
        
    # Handle common name mismatches (e.g., Man City vs Manchester City)
    search_name = "Man City" if team_name == "Manchester City" else team_name
    search_name = "Spurs" if team_name == "Tottenham" else search_name

    team_match = teams_dataframe[teams_dataframe['name'].str.contains(search_name, case=False, na=False)]
    
    if team_match.empty:
        return pd.DataFrame({"Notice": [f"No live data found for {team_name}"]})
    
    team_id = team_match.iloc[0]['id']
    
    # Filter: Only this team, only active status ('a'), sort by form
    roster = players_dataframe[
        (players_dataframe['team'] == team_id) & 
        (players_dataframe['status'] == 'a')
    ].sort_values(by='form', ascending=False).head(8) # Top 8 players
    
    # Format for UI
    roster = roster[['web_name', 'form', 'points_per_game', 'goals_scored', 'assists']]
    roster.columns = ['Player', 'Recent Form', 'PPG', 'Goals', 'Assists']
    return roster

# ==========================================
# 3. SIDEBAR & STATE MANAGEMENT
# ==========================================

st.sidebar.header("⚽ 401 Predictor Tool")

league_select = st.sidebar.selectbox(
    "Competition", 
    ["English Premier League", "Spanish La Liga", "Italian Serie A"],
    key="league_state",
    on_change=refresh_app
)

gw_select = st.sidebar.selectbox(
    "Gameweek / Round", 
    [f"Gameweek {i}" for i in range(1, 39)], 
    index=3,
    key="gw_state",
    on_change=refresh_app
)

# Placeholder fixtures - you will later replace this list by querying your match database
available_matches = [
    "Brentford vs Chelsea",
    "Man City vs Arsenal",
    "Liverpool vs Man Utd",
    "Aston Villa vs Spurs"
]
selected_fixture_name = st.sidebar.selectbox(
    "Select Match Fixture", 
    available_matches,
    key="fixture_state"
)

st.sidebar.markdown("---")
st.sidebar.button("Share App", use_container_width=True)

# ==========================================
# 4. MAIN USER INTERFACE
# ==========================================

# Dynamically extract teams from the string
if " vs " in selected_fixture_name:
    home_team, away_team = selected_fixture_name.split(" vs ")
else:
    home_team, away_team = "Home", "Away"

render_engine_status(is_connected=True)

st.markdown(
    f"<div style='background-color: #1e1e1e; padding: 15px; border-radius: 8px; border-left: 5px solid #2563eb; margin-bottom: 20px; margin-top: 15px;'><h2 style='margin: 0; padding: 0; font-size: 24px;'>{selected_fixture_name}</h2></div>", 
    unsafe_allow_html=True
)

# --- A. Match Analyst Preview ---
st.subheader("🎙️ Match Analyst Preview")
analyst_text = f"**{home_team}** enter this matchup in strong dynamic form, driven by excellent defensive solidity and a high-volume attacking threat. **{away_team}** present an intriguing challenge, operating with high corner volume potential but showing higher disciplinary vulnerability."
st.markdown(f"<div style='color: #e2e8f0; font-size: 16px; line-height: 1.6; margin-bottom: 15px;'>{analyst_text}</div>", unsafe_allow_html=True)
st.markdown("🎯 **Best Suited Markets:** Double Chance (1X), Match Over 1.5 Goals, Corners Over 8.5")
st.markdown("---")

# --- B. Team Tactical Profiles ---
st.subheader("🛡️ Team Tactical Profiles")
tactical_data = {
    "Metric": ["Dynamic Form", "Defensive Solidity", "Duel Success", "Goalkeeper Rating", "Goal Threat", "Eye Test"],
    home_team: [3.00, 4.2, 4.8, 3.67, 4.73, 4.6], 
    away_team: [3.00, 2.6, 1.8, 4.73, 2.47, 2.8]  
}
st.dataframe(pd.DataFrame(tactical_data), hide_index=True, use_container_width=True)
st.markdown("---")

# --- C. Probable Outcomes ---
st.subheader("🔥 Top 5 Most Probable Outcomes")
raw_top_5 = pd.DataFrame({
    "Market": ["DoubleChance 1X", "Over 15 Goals", "Corners Home", "HT Over05 Goals", "Corners Over85"],
    "Model Probability (%)": [85.9000, 84.1000, 81.4000, 77.3000, 74.0000],
    "Confidence Tier": ["🟢 Realistic (> 70%)", "🟢 Realistic (> 70%)", "🟢 Realistic (> 70%)", "🟢 Realistic (> 70%)", "🟢 Realistic (> 70%)"]
})
st.table(clean_market_data(raw_top_5))
st.markdown("---")

# --- D. Key Player Props ---
st.subheader("⚽ Key Player Props")

# Fetch live data
home_players_live = get_team_roster(home_team, players_df, teams_df)
away_players_live = get_team_roster(away_team, players_df, teams_df)

def render_player_cards(df):
    """Helper function to render a dataframe of players as styled HTML cards."""
    if df.empty or "Notice" in df.columns:
        st.warning("Player data currently unavailable.")
        return

    # Create a 2-column grid for the cards
    cols = st.columns(2)
    
    for index, row in df.iterrows():
        col = cols[index % 2]
        with col:
            st.markdown(
                f"""
                <div style='background-color: #161b22; padding: 16px; border-radius: 12px; 
                            border: 1px solid #30363d; border-left: 4px solid #3b82f6; 
                            margin-bottom: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); transition: transform 0.2s;'>
                    <h4 style='margin: 0 0 12px 0; color: #f0f6fc; font-size: 17px; font-weight: 600;'>
                        👤 {row['Player']}
                    </h4>
                    <div style='display: flex; justify-content: space-between; align-items: center; 
                                font-size: 14px; color: #8b949e; background-color: #0d1117; 
                                padding: 10px; border-radius: 8px;'>
                        <div style='text-align: center;'>
                            <div style='font-weight: 700; color: #fbbf24; font-size: 16px;'>{row['Recent Form']}</div>
                            <div style='font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;'>Form</div>
                        </div>
                        <div style='text-align: center;'>
                            <div style='font-weight: 700; color: #4ade80; font-size: 16px;'>{row['PPG']}</div>
                            <div style='font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;'>PPG</div>
                        </div>
                        <div style='text-align: center;'>
                            <div style='font-weight: 700; color: #60a5fa; font-size: 16px;'>{row['Goals']}</div>
                            <div style='font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;'>Goals</div>
                        </div>
                        <div style='text-align: center;'>
                            <div style='font-weight: 700; color: #c084fc; font-size: 16px;'>{row['Assists']}</div>
                            <div style='font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;'>Assists</div>
                        </div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

# Create interactive tabs for clean navigation
tab1, tab2 = st.tabs([f"🏠 {home_team} Roster", f"✈️ {away_team} Roster"])

with tab1:
    render_player_cards(home_players_live)

with tab2:
    render_player_cards(away_players_live)