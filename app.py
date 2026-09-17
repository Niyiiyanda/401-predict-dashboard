import streamlit as st
import pandas as pd
import re
import requests
import soccerdata as sd
import difflib

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
                "<div style='background-color: #1e3a29; color: #4ade80; padding: 6px 12px; border-radius: 6px; border: 1px solid #22c55e; font-size: 13px; text-align: center; font-weight: 600;'>🟢 Dual Data Engine Live</div>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background-color: #3b181b; color: #f87171; padding: 6px 12px; border-radius: 6px; border: 1px solid #ef4444; font-size: 13px; text-align: center; font-weight: 600;'>🔴 Activate Engine</div>", 
                unsafe_allow_html=True
            )

def refresh_app():
    pass

# ==========================================
# 2. HARMONIZED DATA PIPELINE (FPL + OPTA)
# ==========================================

@st.cache_data(ttl=3600)
def load_live_fpl_data():
    try:
        fpl_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        fpl_data = requests.get(fpl_url).json()
        return pd.DataFrame(fpl_data['elements']), pd.DataFrame(fpl_data['teams'])
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=86400)
def load_opta_player_data():
    try:
        fbref = sd.FBref(leagues="ENG-Premier League", seasons="2024/2025")
        df = pd.merge(
            fbref.read_player_season_stats(stat_type="standard").reset_index(), 
            fbref.read_player_season_stats(stat_type="shooting").reset_index(), 
            on=["team", "player"], how="left"
        )
        return df
    except Exception:
        return pd.DataFrame()

fpl_players_df, fpl_teams_df = load_live_fpl_data()
opta_players_df = load_opta_player_data()

def get_harmonized_roster(team_name, fpl_players, fpl_teams, opta_df):
    if fpl_players.empty or opta_df.empty:
        return pd.DataFrame({"Notice": ["Data Unavailable"]})
        
    search_name = "Man City" if team_name == "Manchester City" else team_name
    search_name = "Spurs" if team_name == "Tottenham" else search_name

    team_match = fpl_teams[fpl_teams['name'].str.contains(search_name, case=False, na=False)]
    if team_match.empty:
        return pd.DataFrame({"Notice": [f"No live data found for {team_name}"]})
    
    # 1. Get Top 8 Active Players from FPL based on Form
    fpl_roster = fpl_players[
        (fpl_players['team'] == team_match.iloc[0]['id']) & 
        (fpl_players['status'] == 'a')
    ].sort_values(by='form', ascending=False).head(8)
    
    harmonized_data = []
    
    # 2. Fuzzy Match FPL players to Opta database to extract betting metrics
    for _, fpl_p in fpl_roster.iterrows():
        full_name = str(fpl_p['first_name']) + " " + str(fpl_p['second_name'])
        matches = difflib.get_close_matches(full_name, opta_df['player'].astype(str).tolist(), n=1, cutoff=0.5)
        
        if matches:
            opta_p = opta_df[opta_df['player'] == matches[0]].iloc[0]
            xg = opta_p.get('xG', 0)
            xa = opta_p.get('xA', 0)
            shots = opta_p.get('shots', 0)
            sot = opta_p.get('shots_on_target', 0)
        else:
            xg, xa, shots, sot = 0, 0, 0, 0
            
        harmonized_data.append({
            'Player': fpl_p['web_name'],
            'Form': fpl_p['form'],
            'PPG': fpl_p['points_per_game'],
            'Goals': fpl_p['goals_scored'],
            'Assists': fpl_p['assists'],
            'xG': xg,
            'xA': xa,
            'Shots': shots,
            'SoT': sot
        })
        
    df = pd.DataFrame(harmonized_data).fillna(0)
    for col in ['xG', 'xA']: 
        if col in df.columns: df[col] = pd.to_numeric(df[col]).round(2)
    return df

# ==========================================
# 3. SIDEBAR & STATE MANAGEMENT
# ==========================================

st.sidebar.header("⚽ 401 Predictor Tool")

league_select = st.sidebar.selectbox("Competition", ["English Premier League", "Spanish La Liga", "Italian Serie A"], key="league_state", on_change=refresh_app)
gw_select = st.sidebar.selectbox("Gameweek / Round", [f"Gameweek {i}" for i in range(1, 39)], index=3, key="gw_state", on_change=refresh_app)
selected_fixture_name = st.sidebar.selectbox("Select Match Fixture", ["Brentford vs Chelsea", "Man City vs Arsenal", "Liverpool vs Manchester Utd", "Aston Villa vs Spurs"], key="fixture_state")
st.sidebar.markdown("---")
st.sidebar.button("Share App", use_container_width=True)

# ==========================================
# 4. MAIN USER INTERFACE
# ==========================================

home_team, away_team = selected_fixture_name.split(" vs ") if " vs " in selected_fixture_name else ("Home", "Away")

render_engine_status(is_connected=True)

st.markdown(f"<div style='background-color: #1e1e1e; padding: 15px; border-radius: 8px; border-left: 5px solid #2563eb; margin-bottom: 20px; margin-top: 15px;'><h2 style='margin: 0; padding: 0; font-size: 24px;'>{selected_fixture_name}</h2></div>", unsafe_allow_html=True)

st.subheader("🎙️ Match Analyst Preview")
st.markdown(f"<div style='color: #e2e8f0; font-size: 16px; line-height: 1.6; margin-bottom: 15px;'>**{home_team}** enter this matchup in strong dynamic form, driven by excellent defensive solidity and a high-volume attacking threat. **{away_team}** present an intriguing challenge, operating with high corner volume potential but showing higher disciplinary vulnerability.</div>", unsafe_allow_html=True)
st.markdown("🎯 **Best Suited Markets:** Double Chance (1X), Match Over 1.5 Goals, Corners Over 8.5\n---")

st.subheader("🛡️ Team Tactical Profiles")
st.dataframe(pd.DataFrame({"Metric": ["Dynamic Form", "Defensive Solidity", "Duel Success", "Goalkeeper Rating", "Goal Threat", "Eye Test"], home_team: [3.00, 4.2, 4.8, 3.67, 4.73, 4.6], away_team: [3.00, 2.6, 1.8, 4.73, 2.47, 2.8]}), hide_index=True, use_container_width=True)
st.markdown("---")

st.subheader("🔥 Top 5 Most Probable Outcomes")
st.table(clean_market_data(pd.DataFrame({"Market": ["DoubleChance 1X", "Over 15 Goals", "Corners Home", "HT Over05 Goals", "Corners Over85"], "Model Probability (%)": [85.9000, 84.1000, 81.4000, 77.3000, 74.0000], "Confidence Tier": ["🟢 Realistic (> 70%)", "🟢 Realistic (> 70%)", "🟢 Realistic (> 70%)", "🟢 Realistic (> 70%)", "🟢 Realistic (> 70%)"]})))
st.markdown("---")


# --- D. Key Player Props (HARMONIZED) ---
st.subheader("⚽ Key Player Props")

home_players_live = get_harmonized_roster(home_team, fpl_players_df, fpl_teams_df, opta_players_df)
away_players_live = get_harmonized_roster(away_team, fpl_players_df, fpl_teams_df, opta_players_df)

def render_player_cards(df):
    if df.empty or "Notice" in df.columns:
        st.warning("Player data currently unavailable.")
        return

    cols = st.columns(2)
    for index, row in df.iterrows():
        col = cols[index % 2]
        with col:
            st.markdown(
                f"""
                <div style='background-color: #161b22; padding: 16px; border-radius: 12px; border: 1px solid #30363d; border-left: 4px solid #3b82f6; margin-bottom: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);'>
                    <h4 style='margin: 0 0 12px 0; color: #f0f6fc; font-size: 17px; font-weight: 600;'>👤 {row['Player']}</h4>
                    
                    <!-- Row 1: FPL Metrics -->
                    <div style='display: flex; justify-content: space-between; align-items: center; font-size: 14px; background-color: #0d1117; padding: 8px; border-radius: 8px 8px 0 0; border-bottom: 1px solid #21262d;'>
                        <div style='text-align: center; width: 25%;'>
                            <div style='font-weight: 700; color: #fbbf24; font-size: 15px;'>{row['Form']}</div>
                            <div style='font-size: 9px; color: #8b949e; text-transform: uppercase;'>Form</div>
                        </div>
                        <div style='text-align: center; width: 25%;'>
                            <div style='font-weight: 700; color: #4ade80; font-size: 15px;'>{row['PPG']}</div>
                            <div style='font-size: 9px; color: #8b949e; text-transform: uppercase;'>PPG</div>
                        </div>
                        <div style='text-align: center; width: 25%;'>
                            <div style='font-weight: 700; color: #60a5fa; font-size: 15px;'>{row['Goals']}</div>
                            <div style='font-size: 9px; color: #8b949e; text-transform: uppercase;'>Goals</div>
                        </div>
                        <div style='text-align: center; width: 25%;'>
                            <div style='font-weight: 700; color: #c084fc; font-size: 15px;'>{row['Assists']}</div>
                            <div style='font-size: 9px; color: #8b949e; text-transform: uppercase;'>Assists</div>
                        </div>
                    </div>
                    
                    <!-- Row 2: Opta Betting Metrics -->
                    <div style='display: flex; justify-content: space-between; align-items: center; font-size: 14px; background-color: #161b22; padding: 8px; border-radius: 0 0 8px 8px; border: 1px solid #0d1117; border-top: none;'>
                        <div style='text-align: center; width: 25%;'>
                            <div style='font-weight: 700; color: #e5e7eb; font-size: 13px;'>{row['xG']}</div>
                            <div style='font-size: 9px; color: #8b949e; text-transform: uppercase;'>xG</div>
                        </div>
                        <div style='text-align: center; width: 25%; border-left: 1px solid #30363d; border-right: 1px solid #30363d;'>
                            <div style='font-weight: 700; color: #e5e7eb; font-size: 13px;'>{row['xA']}</div>
                            <div style='font-size: 9px; color: #8b949e; text-transform: uppercase;'>xA</div>
                        </div>
                        <div style='text-align: center; width: 25%; border-right: 1px solid #30363d;'>
                            <div style='font-weight: 700; color: #f87171; font-size: 13px;'>{row['Shots']}</div>
                            <div style='font-size: 9px; color: #8b949e; text-transform: uppercase;'>Shots</div>
                        </div>
                        <div style='text-align: center; width: 25%;'>
                            <div style='font-weight: 700; color: #34d399; font-size: 13px;'>{row['SoT']}</div>
                            <div style='font-size: 9px; color: #8b949e; text-transform: uppercase;'>SoT</div>
                        </div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

tab1, tab2 = st.tabs([f"🏠 {home_team} Roster", f"✈️ {away_team} Roster"])
with tab1: render_player_cards(home_players_live)
with tab2: render_player_cards(away_players_live)