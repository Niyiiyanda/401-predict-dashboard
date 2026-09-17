import streamlit as st
import pandas as pd
import re

# ==========================================
# 1. HELPER FUNCTIONS & UI WIDGETS
# ==========================================

def clean_market_data(df):
    """Formats market names and rounds probabilities to 2 decimal places."""
    def fix_market_string(text):
        # Fix missing decimals in Overs/Unders (e.g., Over 15 -> Over 1.5)
        text = re.sub(r'(Over|Under)\s*(\d)(\d)', r'\1 \2.\3', str(text), flags=re.IGNORECASE)
        # Fix squished words (e.g., CornersHome -> Corners Home)
        text = re.sub(r'(Corners|Goals|Cards)([A-Za-z])', r'\1 \2', text)
        # Catch specific lingering errors from the data pipeline
        text = text.replace("Over05", "Over 0.5").replace("Over85", "Over 8.5")
        return text

    if 'Market' in df.columns:
        df['Market'] = df['Market'].apply(fix_market_string)
    
    if 'Model Probability (%)' in df.columns:
        df['Model Probability (%)'] = df['Model Probability (%)'].apply(lambda x: f"{float(x):.2f}")
        
    return df

def render_engine_status(is_connected=True):
    """Compact status widget replacing the long text line."""
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        if is_connected:
            st.markdown(
                """
                <div style='background-color: #1e3a29; color: #4ade80; padding: 6px 12px; 
                            border-radius: 6px; border: 1px solid #22c55e; font-size: 13px; 
                            text-align: center; font-weight: 600;'>
                    🟢 Data Engine Live
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style='background-color: #3b181b; color: #f87171; padding: 6px 12px; 
                            border-radius: 6px; border: 1px solid #ef4444; font-size: 13px; 
                            text-align: center; font-weight: 600; box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);'>
                    🔴 Activate Engine
                </div>
                """, unsafe_allow_html=True
            )

def refresh_app():
    """Dummy callback to force Streamlit to refresh when dropdowns change."""
    pass

# ==========================================
# 2. DATA DICTIONARY
# ==========================================

fixtures_data = {
    "Man City vs Sunderland": {
        "home": {
            "Team": "Man City", "def_score": 18, "duel_score": 210, "gk_score": 8, "threat_score": 25, "eye_test_score": 4.6,
            "xG": 2.10, "Avg_Goals_Conceded": 0.80, "yellow_cards_avg": 1.2, "red_cards_avg": 0.02, "creativity_avg": 180.0, "threat_avg": 190.0
        },
        "away": {
            "Team": "Sunderland", "def_score": 12, "duel_score": 160, "gk_score": 6, "threat_score": 15, "eye_test_score": 2.8,
            "xG": 1.10, "Avg_Goals_Conceded": 1.90, "yellow_cards_avg": 2.5, "red_cards_avg": 0.1, "creativity_avg": 90.0, "threat_avg": 85.0
        },
        "home_players": [
            {"name": "Phil Foden", "xG": 0.35, "xA": 0.40, "is_penalty_taker": False},
            {"name": "Erling Haaland", "xG": 0.85, "xA": 0.10, "is_penalty_taker": True}
        ],
        "away_players": [
            {"name": "Jack Clarke", "xG": 0.25, "xA": 0.20, "is_penalty_taker": True}
        ]
    },
    "Brentford vs Chelsea": {
        "home": {
            "Team": "Brentford", "def_score": 15, "duel_score": 200, "gk_score": 9, "threat_score": 19, "eye_test_score": 3.4,
            "xG": 1.60, "Avg_Goals_Conceded": 1.40, "yellow_cards_avg": 2.0, "red_cards_avg": 0.06, "creativity_avg": 125.0, "threat_avg": 115.0
        },
        "away": {
            "Team": "Chelsea", "def_score": 14, "duel_score": 195, "gk_score": 7, "threat_score": 21, "eye_test_score": 3.6,
            "xG": 1.65, "Avg_Goals_Conceded": 1.35, "yellow_cards_avg": 2.4, "red_cards_avg": 0.09, "creativity_avg": 155.0, "threat_avg": 145.0
        },
        "home_players": [
            {"name": "Bryan Mbeumo", "xG": 0.45, "xA": 0.20, "is_penalty_taker": True}
        ],
        "away_players": [
            {"name": "Cole Palmer", "xG": 0.50, "xA": 0.40, "is_penalty_taker": True}
        ]
    }
}

# ==========================================
# 3. SIDEBAR & STATE MANAGEMENT
# ==========================================

st.sidebar.header("⚽ 401 Predictor Tool")

# Binding dropdowns to state with on_change callbacks
league_select = st.sidebar.selectbox(
    "Competition", 
    ["English Premier League", "Spanish La Liga", "Italian Serie A"],
    key="league_state",
    on_change=refresh_app
)

gw_select = st.sidebar.selectbox(
    "Gameweek / Round", 
    [f"Gameweek {i}" for i in range(1, 39)], 
    index=3, # Defaults to GW4
    key="gw_state",
    on_change=refresh_app
)

# Fixture dropdown dynamically loads keys from the dictionary
available_matches = list(fixtures_data.keys())
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

# Active Match Data extraction
match_data = fixtures_data[selected_fixture_name]
home_team = match_data["home"]["Team"]
away_team = match_data["away"]["Team"]

# Engine Status
render_engine_status(is_connected=True)

# Cleaned Fixture Title Card
st.markdown(
    f"""
    <div style='background-color: #1e1e1e; padding: 15px; border-radius: 8px; 
                border-left: 5px solid #2563eb; margin-bottom: 20px; margin-top: 15px;'>
        <h2 style='margin: 0; padding: 0; font-size: 24px;'>{selected_fixture_name}</h2>
    </div>
    """, unsafe_allow_html=True
)

# --- A. Match Analyst Preview ---
st.subheader("🎙️ Match Analyst Preview")

# Cleaned of raw math/variables
analyst_text = f"**{home_team}** enter this matchup in strong dynamic form, driven by excellent defensive solidity and a high-volume attacking threat. **{away_team}** present an intriguing challenge, operating with high corner volume potential but showing higher disciplinary vulnerability."

st.markdown(
    f"<div style='color: #e2e8f0; font-size: 16px; line-height: 1.6; margin-bottom: 15px;'>{analyst_text}</div>", 
    unsafe_allow_html=True
)
st.markdown("🎯 **Best Suited Markets:** Double Chance (1X), Match Over 1.5 Goals, Corners Over 8.5")
st.markdown("---")


# --- B. Team Tactical Profiles (Now a Table) ---
st.subheader("🛡️ Team Tactical Profiles")

tactical_data = {
    "Metric": [
        "Dynamic Form", 
        "Defensive Solidity", 
        "Duel Success", 
        "Goalkeeper Rating", 
        "Goal Threat", 
        "Eye Test"
    ],
    home_team: [3.00, 4.2, 4.8, 3.67, 4.73, 4.6], 
    away_team: [3.00, 2.6, 1.8, 4.73, 2.47, 2.8]  
}
df_tactics = pd.DataFrame(tactical_data)
st.dataframe(df_tactics, hide_index=True, use_container_width=True)
st.markdown("---")


# --- C. Probable Outcomes (Decimals Fixed) ---
st.subheader("🔥 Top 5 Most Probable Outcomes")

# Generating mock data matching your current app
raw_top_5 = pd.DataFrame({
    "Market": ["DoubleChance 1X", "Over 15 Goals", "Corners Home", "HT Over05 Goals", "Corners Over85"],
    "Model Probability (%)": [85.9000, 84.1000, 81.4000, 77.3000, 74.0000],
    "Confidence Tier": ["🟢 Realistic (> 70%)", "🟢 Realistic (> 70%)", "🟢 Realistic (> 70%)", "🟢 Realistic (> 70%)", "🟢 Realistic (> 70%)"]
})

cleaned_top_5 = clean_market_data(raw_top_5)
st.table(cleaned_top_5)
st.markdown("---")


# --- D. Key Player Props ---
st.subheader("⚽ Key Player Props (Goalscorer & Assist Probabilities)")

col1, col2 = st.columns(2)
with col1:
    st.write(f"**{home_team}**")
    st.dataframe(pd.DataFrame(match_data["home_players"]), hide_index=True, use_container_width=True)

with col2:
    st.write(f"**{away_team}**")
    st.dataframe(pd.DataFrame(match_data["away_players"]), hide_index=True, use_container_width=True)