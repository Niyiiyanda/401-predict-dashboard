import streamlit as st
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="401 Predict Tool", page_icon="⚽", layout="wide")

# 2. Custom CSS for a modern dashboard look
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .elite-value {
        background-color: #1E3A8A;
        color: #60A5FA;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #3B82F6;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ 401 Predict Tool: Live +EV Dashboard")
st.markdown("Automated value betting engine tracking the Premier League.")

# 3. Your Live Google Sheets CSV Link
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT_psQkzyUQstywqkdRAMJqPyO63OZHiEvhY9V3Bko_gQEw8Yw4qTLttOQ9bbXKGZl1D0AG9atiOZ85/pub?gid=1906320941&single=true&output=csv"

# 4. Read the data directly
@st.cache_data(ttl=300)
def load_data():
    try:
        df = pd.read_csv(SHEET_CSV_URL)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df_markets = load_data()

# 5. Build the UI
if not df_markets.empty:
    st.header("🔥 Gameweek Value Plays")
    
    # Filter out the "PASS" cards
    if 'Action Trigger' in df_markets.columns:
        df_active = df_markets[~df_markets['Action Trigger'].astype(str).str.contains('PASS', na=False)]
    else:
        df_active = df_markets
    
    # Top-level metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total +EV Plays Found", len(df_active))
    
    highest_edge = "0%"
    if not df_active.empty and 'Edge' in df_active.columns:
        highest_edge = df_active['Edge'].max() 
        
    col2.metric("Highest Edge Detected", highest_edge)
    col3.metric("System Status", "Live & Tracking")
    
    st.divider()
    
    st.dataframe(
        df_active,
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("No data found. Run your Colab script to populate the sheet!")
