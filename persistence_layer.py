import json
import pandas as pd
import numpy as np

# ==============================================================================
# 401 PREDICT TOOL: PERSISTENCE LAYER & GOOGLE SHEETS CONNECTOR
# ==============================================================================

SHEET_NAME = "401 Predict Tool - Match & Coefficient Database"
TAB_BETTING_LOG = "Betting_Markets_Log"
TAB_PREDICTIONS_LOG = "Predictions_Log"
TAB_COEFFICIENT_HISTORY = "Sheet1"

def connect_gspread():
    """
    Establishes gspread connection supporting Streamlit Community Cloud (st.secrets),
    Google Colab default authentication, and local service account files.
    """
    # 1. Streamlit st.secrets (Production Cloud Environment)
    try:
        import streamlit as st
        import gspread

        if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            gc = gspread.service_account_from_dict(creds_dict)
            return gc.open(SHEET_NAME)
    except Exception:
        pass

    # 2. Google Colab Default Authentication
    try:
        from google.colab import auth
        from google.auth import default
        import gspread

        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        return gc.open(SHEET_NAME)
    except Exception:
        pass

    # 3. Local Service Account File fallback
    try:
        import gspread
        gc = gspread.service_account(filename="service_account.json")
        return gc.open(SHEET_NAME)
    except Exception:
        pass

    print("ℹ️ Offline / Fallback Mode: Unable to establish live Google Sheets connection.")
    return None


def load_gameweek_history(spreadsheet=None, tab_name=TAB_PREDICTIONS_LOG):
    """
    Loads historical gameweek coefficients and match results from Google Sheets.
    If offline or connection fails, loads local fallback dataset.
    """
    if spreadsheet:
        try:
            worksheet = spreadsheet.worksheet(tab_name)
            data = worksheet.get_all_records()
            if data:
                df = pd.DataFrame(data)
                print(f"✅ Loaded {len(df)} historical records from worksheet '{tab_name}'.")
                return df
        except Exception as e:
            print(f"⚠️ Error reading '{tab_name}' from Google Sheets: {e}")

    # Fallback to local baseline history
    print("ℹ️ Using local baseline Gameweek History dataset.")
    return generate_fallback_history()


def save_gameweek_results(df_results, spreadsheet=None, tab_name=TAB_BETTING_LOG):
    """
    Appends finalized Gameweek predictions and coefficient records to Google Sheets.
    """
    if spreadsheet:
        try:
            import gspread
            try:
                worksheet = spreadsheet.worksheet(tab_name)
            except gspread.exceptions.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=tab_name, rows="1000", cols="15")
                worksheet.append_row(df_results.columns.tolist())

            formatted_rows = df_results.astype(str).values.tolist()
            worksheet.append_rows(formatted_rows)
            print(f"🎉 Successfully saved {len(formatted_rows)} rows to Google Sheets tab '{tab_name}'.")
            return True
        except Exception as e:
            print(f"⚠️ Failed to save to Google Sheets: {e}")
            return False
    else:
        print("ℹ️ Local Mode: Results ready for sync upon cloud deployment.")
        return False


def get_historical_team_dcn(team_name, df_history):
    """
    Reconstructs prior Dynamic Running Coefficient (DC_N) for a team across all played GWs.
    Postponed / blank matches are safely ignored so divisor N does not increment.
    """
    if df_history.empty or "Team" not in df_history.columns or "C_W" not in df_history.columns:
        return 3.0  # Default neutral baseline

    team_rows = df_history[df_history["Team"] == team_name]
    if team_rows.empty:
        return 3.0

    # Extract valid C_W values
    cw_series = pd.to_numeric(team_rows["C_W"], errors="coerce").dropna()
    if cw_series.empty:
        return 3.0

    return round(float(cw_series.mean()), 2)


def generate_fallback_history():
    """Generates a structured historical dataset covering GW1 to GW4 for testing."""
    sample_history = [
        {"Gameweek": 1, "Team": "Arsenal", "C_W": 3.80, "Result": "Win", "xG": 2.10},
        {"Gameweek": 2, "Team": "Arsenal", "C_W": 3.90, "Result": "Win", "xG": 2.45},
        {"Gameweek": 3, "Team": "Arsenal", "C_W": None, "Result": "Postponed", "xG": None},
        {"Gameweek": 4, "Team": "Arsenal", "C_W": 3.60, "Result": "Win", "xG": 1.85},
        {"Gameweek": 1, "Team": "Brighton", "C_W": 3.20, "Result": "Win", "xG": 1.60},
        {"Gameweek": 2, "Team": "Brighton", "C_W": 2.90, "Result": "Loss", "xG": 0.95},
        {"Gameweek": 3, "Team": "Brighton", "C_W": 3.10, "Result": "Draw", "xG": 1.30},
        {"Gameweek": 4, "Team": "Brighton", "C_W": 3.20, "Result": "Win", "xG": 1.75},
    ]
    return pd.DataFrame(sample_history)


if __name__ == "__main__":
    print("Testing Persistence Layer Module...")
    ss = connect_gspread()
    df_hist = load_gameweek_history(ss)
    ars_dcn = get_historical_team_dcn("Arsenal", df_hist)
    bha_dcn = get_historical_team_dcn("Brighton", df_hist)
    print(f"Arsenal Reconstructed DC_N: {ars_dcn}")
    print(f"Brighton Reconstructed DC_N: {bha_dcn}")
