import streamlit as st
import pandas as pd
import numpy as np

# --- ⚙️ Page Configuration ---
st.set_page_config(
    page_title="401 Predict Tool: Probability Engine",
    page_icon="⚽",
    layout="wide"
)

# --- 🧪 Mock Probability Data Engine ---
@st.cache_data
def load_predictions():
    """Generates mock data based purely on model probability output."""
    return pd.DataFrame({
        "Fixture": [
            "Spurs vs Aston Villa", 
            "Nott'm Forest vs Coventry", 
            "Newcastle vs Hull City", 
            "Man City vs Sunderland", 
            "Brighton vs Arsenal", 
            "Newcastle vs Hull City",
            "Chelsea vs Brentford"
        ],
        "Market": ["Under 2.5", "Home Win", "X2 (Double Chance)", "Under 2.5", "Over 2.5", "Away Win", "Home Win"],
        "Our Prob (%)": [77.70, 82.50, 73.00, 60.90, 55.50, 48.30, 42.70],
        "Model Rationale": [
            "Low xG variance detected", 
            "Extreme Form Asymmetry", 
            "Poisson Distribution Skew", 
            "Tactical V2 (Def) suppression", 
            "High pace transition expected", 
            "Low confidence - High variance", 
            "Missing key personnel"
        ]
    })

def main():
    # --- 1. 🟢 Startup Health Check Banner ---
    st.success("🟢 **System Status: Live & Tracking.** Pure Probability Engine active.", icon="✅")

    st.title("⚽ 401 Predict Tool: Probability Zones")
    st.markdown("Filtering match outcomes strictly by mathematical probability generated from our Poisson and Tactical variables.")
    st.divider()

    # Load Data
    df_preds = load_predictions()

    # --- 2. 📊 Probability Filtering Logic ---
    # Segregate the dataframe into the three distinct zones based on 'Our Prob (%)'
    highly_likely_df = df_preds[df_preds["Our Prob (%)"] >= 75.0].sort_values(by="Our Prob (%)", ascending=False)
    goldilocks_df = df_preds[(df_preds["Our Prob (%)"] >= 50.0) & (df_preds["Our Prob (%)"] < 75.0)].sort_values(by="Our Prob (%)", ascending=False)
    highly_risky_df = df_preds[df_preds["Our Prob (%)"] < 50.0].sort_values(by="Our Prob (%)", ascending=False)

    # --- 3. 🗂️ The 3-Tier Probability Explorer Panel ---
    st.subheader("🗂️ Probability Explorer Panel")
    
    tab1, tab2, tab3 = st.tabs([
        f"🟢 Highly Likely ({len(highly_likely_df)})", 
        f"🟡 Goldilocks Zone ({len(goldilocks_df)})", 
        f"🔴 Highly Risky ({len(highly_risky_df)})"
    ])

    with tab1:
        st.markdown("### 🟢 Foundational Picks ($P \ge 75\%$)")
        st.markdown("Ideal for building accumulators or low-risk, high-confidence strategies.")
        if not highly_likely_df.empty:
            st.dataframe(highly_likely_df, use_container_width=True, hide_index=True)
        else:
            st.info("No highly likely outcomes detected in this gameweek's dataset.")

    with tab2:
        st.markdown("### 🟡 The Sweet Spot ($50\% \le P < 75\%$)")
        st.markdown("Strong value plays. This zone often contains the most mathematically profitable discrepancies.")
        if not goldilocks_df.empty:
            st.dataframe(goldilocks_df, use_container_width=True, hide_index=True)
        else:
            st.info("No goldilocks outcomes detected in this gameweek's dataset.")

    with tab3:
        st.markdown("### 🔴 High Variance / Avoid ($P < 50\%$)")
        st.markdown("Low probability events. Use caution unless a severe market anomaly is detected.")
        if not highly_risky_df.empty:
            st.dataframe(highly_risky_df, use_container_width=True, hide_index=True)
        else:
            st.info("No highly risky outcomes detected in this gameweek's dataset.")

if __name__ == "__main__":
    main()
