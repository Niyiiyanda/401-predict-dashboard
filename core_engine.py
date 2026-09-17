import pandas as pd
import numpy as np
from scipy.stats import poisson, norm, skellam

# ==============================================================================
# 401 PREDICT TOOL: CORE MATHEMATICAL ENGINE (1.0 to 10.0 & 1.0 to 5.0 SCALES)
# ==============================================================================

def normalize_to_scale(val, min_val, max_val, scale=10.0, reverse=False):
    """
    Normalizes a raw metric onto a specified scale (default 1.0 to 10.0).
    Formula: 1.0 + ((val - min_val) / (max_val - min_val)) * (scale - 1.0)
    """
    if max_val == min_val:
        return round((scale + 1.0) / 2.0, 2)
    scaled = 1.0 + ((val - min_val) / (max_val - min_val)) * (scale - 1.0)
    if reverse:
        scaled = (scale + 1.0) - scaled
    return float(np.clip(round(scaled, 2), 1.0, scale))


def compute_weekly_coefficients(team_stats_dict, scale=10.0):
    """
    Computes V1-V5 on the specified scale (1.0-10.0 or 1.0-5.0) 
    and calculates the unweighted Weekly Coefficient (C_W).
    
    Variables:
      V1: Defensive Solidity
      V2: Duel Success
      V3: Goalkeeper Resistance
      V4: Goal Threat Efficiency
      V5: Qualitative Eye Test
    """
    v1 = normalize_to_scale(team_stats_dict.get("def_score", 0), min_val=0, max_val=25, scale=scale)
    v2 = normalize_to_scale(team_stats_dict.get("duel_score", 100), min_val=100, max_val=300, scale=scale)
    v3 = normalize_to_scale(team_stats_dict.get("gk_score", 0), min_val=0, max_val=15, scale=scale)
    v4 = normalize_to_scale(team_stats_dict.get("threat_score", 0), min_val=0, max_val=30, scale=scale)
    
    # Eye test score (if on 1-5 scale, re-scale if scale=10)
    raw_eye = team_stats_dict.get("eye_test_score", 3.0)
    if scale == 10.0 and raw_eye <= 5.0:
        v5 = float(np.clip(round(raw_eye * 2.0, 2), 1.0, 10.0))
    else:
        v5 = float(np.clip(round(raw_eye, 2), 1.0, scale))

    # C_W is the unweighted arithmetic mean of V1-V5
    cw = (v1 + v2 + v3 + v4 + v5) / 5.0

    return {
        "V1_Defense": round(v1, 2),
        "V2_Duel": round(v2, 2),
        "V3_GK": round(v3, 2),
        "V4_Threat": round(v4, 2),
        "V5_EyeTest": round(v5, 2),
        "C_W": round(cw, 2)
    }


def update_dynamic_running_coefficient(historical_cw_list, default_neutral=6.0):
    """
    Calculates Dynamic Running Coefficient (DC_N) across played gameweeks.
    SAFEGUARD RULE: Ignores blank/postponed matches (None / NaN). 
    The divisor N only increments for played matches.
    """
    valid_cw = [cw for cw in historical_cw_list if cw is not None and not np.isnan(cw)]
    if not valid_cw:
        return default_neutral
    return round(float(np.mean(valid_cw)), 2)


def calculate_comprehensive_probabilities(home_stats, away_stats, home_players=None, away_players=None):
    """
    Calculates probabilities across 21 distinct betting outcome categories 
    using Poisson, Skellam, Normal Z-Score, and Zero-Event Exponential models.
    Returns at least 12 and up to 21 market keys in every match payload.
    """
    # 1. Expected Goal Poisson Expectations
    h_scored = home_stats.get("xG", home_stats.get("Avg_Goals_Scored", 1.6))
    a_conceded = away_stats.get("Avg_Goals_Conceded", 1.1)
    a_scored = away_stats.get("xG", away_stats.get("Avg_Goals_Scored", 1.2))
    h_conceded = home_stats.get("Avg_Goals_Conceded", 0.9)

    lambda_home = (h_scored + a_conceded) / 2.0
    lambda_away = (a_scored + h_conceded) / 2.0

    # Poisson Goal Distributions (0 to 5 goals)
    p_home = [poisson.pmf(i, lambda_home) for i in range(6)]
    p_away = [poisson.pmf(i, lambda_away) for i in range(6)]

    p_h0, p_a0 = p_home[0], p_away[0]
    p_h1, p_a1 = p_home[1], p_away[1]

    # --- Market 1, 2, 3: Match Winner 1X2 ---
    prob_home_win = sum(p_home[h] * sum(p_away[:h]) for h in range(1, 6)) * 100.0
    prob_draw = sum(p_home[i] * p_away[i] for i in range(6)) * 100.0
    prob_away_win = sum(p_away[a] * sum(p_home[:a]) for a in range(1, 6)) * 100.0

    # Non-draw sum for Draw No Bet
    non_draw_sum = max(1.0 - (prob_draw / 100.0), 0.01)

    # --- Market 4, 5, 6: Double Chance ---
    prob_dc_1x = prob_home_win + prob_draw
    prob_dc_x2 = prob_away_win + prob_draw
    prob_dc_12 = prob_home_win + prob_away_win

    # --- Market 7, 8: Draw No Bet (DNB) ---
    prob_dnb_home = (prob_home_win / 100.0 / non_draw_sum) * 100.0
    prob_dnb_away = (prob_away_win / 100.0 / non_draw_sum) * 100.0

    # --- Market 9, 10: First Half Goals ---
    ht_lambda_home = lambda_home * 0.45
    ht_lambda_away = lambda_away * 0.45
    ht_lambda_total = ht_lambda_home + ht_lambda_away
    
    prob_ht_over05 = (1.0 - np.exp(-ht_lambda_total)) * 100.0
    prob_ht_over15 = (1.0 - (poisson.pmf(0, ht_lambda_home)*poisson.pmf(0, ht_lambda_away) +
                             poisson.pmf(1, ht_lambda_home)*poisson.pmf(0, ht_lambda_away) +
                             poisson.pmf(0, ht_lambda_home)*poisson.pmf(1, ht_lambda_away))) * 100.0

    # --- Market 11, 12, 13, 14: Match Goal Totals ---
    prob_over15 = (1.0 - (p_h0 * p_a0) - (p_h1 * p_a0) - (p_h0 * p_a1)) * 100.0
    prob_under25 = sum(p_home[h] * p_away[a] for h in range(6) for a in range(6) if h + a <= 2) * 100.0
    prob_over25 = 100.0 - prob_under25
    prob_over35 = (1.0 - sum(p_home[h] * p_away[a] for h in range(6) for a in range(6) if h + a <= 3)) * 100.0

    # --- Market 15, 16: Both Teams To Score (BTTS) ---
    prob_btts_yes = (1.0 - p_h0) * (1.0 - p_a0) * 100.0
    prob_btts_no = 100.0 - prob_btts_yes

    # --- Market 17, 18: Clean Sheets ---
    prob_cs_home = p_a0 * 100.0
    prob_cs_away = p_h0 * 100.0

    # --- Market 19: Win to Nil ---
    prob_w2n_home = (prob_home_win / 100.0) * p_a0 * 100.0

    # --- Market 20: Corners Over 8.5 ---
    corners_h = home_stats.get("creativity_avg", 150.0) / 25.0
    corners_a = away_stats.get("creativity_avg", 120.0) / 25.0
    lambda_corners = corners_h + corners_a
    prob_corners_over85 = (1.0 - sum(poisson.pmf(k, lambda_corners) for k in range(9))) * 100.0

    # --- Market 21: Cards Over 3.5 ---
    cards_h = home_stats.get("yellow_cards_avg", 1.8)
    cards_a = away_stats.get("yellow_cards_avg", 2.1)
    lambda_cards = cards_h + cards_a
    prob_cards_over35 = (1.0 - sum(poisson.pmf(k, lambda_cards) for k in range(4))) * 100.0

    # Player Props calculation
    player_props_list = []
    if home_players:
        for p in home_players:
            xg = p.get("xG", 0.3)
            xa = p.get("xA", 0.2)
            if p.get("is_penalty_taker", False):
                xg = xg + 0.15
            p_goal = (1.0 - np.exp(-xg)) * 100.0
            p_assist = (1.0 - np.exp(-xa)) * 100.0
            player_props_list.append({
                "Player": p.get("name", "Player"),
                "Team": home_stats.get("Team", "Home"),
                "Anytime_Goalscorer_Prob": round(p_goal, 1),
                "Anytime_Assist_Prob": round(p_assist, 1)
            })
            
    if away_players:
        for p in away_players:
            xg = p.get("xG", 0.3)
            xa = p.get("xA", 0.2)
            if p.get("is_penalty_taker", False):
                xg = xg + 0.15
            p_goal = (1.0 - np.exp(-xg)) * 100.0
            p_assist = (1.0 - np.exp(-xa)) * 100.0
            player_props_list.append({
                "Player": p.get("name", "Player"),
                "Team": away_stats.get("Team", "Away"),
                "Anytime_Goalscorer_Prob": round(p_goal, 1),
                "Anytime_Assist_Prob": round(p_assist, 1)
            })

    # Return Payload with 21 core market outcome keys + auxiliary expectations
    return {
        # 21 Core Market Outcome Keys
        "1X2_Home_Win": round(prob_home_win, 1),
        "1X2_Draw": round(prob_draw, 1),
        "1X2_Away_Win": round(prob_away_win, 1),
        "DoubleChance_1X": round(prob_dc_1x, 1),
        "DoubleChance_X2": round(prob_dc_x2, 1),
        "DoubleChance_12": round(prob_dc_12, 1),
        "Draw_No_Bet_Home": round(prob_dnb_home, 1),
        "Draw_No_Bet_Away": round(prob_dnb_away, 1),
        "1stHalf_Over_05_Goals": round(prob_ht_over05, 1),
        "1stHalf_Over_15_Goals": round(prob_ht_over15, 1),
        "Over_15_Match_Goals": round(prob_over15, 1),
        "Over_25_Match_Goals": round(prob_over25, 1),
        "Over_35_Match_Goals": round(prob_over35, 1),
        "Under_25_Match_Goals": round(prob_under25, 1),
        "BTTS_Yes": round(prob_btts_yes, 1),
        "BTTS_No": round(prob_btts_no, 1),
        "Home_CleanSheet": round(prob_cs_home, 1),
        "Away_CleanSheet": round(prob_cs_away, 1),
        "Home_Win_To_Nil": round(prob_w2n_home, 1),
        "Corners_Over_85": round(prob_corners_over85, 1),
        "Cards_Over_35": round(prob_cards_over35, 1),
        
        # Auxiliary Expectations & Player Props
        "Exp_Home_Goals": round(lambda_home, 2),
        "Exp_Away_Goals": round(lambda_away, 2),
        "Exp_Match_Corners": round(lambda_corners, 1),
        "Exp_Match_Cards": round(lambda_cards, 1),
        "Player_Props": player_props_list
    }


# Backwards compatibility function aliases
def calculate_comprehensive_match_probabilities(home_stats, away_stats):
    return calculate_comprehensive_probabilities(home_stats, away_stats)

def calculate_poisson_and_normal_probabilities(home_stats, away_stats):
    return calculate_comprehensive_probabilities(home_stats, away_stats)
