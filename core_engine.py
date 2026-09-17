import numpy as np
import pandas as pd
from scipy.stats import poisson, norm, skellam

# ==============================================================================
# 401 PREDICT TOOL: COMPREHENSIVE FPL-POWERED CORE ENGINE
# Incorporating Cards, Corners, Player Props, Half-Time, and Goal/Match Markets
# ==============================================================================

def normalize_to_5_scale(val, min_val, max_val, reverse=False):
    """
    Normalizes a raw performance metric onto a strict 1.0 to 5.0 scale.
    Formula: 1.0 + ((val - min_val) / (max_val - min_val)) * 4.0
    """
    if max_val == min_val:
        return 3.0
    scaled = 1.0 + ((val - min_val) / (max_val - min_val)) * 4.0
    if reverse:
        scaled = 6.0 - scaled
    return float(np.clip(scaled, 1.0, 5.0))


def compute_weekly_coefficients(team_stats_dict):
    """
    Computes V1-V5 on a 1.0-5.0 scale and calculates the unweighted Weekly Coefficient (C_W).
    
    Variables:
      V1: Defensive Solidity (xGA / Def Score)
      V2: Duel Success (Duels Won % / BPS Proxy)
      V3: Goalkeeper Rating (Save % + Keeper Rating)
      V4: Goal Threat (xG / Shot Accuracy + Big Chances)
      V5: Eye Test (Qualitative Tactical Observation Score 1.0-5.0)
    """
    v1 = normalize_to_5_scale(team_stats_dict.get("def_score", 0), min_val=0, max_val=25)
    v2 = normalize_to_5_scale(team_stats_dict.get("duel_score", 100), min_val=100, max_val=300)
    v3 = normalize_to_5_scale(team_stats_dict.get("gk_score", 0), min_val=0, max_val=15)
    v4 = normalize_to_5_scale(team_stats_dict.get("threat_score", 0), min_val=0, max_val=30)
    
    # V5 Eye Test is provided directly on a 1.0-5.0 scale
    v5 = float(np.clip(team_stats_dict.get("eye_test_score", 3.0), 1.0, 5.0))

    # C_W is the UNWEIGHTED mean of all 5 variables
    cw = (v1 + v2 + v3 + v4 + v5) / 5.0

    return {
        "V1_Defense": round(v1, 2),
        "V2_Duel": round(v2, 2),
        "V3_GK": round(v3, 2),
        "V4_Threat": round(v4, 2),
        "V5_EyeTest": round(v5, 2),
        "C_W": round(cw, 2)
    }


def update_dynamic_running_coefficient(historical_cw_list):
    """
    Calculates Dynamic Running Coefficient (DC_N) across played gameweeks.
    SAFEGUARD RULE: Ignores blank/postponed gameweeks (represented as None or np.nan).
    The divisor N only increments for played matches.
    """
    valid_cw = [cw for cw in historical_cw_list if cw is not None and not np.isnan(cw)]
    if not valid_cw:
        return 3.0  # Baseline neutral
    return round(float(np.mean(valid_cw)), 2)


def calculate_comprehensive_probabilities(home_stats, away_stats, home_players=None, away_players=None):
    """
    Calculates exact mathematical probabilities across all Glossaries:
      1. Match Result & Goals (1X2, Double Chance, Over/Under 1.5, 2.5, BTTS, Clean Sheets)
      2. Half-Time Markets (1st Half Goals Over 0.5, 1.5, 1st Half 1X2)
      3. Disciplinary Markets (Total Cards, Booking Points, Red Card Yes/No, Bookings 1X2)
      4. Corner Markets (Total Corners Over 8.5/9.5/10.5, Corners 1X2)
      5. Player Props (Anytime Goalscorer, Player Assists, Shots on Target)
    """
    # --------------------------------------------------------------------------
    # 1. MATCH RESULT & GOAL MARKETS (Poisson & xG)
    # --------------------------------------------------------------------------
    xg_home = home_stats.get("xG", (home_stats.get("Avg_Goals_Scored", 1.8) + away_stats.get("Avg_Goals_Conceded", 1.2)) / 2.0)
    xg_away = away_stats.get("xG", (away_stats.get("Avg_Goals_Scored", 1.2) + home_stats.get("Avg_Goals_Conceded", 0.8)) / 2.0)

    p_home = [poisson.pmf(i, xg_home) for i in range(6)]
    p_away = [poisson.pmf(i, xg_away) for i in range(6)]

    p_home_0, p_away_0 = p_home[0], p_away[0]
    p_home_1, p_away_1 = p_home[1], p_away[1]

    # 1X2
    prob_home_win = sum(p_home[h] * sum(p_away[:h]) for h in range(1, 6)) * 100.0
    prob_draw = sum(p_home[i] * p_away[i] for i in range(6)) * 100.0
    prob_away_win = sum(p_away[a] * sum(p_home[:a]) for a in range(1, 6)) * 100.0

    # Goal Totals & BTTS
    prob_btts = (1.0 - p_home_0) * (1.0 - p_away_0) * 100.0
    prob_match_over15 = (1.0 - (p_home_0 * p_away_0) - (p_home_1 * p_away_0) - (p_home_0 * p_away_1)) * 100.0
    prob_match_over25 = (1.0 - sum(p_home[h] * p_away[a] for h in range(6) for a in range(6) if h + a <= 2)) * 100.0
    
    # Clean Sheets
    prob_clean_sheet_home = p_away_0 * 100.0
    prob_clean_sheet_away = p_home_0 * 100.0

    # Win to Nil
    prob_win_to_nil_home = prob_home_win * p_away_0
    prob_win_to_nil_away = prob_away_win * p_home_0

    # --------------------------------------------------------------------------
    # 2. HALF-TIME MARKETS (45% goal distribution model)
    # --------------------------------------------------------------------------
    lambda_ht_home = 0.45 * xg_home
    lambda_ht_away = 0.45 * xg_away

    p_ht_home = [poisson.pmf(i, lambda_ht_home) for i in range(4)]
    p_ht_away = [poisson.pmf(i, lambda_ht_away) for i in range(4)]

    prob_ht_over05 = (1.0 - (p_ht_home[0] * p_ht_away[0])) * 100.0
    prob_ht_over15 = (1.0 - (p_ht_home[0] * p_ht_away[0]) - (p_ht_home[1] * p_ht_away[0]) - (p_ht_home[0] * p_ht_away[1])) * 100.0

    # --------------------------------------------------------------------------
    # 3. DISCIPLINARY MARKETS (Yellow Cards, Red Cards, Booking Points)
    # --------------------------------------------------------------------------
    yc_home = home_stats.get("yellow_cards_avg", 1.8)
    yc_away = away_stats.get("yellow_cards_avg", 2.1)
    rc_home = home_stats.get("red_cards_avg", 0.08)
    rc_away = away_stats.get("red_cards_avg", 0.10)

    # Total Expected Cards & Booking Points (Yellow = 10 pts, Red = 25 pts)
    exp_cards_home = yc_home + rc_home
    exp_cards_away = yc_away + rc_away
    exp_match_cards = exp_cards_home + exp_cards_away

    exp_booking_pts_home = (yc_home * 10) + (rc_home * 25)
    exp_booking_pts_away = (yc_away * 10) + (rc_away * 25)
    exp_match_booking_pts = exp_booking_pts_home + exp_booking_pts_away

    # Card Over/Under Probabilities (Poisson)
    prob_cards_over35 = (1.0 - poisson.cdf(3, exp_match_cards)) * 100.0
    prob_cards_over45 = (1.0 - poisson.cdf(4, exp_match_cards)) * 100.0

    # Red Card (Sending Off Yes/No) Probability
    prob_red_card_yes = (1.0 - np.exp(-(rc_home + rc_away))) * 100.0

    # Bookings 1X2 (Skellam Distribution)
    prob_home_more_cards = (1.0 - skellam.cdf(0, exp_cards_home, exp_cards_away)) * 100.0
    prob_cards_draw = skellam.pmf(0, exp_cards_home, exp_cards_away) * 100.0
    prob_away_more_cards = skellam.cdf(-1, exp_cards_home, exp_cards_away) * 100.0

    # --------------------------------------------------------------------------
    # 4. CORNER MARKETS (ICT Creativity & Threat Proxies)
    # --------------------------------------------------------------------------
    creativity_home = home_stats.get("creativity_avg", 150.0)
    threat_home = home_stats.get("threat_avg", 140.0)
    creativity_away = away_stats.get("creativity_avg", 120.0)
    threat_away = away_stats.get("threat_avg", 110.0)

    # Linear proxy model: Base corners + Creativity/Threat influence
    lambda_corners_home = max(2.5, 2.0 + (0.015 * creativity_home) + (0.010 * threat_home))
    lambda_corners_away = max(2.0, 1.5 + (0.015 * creativity_away) + (0.010 * threat_away))
    lambda_corners_match = lambda_corners_home + lambda_corners_away

    prob_corners_over85 = (1.0 - poisson.cdf(8, lambda_corners_match)) * 100.0
    prob_corners_over95 = (1.0 - poisson.cdf(9, lambda_corners_match)) * 100.0
    prob_corners_over105 = (1.0 - poisson.cdf(10, lambda_corners_match)) * 100.0

    # Corners 1X2 (Skellam Distribution)
    prob_home_more_corners = (1.0 - skellam.cdf(0, lambda_corners_home, lambda_corners_away)) * 100.0
    prob_corners_draw = skellam.pmf(0, lambda_corners_home, lambda_corners_away) * 100.0
    prob_corners_away = skellam.cdf(-1, lambda_corners_home, lambda_corners_away) * 100.0

    # --------------------------------------------------------------------------
    # 5. PLAYER PROPS (Anytime Goalscorer & Assists)
    # --------------------------------------------------------------------------
    player_prop_outcomes = []

    def evaluate_players(player_list, team_name):
        if not player_list:
            return
        for p in player_list:
            p_name = p.get("name", "Player")
            xg_p = p.get("xG", 0.35)
            xa_p = p.get("xA", 0.20)
            is_pen_taker = p.get("is_penalty_taker", False)
            
            # Adjusted xG for penalty duties
            xg_adj = xg_p + (0.20 if is_pen_taker else 0.0)

            prob_score = (1.0 - np.exp(-xg_adj)) * 100.0
            prob_assist = (1.0 - np.exp(-xa_p)) * 100.0

            player_prop_outcomes.append({
                "Player": f"{p_name} ({team_name})",
                "Anytime_Goalscorer_Prob": round(prob_score, 1),
                "Anytime_Assist_Prob": round(prob_assist, 1)
            })

    evaluate_players(home_players, home_stats.get("Team", "Home"))
    evaluate_players(away_players, away_stats.get("Team", "Away"))

    # Return Comprehensive Probabilities Dictionary
    return {
        # Match Result & Goals
        "1X2_Home_Win": round(prob_home_win, 1),
        "1X2_Draw": round(prob_draw, 1),
        "1X2_Away_Win": round(prob_away_win, 1),
        "DoubleChance_1X": round(prob_home_win + prob_draw, 1),
        "DoubleChance_X2": round(prob_away_win + prob_draw, 1),
        "Over_15_Goals": round(prob_match_over15, 1),
        "Over_25_Goals": round(prob_match_over25, 1),
        "BTTS_GG": round(prob_btts, 1),
        "Home_CleanSheet": round(prob_clean_sheet_home, 1),
        "Away_CleanSheet": round(prob_clean_sheet_away, 1),
        "Home_Win_To_Nil": round(prob_win_to_nil_home, 1),
        "Away_Win_To_Nil": round(prob_win_to_nil_away, 1),

        # Half-Time Markets
        "HT_Over05_Goals": round(prob_ht_over05, 1),
        "HT_Over15_Goals": round(prob_ht_over15, 1),

        # Disciplinary Markets
        "Exp_Match_Cards": round(exp_match_cards, 2),
        "Exp_Booking_Points": round(exp_match_booking_pts, 1),
        "Cards_Over35": round(prob_cards_over35, 1),
        "Cards_Over45": round(prob_cards_over45, 1),
        "Sending_Off_RedCard_Yes": round(prob_red_card_yes, 1),
        "Bookings_1X2_Home": round(prob_home_more_cards, 1),
        "Bookings_1X2_Draw": round(prob_cards_draw, 1),
        "Bookings_1X2_Away": round(prob_away_more_cards, 1),

        # Corner Markets
        "Exp_Match_Corners": round(lambda_corners_match, 1),
        "Corners_Over85": round(prob_corners_over85, 1),
        "Corners_Over95": round(prob_corners_over95, 1),
        "Corners_Over105": round(prob_corners_over105, 1),
        "Corners_1X2_Home": round(prob_home_more_corners, 1),
        "Corners_1X2_Draw": round(prob_corners_draw, 1),
        "Corners_1X2_Away": round(prob_corners_away, 1),

        # Player Props List
        "Player_Props": player_prop_outcomes
    }
