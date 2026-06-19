import numpy as np

# Input/output paths
INPUT_DIRECTORY = 'tennis_atp'
OUTPUT_DIRECTORY = 'processed_data'

# Years used for training - upper bound is exclusive
YEARS = range(2020, 2025)

# Elo parameters
INITIAL_ELO = 1200 # The starting Elo value
K_FACTOR = 40      # The maximum increase/decrease in rating from 1 match

# Rolling config - represented as a dictionary of dictionaries
ROLLING_CONFIG = {
    'serve_win_ratio'       : { 'window' : 10, 'min_periods' : 2},
    'first_serve_ratio'     : { 'window' : 10, 'min_periods' : 2},
    'first_serve_win_ratio' : { 'window' : 10, 'min_periods' : 2},
    'bp_conversion'         : { 'window' : 10, 'min_periods' : 2},
    'bp_save_rate'          : { 'window' : 10, 'min_periods' : 2},
    'ace_rate'              : { 'window' : 10, 'min_periods' : 2},
    'double_fault_rate'     : { 'window' : 10, 'min_periods' : 2},
}
SURFACE_WIN_RATIO_CONFIG = { 'window' : 10, 'min_periods' : 2 }


# Exponential moving-average for current form
RECENT_FORM_EMA_SPAN = 5
RECENT_FORM_EMA_MIN_PERIODS = 3

# How far to look back when considering a win streak
WIN_STREAK_WINDOW = 10

# How many days to look back when calculating rank changes
RANK_LOOKBACK_DAYS = 30

# On a player's first match in the dataset, what is their previous match?
DEFAULT_DAYS_SINCE_MATCH = 14

# Numerical label encoding - these are ordinal, which works fine for Decision Trees
ENTRY_TYPE_MAP = {
    np.nan : 0,    # Standard entry
    'SE'   : 1,    # (S)pecial (E)xempt
    'PR'   : 2,    # (P)rotected (R)anking
    'UP'   : 2,    # (U)sed (P)rotection - similar to PR, tends to be high class players
    'Q'    : 3,    # (Q)ualifier
    'WC'   : 4,    # (W)ild (C)ard
    'LL'   : 5,    # (L)ucky (L)oser
    'Alt'  : 6,    # (Alt)ernate
    'ALT'  : 6,    # (ALT)ernate
    'ITF'  : 6,    # ITF tour - generally newer players, unfavourable
    'W'    : 6
}

# Encodings for surfaces
SURFACE_MAP = {
    np.nan : 0,
    'Hard' : 1,
    'Clay' : 2,
    'Grass' : 3
}

# Tournament mapping - from most to least prestigious
TOURNAMENT_LEVEL_MAP = {
    'G': 0,  # Grand Slam
    'F': 1,  # Finals
    'M': 2,  # Masters
    'A': 3,  # ATP Tour
    'O': 4,  # Other
    'D': 5   # Davis Cup
}

# All columns available in the unprocessed dataset
COLUMN_NAMES = [
    'tournament_id',
    'tournament_name',
    'surface',
    'draw_size',
    'tournament_level',
    'tournament_date',
    'match_number',
    'winner_id',
    'winner_seed',
    'winner_entry_type',
    'winner_name',
    'winner_hand',
    'winner_height_cm',
    'winner_country_code',
    'winner_age',
    'loser_id',
    'loser_seed',
    'loser_entry_type',
    'loser_name',
    'loser_hand',
    'loser_height_cm',
    'loser_country_code',
    'loser_age',
    'score',
    'best_of_sets',
    'round',
    'match_duration_minutes',
    'winner_aces',
    'winner_double_faults',
    'winner_service_points_played',
    'winner_first_serves_in',
    'winner_first_serve_points_won',
    'winner_second_serve_points_won',
    'winner_service_games_played',
    'winner_break_points_saved',
    'winner_break_points_faced',
    'loser_aces',
    'loser_double_faults',
    'loser_service_points_played',
    'loser_first_serves_in',
    'loser_first_serve_points_won',
    'loser_second_serve_points_won',
    'loser_service_games_played',
    'loser_break_points_saved',
    'loser_break_points_faced',
    'winner_rank',
    'winner_rank_points',
    'loser_rank',
    'loser_rank_points'
]

NEUTRAL_COLUMNS = [
    'unique_match_id', 
    'tournament_date', 
    'surface', 
    'tournament_level', 
    'best_of_sets'
]

# Columns which should be relative rather than separate for A and B
DIFFERENTIAL_COLUMNS = [
    'rank',
    'rank_points',
    'elo',
    'age',
    'height_cm',
    'rolling_serve_win_ratio',
    'rolling_first_serve_ratio',
    'rolling_first_serve_win_ratio',
    'rolling_bp_conversion',
    'rolling_bp_save_rate',
    'rolling_surface_win_ratio',
    'rolling_ace_rate',
    'rolling_double_fault_rate',
    'recent_form_ema',
    'win_streak',
    'rank_change_30d',
    'days_since_last_match',
]

# Final feature columns for model training
FEATURE_COLUMNS = [
    'A_entry_type',
    'A_hand',
    'A_has_seed',
    'B_entry_type',
    'B_hand',
    'B_has_seed',
    'surface',
    'tournament_level',
    'best_of_sets',
    'rank_diff',
    'rank_points_diff',
    'elo_diff',
    'age_diff',
    'height_cm_diff',
    'rolling_serve_win_ratio_diff',
    'rolling_first_serve_ratio_diff',
    'rolling_first_serve_win_ratio_diff',
    'rolling_bp_conversion_diff',
    'rolling_bp_save_rate_diff',
    'rolling_surface_win_ratio_diff',
    'rolling_ace_rate_diff',
    'rolling_double_fault_rate_diff',
    'recent_form_ema_diff',
    'win_streak_diff',
    'rank_change_30d_diff',
    'days_since_last_match_diff'
]

# Numeric columns that may have NaN values to average out...
NUMERIC_NAN_COLUMNS = [
    'ace_rate', 
    'aces', 
    'age', 
    'bp_conversion', 
    'bp_save_rate', 
    'break_points_faced', 
    'break_points_saved', 
    'double_fault_rate', 
    'double_faults', 
    'first_serve_points_won', 
    'first_serve_ratio', 
    'first_serve_win_ratio', 
    'first_serves_in', 
    'height_cm', 
    'rank', 
    'rank_points', 
    'recent_form_ema', 
    'rolling_bp_conversion', 
    'rolling_bp_save_rate', 
    'rolling_first_serve_ratio', 
    'rolling_first_serve_win_ratio', 
    'rolling_serve_win_ratio', 
    'rolling_surface_win_ratio',
    'rolling_ace_rate', 
    'rolling_double_fault_rate',
    'second_serve_points_won', 
    'second_serve_win_ratio', 
    'serve_win_ratio', 
    'service_games_played', 
    'service_points_played', 
    'win_streak'
]

DEFAULT_DAYS_SINCE_LAST_MATCH = 14