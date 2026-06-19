import pandas as pd
import logging

import config

class FeatureEngineer:
    """
    Creates features from raw match data
    """

    def create_unique_match_id(self, matches):
        matches['unique_match_id'] = matches['tournament_id'] + '_' + matches['match_number'].astype(str)
        return matches

    def create_opposite_bp_stats(self, matches):
        # These stats depend on opposite player's break point stats, so need to be processed in wide format
        matches['winner_bp_conversion'] = (
            matches['loser_break_points_faced'] - matches['loser_break_points_saved']
        ) / matches['loser_break_points_faced']
        
        matches['loser_bp_conversion'] = (
            matches['winner_break_points_faced'] - matches['winner_break_points_saved']
        ) / matches['winner_break_points_faced']  
        return matches
    
    def convert_to_long_format(self, matches):
        ''
        logging.info('Converting matches to long format')
        
        neutral_columns = config.NEUTRAL_COLUMNS.copy()
        
        # Prepare winner data
        winner_columns = [col for col in matches.columns if 'winner' in col] + neutral_columns
        winners = matches[winner_columns].copy()
        winners.columns = [col.replace('winner_', '') for col in winners.columns]
        winners['outcome'] = 1
        
        # Prepare loser data
        loser_columns = [col for col in matches.columns if 'loser' in col] + neutral_columns
        losers = matches[loser_columns].copy()
        losers.columns = [col.replace('loser_', '') for col in losers.columns]
        losers['outcome'] = 0
        
        # Combine and sort
        long_data = pd.concat([winners, losers], ignore_index=True)
        long_data = long_data.sort_values(['tournament_date', 'unique_match_id'])
        
        logging.info(f'Long format created: {len(long_data)} player-match records')
        
        return long_data

    def convert_to_wide_format(self, long_data):
        logging.info('\tConverting back to wide format...')
        player_columns = [col for col in long_data.columns if col not in config.NEUTRAL_COLUMNS and col not in ['id', 'country_code', 'outcome']]
        winners = long_data[long_data['outcome'] == 1]
        losers = long_data[long_data['outcome'] == 0]

        # We generate 2 copies of each match - with A and B swapped, to prevent 
        v1_winners = winners.rename(columns={col: 'A_' + col for col in player_columns})
        v1_losers = losers.rename(columns={col: 'B_' + col for col in player_columns})

        wide_v1 = pd.merge(
            v1_winners[config.NEUTRAL_COLUMNS + ['A_' + col for col in player_columns]],
            v1_losers[['unique_match_id'] + ['B_' + col for col in player_columns]],
            on='unique_match_id'
        )
        wide_v1['target'] = 1
        
        v2_losers = losers.rename(columns={col: 'A_' + col for col in player_columns})
        v2_winners = winners.rename(columns={col: 'B_' + col for col in player_columns})
        
        wide_v2 = pd.merge(
            v2_losers[config.NEUTRAL_COLUMNS + ['A_' + col for col in player_columns]],
            v2_winners[['unique_match_id'] + ['B_' + col for col in player_columns]],
            on='unique_match_id'
        )
        wide_v2['target'] = 0

        logging.info(f'\tWide format created: {len(wide_v1) * 2} match records')

        return pd.concat([wide_v1, wide_v2], axis=0).reset_index(drop=False)

    def create_differential_features(self, wide_data):
        logging.info('\tCreating differential features...')

        for column in config.DIFFERENTIAL_COLUMNS:
            column_A = f'A_{column}'
            column_B = f'B_{column}'

            if column_A in wide_data.columns and column_B in wide_data.columns:
                wide_data[f'{column}_diff'] = wide_data[column_A] - wide_data[column_B]
                wide_data = wide_data.drop([column_A, column_B], axis=1)

            else:
                logging.warning(f'Could not create differential for {column}: missing {column_A} or {column_B}')

        return wide_data


    def create_match_statistics(self, long_data):
        logging.info('\tGenerating serve/double fault/breakpoint statistics...')
        
        long_data['serve_win_ratio'] = (
            long_data['first_serve_points_won'] + long_data['second_serve_points_won']
        ) / long_data['service_points_played']
        
        long_data['first_serve_ratio'] = (
            long_data['first_serves_in']
        ) / long_data['service_points_played']
        
        long_data['first_serve_win_ratio'] = (
            long_data['first_serve_points_won']
        ) / long_data['first_serves_in']
        
        second_serve_points = (
            long_data['service_points_played'] - 
            long_data['first_serves_in'] - 
            long_data['double_faults']
        )

        long_data['second_serve_win_ratio'] = (
            long_data['second_serve_points_won']
        ) / second_serve_points
        
        long_data['ace_rate'] = (
            long_data['aces']
        ) / long_data['service_points_played']
        
        long_data['double_fault_rate'] = (
            long_data['double_faults']
        ) / long_data['service_points_played']
    
        long_data['bp_save_rate'] = (
            long_data['break_points_saved']
        ) / long_data['break_points_faced']
        
        return long_data

    def create_rolling_features(self, long_data):
        logging.info('\tGenerating rolling features...')

        for stat, params in config.ROLLING_CONFIG.items():
            if stat not in long_data.columns:
                logging.warning(f"'{stat}' not found, ignoring!")
                continue
            
            long_data[f'rolling_{stat}'] = (
                long_data.groupby('id')[stat]
                .transform(
                    lambda x : x.shift(1).rolling(
                        params['window'], 
                        min_periods=params['min_periods']
                    ).mean()
                )
            )

        long_data['rolling_surface_win_ratio'] = (
            long_data.groupby(['id', 'surface'])['outcome']
            .transform(
                lambda x : x.shift(1).rolling(
                    config.SURFACE_WIN_RATIO_CONFIG['window'],
                    min_periods=config.SURFACE_WIN_RATIO_CONFIG['min_periods']
                ).mean()
            )
        )
        
        return long_data
    
    def create_momentum_features(self, long_data):
        logging.info('\tGenerating momentum features...')
        
        long_data['recent_form_ema'] = (
            long_data.groupby('id')['outcome']
            .transform(
                lambda x : x.shift(1).ewm(
                    span=config.RECENT_FORM_EMA_SPAN,
                    min_periods=config.RECENT_FORM_EMA_MIN_PERIODS
                ).mean()
            )
        )
        
        long_data['win_streak'] = (
            long_data.groupby('id')['outcome']
            .transform(
                lambda x : x.shift(1).rolling(
                    config.WIN_STREAK_WINDOW,
                    min_periods=1
                ).sum()
            )
        )
                
        return long_data
    
    def create_time_specific_features(self, long_data):
        logging.info('\tGenerating time-specific features...')
        
        long_data['days_since_last_match'] = (
            long_data.groupby('id')['tournament_date']
            .diff()
            .dt.days
            .fillna(config.DEFAULT_DAYS_SINCE_LAST_MATCH)
        )
        
        long_data = self._add_rank_change_feature(long_data)
        
        return long_data
    
    def _add_rank_change_feature(self, long_data: pd.DataFrame) -> pd.DataFrame:
        long_data_sorted = long_data.sort_values(['tournament_date', 'id']).copy()
        
        lookup = long_data_sorted[['id', 'tournament_date', 'rank']].copy()
        lookup['tournament_date'] = lookup['tournament_date'] + pd.Timedelta(days=config.RANK_LOOKBACK_DAYS)
        lookup = lookup.rename(columns={'rank': 'rank_30d_ago'})
        
        result = pd.merge_asof(
            long_data_sorted,
            lookup[['id', 'tournament_date', 'rank_30d_ago']],
            on='tournament_date',
            by='id',
            direction='backward'
        )
        result['rank_change_30d'] = result['rank_30d_ago'] - result['rank']
        result = result.drop('rank_30d_ago', axis=1)
        
        return result

    def create_categorical_features(self, long_data):
        logging.info('\tGenerating categorical features')
        
        # Binary indicator for seeded players
        long_data['has_seed'] = long_data['seed'].notna().astype(int)
        long_data = long_data.drop('seed', axis=1)
                
        return long_data
    
    def engineer_features(self, matches):      
        matches = self.create_unique_match_id(matches)
        matches = self.create_opposite_bp_stats(matches)
        long_data = self.convert_to_long_format(matches)
        
        # Create features
        long_data = self.create_match_statistics(long_data)
        long_data = self.create_rolling_features(long_data)
        long_data = self.create_momentum_features(long_data)
        long_data = self.create_time_specific_features(long_data)
        long_data = self.create_categorical_features(long_data)
        
        return long_data

    def fill_missing_values(self, long_data):
        numeric_columns = config.NUMERIC_NAN_COLUMNS

        logging.info('\tFilling missing values...')
        
        existing_columns = [col for col in numeric_columns if col in long_data.columns]
        missing_columns = set(numeric_columns) - set(existing_columns)
        if missing_columns:
            logging.info(f'\tColumns not found: {missing_columns}')
        
        initial_nan_count = long_data[existing_columns].isna().sum().sum()
        logging.info(f'\tInitial NaN count: {initial_nan_count:,}')
        
        for col in existing_columns:
            if long_data[col].isna().any():
                long_data[col] = long_data.groupby('id')[col].transform(
                    lambda x: x.fillna(x.mean())
                )
                
                global_mean = long_data[col].mean()
                long_data[col] = long_data[col].fillna(global_mean)
        
        final_nan_count = long_data[existing_columns].isna().sum().sum()
        logging.info(
            f'NaN reduction: {initial_nan_count} -> {final_nan_count}'
        )
        
        return long_data