import os

import numpy as np
import pandas as pd

import config
import logging 
from feature_engineer import FeatureEngineer
from data_encoder import DataEncoder

#####################
# UTILITY FUNCTIONS #
#####################

def calculate_elo_features(matches, initial_elo, k_factor):
    # Initial Elo

    current_elos = {}

    # This refers to the previous elo ranking (before the match)
    winner_elos = []
    loser_elos = []

    for _, row in matches.iterrows():
        winner_id, loser_id = row['winner_id'], row['loser_id']

        winner_elo = current_elos.get(winner_id, initial_elo)
        loser_elo = current_elos.get(loser_id, initial_elo)
        winner_elos.append(winner_elo)
        loser_elos.append(loser_elo)

        # A sigmoidal (S-shaped) win probability prediction (a form of simplistic logistic regression)
        win_prob = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))

        # The k-factor (the maximum rating change posssible) by the win probability
        # I.e. if the winner was predicted as unlikely to do win, their rating change is much more significant
        rating_change = k_factor * (1 - win_prob)

        # Update Elos
        current_elos[winner_id] = winner_elo + rating_change
        current_elos[loser_id] = loser_elo - rating_change

    matches['winner_elo'] = winner_elos
    matches['loser_elo'] = loser_elos


#####################
# MAIN PIPELINE     #
#####################


class TennisPipeline:
    def __init__(self, 
                 input_dir=config.INPUT_DIRECTORY, 
                 output_dir=config.OUTPUT_DIRECTORY, 
                 years=config.YEARS):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.years = years

        logging.getLogger().setLevel(logging.INFO)
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s - %(message)s"
        )

        logging.info('=' * 80)
        logging.info('Tennis Data Pipeline')
        logging.info('=' * 80)

        logging.info(f'Input data directory: {os.path.relpath(input_dir)}')
        logging.info(f'Output data directory: {os.path.relpath(output_dir)}')
        logging.info(f'Years: {self.years.start}-{self.years.stop-1}')

    def execute(self):
        logging.info('Starting pipeline execution')

        logging.info('=' * 80)
        logging.info('Loading raw match data...')

        matches = pd.concat([pd.read_csv(f'{self.input_dir}/atp_matches_{year}.csv') for year in self.years])
        matches.columns = config.COLUMN_NAMES
        matches['tournament_date'] = pd.to_datetime(matches['tournament_date'], format='%Y%m%d')

        logging.info(f'Loaded DataFrame of shape {len(matches)} by {len(matches.columns)}')
        logging.info(f'Memory usage: {matches.memory_usage(deep=True).sum() / (1024 * 1024):.2f}MB')
        logging.info(f'NaN count: {matches.isna().sum().sum()}')

        logging.info('=' * 80)
        logging.info(f'Calculating Elo ratings (Initial={config.INITIAL_ELO}, K={config.K_FACTOR})...')

        calculate_elo_features(matches, config.K_FACTOR, config.INITIAL_ELO)

        logging.info('=' * 80)
        logging.info('Feature engineering...')

        engineer = FeatureEngineer()
        long_data = engineer.engineer_features(matches)

        logging.info('=' * 80)
        logging.info('Performing label encoding...')

        data_encoder = DataEncoder()
        long_data = data_encoder.encode_all(long_data)

        logging.info('=' * 80)
        logging.info('Filling NaNs...')

        long_data = engineer.fill_missing_values(long_data)

        logging.info('=' * 80)
        logging.info('Reverting to wide format...')

        wide_data = engineer.convert_to_wide_format(long_data)


        logging.info('=' * 80)
        logging.info('Final feature processing...')

        wide_data = engineer.create_differential_features(wide_data)

        logging.info('=' * 80)
        logging.info('Final stage - feature extraction...')

        # Set exclusion operator to extract those that exist in feature columns, not in wide columns
        missing_features = set(config.FEATURE_COLUMNS) - set(wide_data.columns)
        if missing_features:
            raise ValueError(f'Features are missing: {missing_features}')

        x = wide_data[config.FEATURE_COLUMNS].copy()
        y = wide_data['target'].copy()

        logging.info('Done!')
        logging.info('Features preview:')
        logging.info(x.tail())
        logging.info('Labels preview:')
        logging.info(y.tail())

        return x, y
    
    def export(self, x, y):
        logging.info("-" * 80)
        logging.info("Saving final outputs...")
        logging.info("-" * 80)
        
        # Save as CSV
        features_path = f'{self.output_dir}/tennis_model_features.csv'
        labels_path = f'{self.output_dir}/tennis_model_labels.csv'
        
        x.to_csv(features_path, index=False)
        y.to_csv(labels_path, index=False)
        
        logging.info(f"Features saved: {features_path} ({os.path.getsize(features_path) / (1024 * 1024):.2f}MB)")
        logging.info(f"Features saved: {labels_path} ({os.path.getsize(labels_path) / (1024 * 1024):.2f}MB)")