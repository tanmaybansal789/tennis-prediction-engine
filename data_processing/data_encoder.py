import pandas as pd
import numpy as np
import logging

import config

class DataEncoder:
    """
    Encodes categorical variables to numeric values
    """
    
    def encode_entry_type(self, df):
        logging.info('\tEncoding entry type')
        df['entry_type'] = df['entry_type'].map(config.ENTRY_TYPE_MAP)
        return df
    
    def encode_hand(self, df):
        logging.info('\tEncoding hand')
        df['hand'] = (df['hand'] == 'R').astype(int)
        return df
    
    def encode_surface(self, df):
        logging.info('\tEncoding surface')
        df['surface'] = df['surface'].map(config.SURFACE_MAP)
        return df
    
    def encode_tournament_level(self, df):
        logging.info('\tEncoding tournament level')
        df['tournament_level'] = df['tournament_level'].map(config.TOURNAMENT_LEVEL_MAP)
        return df
    
    def encode_all(self, df):
        df = self.encode_entry_type(df)
        df = self.encode_hand(df)
        df = self.encode_surface(df)
        df = self.encode_tournament_level(df)
        return df