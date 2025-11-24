import pandas as pd
import numpy as np
import logging

def add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for the forecasting model.
    """
    df = df.copy()
    logging.info("Engineering features (Time + Physics + Text)...")

    # 1. Time Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # 2. Lag Features
    df['lag_24h'] = df['load_mw'].shift(24)
    df['lag_168h'] = df['load_mw'].shift(168)

    # 3. Physics
    if 'temp_c' in df.columns:
        df['temp_squared'] = df['temp_c'] ** 2
        df['temp_x_hour'] = df['temp_c'] * df['hour']
    
    # 4. Cloudiness / Text Analysis
    if 'weather' in df.columns:
        # Normalize text
        w_text = df['weather'].fillna('clear').str.lower()
        
        df['is_cloudy'] = w_text.str.contains('cloud|overcast').astype(int)
        df['is_precip'] = w_text.str.contains('rain|snow|drizzle|storm|flurries').astype(int)
        df['is_clear'] = w_text.str.contains('clear|sunny').astype(int)
    else:
        # Fallback
        df['is_cloudy'] = 0
        df['is_precip'] = 0
        df['is_clear'] = 1

    # Fill missing numeric weather
    df = df.ffill()

    # Cleanup NaNs from lags
    df = df.dropna()
    
    return df