import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_absolute_error

class GridForecaster:
    def __init__(self):
        # Production-grade hyperparameters
        self.model = xgb.XGBRegressor(
            n_estimators=1000,        # More trees for better accuracy
            learning_rate=0.03,       # Slower learning = better generalization
            max_depth=7,              # Deeper trees to capture complex weather logic
            subsample=0.8,            # Prevent overfitting
            colsample_bytree=0.8,
            objective='reg:squarederror',
            n_jobs=-1,
            early_stopping_rounds=50
        )
        
        self.features = [
            'hour', 'day_of_week', 'is_weekend', 'month',
            'lag_24h', 'lag_168h',
            'temp_c', 'temp_squared', 'temp_x_hour',
            'humidity', 'wind_speed_kmh',
            'is_cloudy', 'is_precip', 'is_clear'  # <-- NEW FEATURES
        ]
        
    def train(self, df: pd.DataFrame):
        """
        Splits data and trains the model.
        """
        # Validation Split: Use the last 2 months (Nov-Dec) for testing
        split_idx = len(df) - (24 * 60) 
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # Ensure all features exist (handle cases where humidity/wind might be missing)
        valid_features = [f for f in self.features if f in df.columns]
        if len(valid_features) < len(self.features):
            logging.warning(f"Some features missing. Using: {valid_features}")

        X_train = train_df[valid_features]
        y_train = train_df['load_mw']
        X_test = test_df[valid_features]
        y_test = test_df['load_mw']

        logging.info(f"Training XGBoost on {len(train_df)} samples...")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100
        )

        # Evaluate
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        # Calculate MAPE (Mean Absolute Percentage Error) - Industry Standard Metric
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        
        logging.info(f"Model Trained. Test Set MAE: {mae:.2f} MW")
        logging.info(f"Test Set MAPE: {mape:.2f}% (Target < 3%)")
        
        return test_df, preds