import pandas as pd
import logging
from pathlib import Path
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, raw_path: str, processed_path: str):
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_ieso_demand(self, filename: str) -> pd.DataFrame:
        """Loads IESO Load Data"""
        file_path = self.raw_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot find {file_path}")

        logging.info(f"Loading LOAD data from {file_path}...")
        # Skip 3 rows of metadata
        df = pd.read_csv(file_path, header=3) 

        # Clean columns
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        
        # Fix time
        if 'hour' in df.columns:
            df['hour'] = df['hour'] - 1
        
        try:
            df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
        except KeyError:
            logging.error(f"Column error. Found: {df.columns}")
            raise

        df = df.sort_values('timestamp').set_index('timestamp')
        
        # Select 'ontario_demand'
        target_col = 'ontario_demand' if 'ontario_demand' in df.columns else 'market_demand'
        df = df.rename(columns={target_col: 'load_mw'})
        
        return df[['load_mw']]

    def load_weather(self, pattern: str = "en_climate_hourly_*.csv") -> pd.DataFrame:
        """
        Loads ALL weather CSVs matching the pattern in data/raw/
        """
        files = list(self.raw_path.glob(pattern))
        if not files:
            logging.warning(f"No weather files found matching {pattern}")
            return pd.DataFrame()

        logging.info(f"Found {len(files)} weather files. Loading...")
        
        df_list = []
        for f in files:
            try:
                # Standard Env Canada format
                temp = pd.read_csv(f)
                df_list.append(temp)
            except Exception as e:
                logging.error(f"Failed to read {f}: {e}")

        if not df_list:
            return pd.DataFrame()

        # Combine all months
        full_weather = pd.concat(df_list, ignore_index=True)

        # Clean Up
        # Parse 'Date/Time (LST)' to timestamp
        full_weather['timestamp'] = pd.to_datetime(full_weather['Date/Time (LST)'])
        full_weather = full_weather.set_index('timestamp').sort_index()

        # Select meaningful columns
        cols_to_keep = {
            'Temp (°C)': 'temp_c',
            'Rel Hum (%)': 'humidity',
            'Wind Spd (km/h)': 'wind_speed_kmh',
            'Weather': 'weather'  # <--- CRITICAL: Keep the text column for cloudiness
        }
        
        # Rename
        clean_weather = full_weather.rename(columns=cols_to_keep)
        
        # Only keep columns that exist
        valid_cols = [c for c in clean_weather.columns if c in cols_to_keep.values()]
        clean_weather = clean_weather[valid_cols]

        # Deduplicate (sometimes weather files overlap)
        clean_weather = clean_weather[~clean_weather.index.duplicated(keep='first')]

        logging.info(f"Weather data loaded: {len(clean_weather)} rows.")
        return clean_weather

    def merge_data(self, df_load: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
        """Inner Join Load and Weather (Only keeps times where we have BOTH)"""
        if df_weather.empty:
            logging.warning("Weather data is empty. Returning Load only.")
            return df_load

        logging.info("Merging Load and Weather data...")
        # inner join = discard rows where we don't have weather
        merged = df_load.join(df_weather, how='inner')
        
        logging.info(f"Merged Dataset Size: {len(merged)} rows.")
        return merged

    def save_processed(self, df: pd.DataFrame, filename: str):
        out_path = self.processed_path / filename
        df.to_csv(out_path)
        logging.info(f"Saved processed data to {out_path}")