"""
FactoryGuard AI - Feature Engineering Pipeline
Production-grade time-series feature extraction with joblib serialization
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Production-ready feature engineering for time-series sensor data
    - Rolling statistics (1h, 6h, 12h windows)
    - Lag features
    - Rate of change
    - Cross-sensor interactions
    """

    def __init__(self, windows=[1, 6, 12], lags=[1, 6, 12]):
        self.windows = windows
        self.lags = lags

    def fit(self, X, y=None):
        """Fit method (required for sklearn pipeline)"""
        return self

    def transform(self, X):
        """Transform raw sensor data into features"""
        df = X.copy()

        # Sort by robot and timestamp
        df = df.sort_values(['robot_id', 'timestamp'])

        features = []

        # Process each robot separately (time-series grouped by robot)
        for robot_id, group in df.groupby('robot_id'):
            robot_features = group[['timestamp', 'robot_id', 'vibration',
                                    'temperature', 'pressure']].copy()

            # Handle missing values (forward fill with limit)
            robot_features['vibration'] = robot_features['vibration'].fillna(method='ffill', limit=3)
            robot_features['temperature'] = robot_features['temperature'].fillna(method='ffill', limit=3)
            robot_features['pressure'] = robot_features['pressure'].fillna(method='ffill', limit=3)

            # Fill remaining with median
            robot_features['vibration'] = robot_features['vibration'].fillna(
                robot_features['vibration'].median())
            robot_features['temperature'] = robot_features['temperature'].fillna(
                robot_features['temperature'].median())
            robot_features['pressure'] = robot_features['pressure'].fillna(
                robot_features['pressure'].median())

            # Rolling statistics for each window
            for window in self.windows:
                for col in ['vibration', 'temperature', 'pressure']:
                    # Rolling mean
                    robot_features[f'{col}_mean_{window}h'] = \
                        robot_features[col].rolling(window=window, min_periods=1).mean()

                    # Rolling std dev
                    robot_features[f'{col}_std_{window}h'] = \
                        robot_features[col].rolling(window=window, min_periods=1).std().fillna(0)

                    # Rolling min/max
                    robot_features[f'{col}_min_{window}h'] = \
                        robot_features[col].rolling(window=window, min_periods=1).min()
                    robot_features[f'{col}_max_{window}h'] = \
                        robot_features[col].rolling(window=window, min_periods=1).max()

                    # Exponential moving average
                    robot_features[f'{col}_ema_{window}h'] = \
                        robot_features[col].ewm(span=window, min_periods=1).mean()

            # Lag features
            for lag in self.lags:
                for col in ['vibration', 'temperature', 'pressure']:
                    robot_features[f'{col}_lag_{lag}h'] = \
                        robot_features[col].shift(lag).fillna(method='bfill')

            # Rate of change (first derivative)
            for col in ['vibration', 'temperature', 'pressure']:
                robot_features[f'{col}_rate_change'] = \
                    robot_features[col].diff().fillna(0)

                # Second derivative (acceleration)
                robot_features[f'{col}_acceleration'] = \
                    robot_features[f'{col}_rate_change'].diff().fillna(0)

            # Cross-sensor interactions
            robot_features['temp_vibration_interaction'] = \
                robot_features['temperature'] * robot_features['vibration']
            robot_features['pressure_vibration_interaction'] = \
                robot_features['pressure'] * robot_features['vibration']

            # Sensor reading ranges
            robot_features['vibration_range_12h'] = \
                robot_features['vibration_max_12h'] - robot_features['vibration_min_12h']
            robot_features['temperature_range_12h'] = \
                robot_features['temperature_max_12h'] - robot_features['temperature_min_12h']

            features.append(robot_features)

        # Combine all robots
        result = pd.concat(features, ignore_index=True)

        return result


# Load raw data
print("Loading raw sensor data...")
df = pd.read_csv('sensor_data_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"✓ Loaded {len(df):,} records")
print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Engineer features
print("\nEngineering time-series features...")
print("- Rolling statistics (1h, 6h, 12h windows)")
print("- Lag features (t-1, t-6, t-12)")
print("- Rate of change & acceleration")
print("- Cross-sensor interactions")

feature_engineer = TimeSeriesFeatureEngineer(windows=[1, 6, 12], lags=[1, 6, 12])
df_features = feature_engineer.fit_transform(df)

# Add target variable back
if 'failure_within_24h' in df.columns:
    df_features['failure_within_24h'] = df['failure_within_24h'].values

# Save engineered features
df_features.to_csv('sensor_data_features.csv', index=False)

# Save feature engineering pipeline
joblib.dump(feature_engineer, 'models/feature_engineer.joblib', compress=3)

print(f"\n✓ Features engineered: {df_features.shape[1]} total columns")
print(f"✓ Original features: 3 (vibration, temperature, pressure)")
print(f"✓ Engineered features: {df_features.shape[1] - 5}")
print(f"\n✓ Saved to: sensor_data_features.csv")
print(f"✓ Pipeline saved to: models/feature_engineer.joblib")

# Display feature summary
print("\n" + "=" * 60)
print("FEATURE SUMMARY:")
print("=" * 60)
feature_cols = [col for col in df_features.columns
                if col not in ['timestamp', 'robot_id', 'failure_within_24h',
                               'vibration', 'temperature', 'pressure']]
print(f"Total engineered features: {len(feature_cols)}")
print("\nFeature categories:")
print(
    f"  - Rolling statistics: {sum('mean' in c or 'std' in c or 'min' in c or 'max' in c or 'ema' in c for c in feature_cols)}")
print(f"  - Lag features: {sum('lag' in c for c in feature_cols)}")
print(f"  - Rate of change: {sum('rate_change' in c or 'acceleration' in c for c in feature_cols)}")
print(f"  - Interactions: {sum('interaction' in c or 'range' in c for c in feature_cols)}")

print("\n" + "=" * 60)
print("SAMPLE ENGINEERED FEATURES:")
print("=" * 60)
display_cols = ['robot_id', 'vibration', 'vibration_mean_12h', 'vibration_std_12h',
                'temperature', 'temperature_ema_6h', 'temp_vibration_interaction',
                'failure_within_24h']
print(df_features[display_cols].head(10))

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION:")
print("=" * 60)
print(df_features['failure_within_24h'].value_counts())
print(f"Failure rate: {df_features['failure_within_24h'].mean():.2%}")