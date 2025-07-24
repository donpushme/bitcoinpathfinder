import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timezone

class DataProcessor:
    """Process Bitcoin price data for AI model training and inference"""
    
    def __init__(self):
        self.required_fields = ['timestamp', 'open', 'high', 'low', 'close']  # CSV format fields
    
    def validate_csv_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the CSV data has the required format:
        timestamp,open,high,low,close,volume
        """
        if not isinstance(df, pd.DataFrame):
            return False
        
        # Check if all required fields are present
        for field in self.required_fields:
            if field not in df.columns:
                return False
        
        # Check if we have data
        if len(df) == 0:
            return False
        
        return True
    
    def process_csv_data(self, csv_data: str) -> pd.DataFrame:
        """Process CSV data format to pandas DataFrame"""
        # Read CSV data
        from io import StringIO
        df = pd.read_csv(StringIO(csv_data))
        
        if not self.validate_csv_data(df):
            raise ValueError("Invalid CSV data format")
        
        # Convert timestamp to datetime and extract timestamp
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['datetime'].astype('int64') // 10**9  # Convert to Unix timestamp
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Basic validation
        df = self._validate_price_data(df)
        
        return df
    
    def _validate_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean price data"""
        # Remove rows with invalid prices
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
        
        # Check OHLC relationships
        df = df[
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) & 
            (df['high'] >= df['low']) &
            (df['low'] <= df['open']) & 
            (df['low'] <= df['close'])
        ]
        
        # Remove extreme outliers (more than 50% change in one period)
        df['price_change'] = df['close'].pct_change().abs()
        df = df[df['price_change'] <= 0.5]
        df = df.drop('price_change', axis=1)
        
        # Fill any remaining NaN values
        df = df.ffill().bfill()
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for feature engineering"""
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        
        # Moving averages
        for window in [5, 10, 20]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'price_vs_sma_{window}'] = df['close'] / df[f'sma_{window}'] - 1
        
        # Volatility measures
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'realized_vol_{window}'] = np.sqrt(
                df['log_returns'].rolling(window=window).apply(lambda x: np.sum(x**2))
            )
        
        # Volume-like measures (using price range as proxy)
        df['volume_proxy'] = df['price_range'] * df['close']
        for window in [5, 10]:
            df[f'avg_volume_{window}'] = df['volume_proxy'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume_proxy'] / df[f'avg_volume_{window}']
        
        # Higher moments
        for window in [10, 20]:
            returns_window = df['returns'].rolling(window=window)
            df[f'skewness_{window}'] = returns_window.skew()
            df[f'kurtosis_{window}'] = returns_window.kurt()
        
        # Trend indicators
        df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['price_momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        return df
    
    def process_training_data(self, csv_data: str, sequence_length: int = 30) -> List[Dict]:
        """
        Process raw CSV training data into format suitable for AI model training.
        
        Args:
            csv_data: Raw CSV data string
            sequence_length: Length of sequences for model training
            
        Returns:
            List of processed data dictionaries
        """
        # Convert to DataFrame
        df = self.process_csv_data(csv_data)
        print(f"Converted {len(df)} data points to DataFrame")
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        print("Calculated technical indicators")
        
        # Remove rows with NaN values
        df = df.dropna()
        print(f"After cleaning: {len(df)} data points")
        
        if len(df) < sequence_length:
            raise ValueError(f"Insufficient data after cleaning. Need at least {sequence_length}, got {len(df)}")
        
        # Convert back to list of dictionaries for model training
        processed_data = []
        for _, row in df.iterrows():
            processed_data.append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'returns': row.get('returns', 0),
                'volatility': row.get('volatility_5', 0),
                'skewness': row.get('skewness_10', 0),
                'kurtosis': row.get('kurtosis_10', 0)
            })
        
        return processed_data
    
    def create_realtime_features(self, recent_data: List[Dict]) -> List[Dict]:
        """
        Create features for real-time prediction from recent price data.
        
        Args:
            recent_data: List of recent price dictionaries
            
        Returns:
            List of processed data with features
        """
        if len(recent_data) < 20:  # Need minimum data for indicators
            # If insufficient data, return basic format
            return [
                {
                    'timestamp': item['timestamp'],
                    'open': item['open'],
                    'high': item['high'],
                    'low': item['low'],
                    'close': item['close']
                }
                for item in recent_data
            ]
        
        # Convert to DataFrame
        df = pd.DataFrame(recent_data)
        
        # Calculate basic technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Fill NaN values for recent data
        df = df.ffill().fillna(0)
        
        # Convert back to list format
        processed_data = []
        for _, row in df.iterrows():
            processed_data.append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'returns': row.get('returns', 0),
                'volatility': row.get('volatility_5', 0),
                'skewness': row.get('skewness_10', 0),
                'kurtosis': row.get('kurtosis_10', 0)
            })
        
        return processed_data
    
    def get_data_statistics(self, csv_data: str) -> Dict[str, float]:
        """Get basic statistics about the data"""
        df = self.process_csv_data(csv_data)
        df = self.calculate_technical_indicators(df)
        
        stats = {
            'total_points': len(df),
            'time_span_days': (df['timestamp'].max() - df['timestamp'].min()) / 86400,
            'mean_price': df['close'].mean(),
            'price_std': df['close'].std(),
            'mean_returns': df['returns'].mean(),
            'returns_volatility': df['returns'].std(),
            'min_price': df['close'].min(),
            'max_price': df['close'].max(),
            'skewness': df['returns'].skew(),
            'kurtosis': df['returns'].kurt()
        }
        
        return stats
