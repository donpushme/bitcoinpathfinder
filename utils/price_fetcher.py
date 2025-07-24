import requests
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict
import time
import json

class BitcoinPriceFetcher:
    """Fetch Bitcoin price data from external APIs"""
    
    def __init__(self):
        self.base_urls = {
            'pyth': "https://benchmarks.pyth.network/v1/shims/tradingview/history",
            'binance': "https://api.binance.com/api/v3/klines",
            'coinbase': "https://api.pro.coinbase.com/products/BTC-USD/candles"
        }
        self.request_timeout = 10
        self.max_retries = 3
    
    def fetch_latest_bitcoin_price(self) -> Tuple[Optional[float], Optional[int]]:
        """
        Fetch the latest Bitcoin price using the provided function logic.
        
        Returns:
            Tuple of (price, timestamp) or (None, None) if failed
        """
        try:
            # Using Pyth Network API (free, no API key required)
            # Get the current time and round down to the nearest 5-minute interval
            current_dt = datetime.now(timezone.utc)            
            # Round down to the nearest 1-2 minute interval that has passed
            # Get the most recent price data available (1-2 minutes ago)
            end_dt = current_dt - timedelta(minutes=2)
            end_dt = end_dt.replace(second=0, microsecond=0)  # Round to minute
            start_dt = end_dt - timedelta(days=1)
            
            trading_pair = 'Crypto.BTC/USD'
            url = self.base_urls['pyth']
            
            response = requests.get(url, params={
                "symbol": trading_pair,
                "resolution": '5',
                "from": int(start_dt.timestamp()),
                "to": int(end_dt.timestamp())
            }, timeout=self.request_timeout)
            
            response.raise_for_status()
            data = response.json()
            
            if 'c' in data and 't' in data and len(data['c']) > 0:
                price = float(data['c'][-1])
                timestamp = int(data['t'][-1])
                return price, timestamp
            else:
                return None, None
            
        except Exception as e:
            print(f"Error fetching price from Pyth: {e}")
            # Try fallback to Binance
            return self._fetch_from_binance_fallback()
    
    def _fetch_from_binance_fallback(self) -> Tuple[Optional[float], Optional[int]]:
        """Fallback method using Binance API"""
        try:
            url = self.base_urls['binance']
            params = {
                'symbol': 'BTCUSDT',
                'interval': '5m',
                'limit': 1
            }
            
            response = requests.get(url, params=params, timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                # Binance returns: [timestamp, open, high, low, close, volume, ...]
                latest = data[0]
                price = float(latest[4])  # Close price
                timestamp = int(latest[0]) // 1000  # Convert milliseconds to seconds
                return price, timestamp
            
            return None, None
            
        except Exception as e:
            print(f"Error fetching price from Binance: {e}")
            return None, None
    
    def get_recent_data(self, days: int = 7, interval: str = '5m') -> List[Dict]:
        """
        Get recent historical data for model training/prediction.
        
        Args:
            days: Number of days of historical data
            interval: Data interval ('5m', '1h', '1d')
            
        Returns:
            List of price data dictionaries
        """
        try:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=days)
            
            # Try Pyth first
            data = self._fetch_historical_pyth(start_dt, end_dt, interval)
            if data:
                return data
            
            # Fallback to Binance
            data = self._fetch_historical_binance(start_dt, end_dt, interval)
            if data:
                return data
            
            return []
            
        except Exception as e:
            print(f"Error fetching recent data: {e}")
            return []
    
    def _fetch_historical_pyth(self, start_dt: datetime, end_dt: datetime, 
                              interval: str = '5m') -> List[Dict]:
        """Fetch historical data from Pyth Network"""
        try:
            # Map interval to Pyth resolution
            resolution_map = {
                '1m': '1',
                '5m': '5',
                '15m': '15',
                '1h': '60',
                '4h': '240',
                '1d': 'D'
            }
            resolution = resolution_map.get(interval, '5')
            
            trading_pair = 'Crypto.BTC/USD'
            url = self.base_urls['pyth']
            
            response = requests.get(url, params={
                "symbol": trading_pair,
                "resolution": resolution,
                "from": int(start_dt.timestamp()),
                "to": int(end_dt.timestamp())
            }, timeout=self.request_timeout)
            
            response.raise_for_status()
            data = response.json()
            
            if all(key in data for key in ['t', 'o', 'h', 'l', 'c']):
                # Convert to list of dictionaries
                result = []
                for i in range(len(data['t'])):
                    result.append({
                        'timestamp': data['t'][i],
                        'open': data['o'][i],
                        'high': data['h'][i],
                        'low': data['l'][i],
                        'close': data['c'][i]
                    })
                return result
            
            return []
            
        except Exception as e:
            print(f"Error fetching historical data from Pyth: {e}")
            return []
    
    def _fetch_historical_binance(self, start_dt: datetime, end_dt: datetime, 
                                 interval: str = '5m') -> List[Dict]:
        """Fetch historical data from Binance"""
        try:
            # Map interval to Binance format
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            binance_interval = interval_map.get(interval, '5m')
            
            url = self.base_urls['binance']
            params = {
                'symbol': 'BTCUSDT',
                'interval': binance_interval,
                'startTime': int(start_dt.timestamp() * 1000),
                'endTime': int(end_dt.timestamp() * 1000),
                'limit': 1000
            }
            
            response = requests.get(url, params=params, timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
            
            result = []
            for item in data:
                result.append({
                    'timestamp': int(item[0]) // 1000,  # Convert milliseconds to seconds
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4])
                })
            
            return result
            
        except Exception as e:
            print(f"Error fetching historical data from Binance: {e}")
            return []
    
    def validate_price_data(self, data: List[Dict]) -> bool:
        """Validate fetched price data"""
        if not data or len(data) == 0:
            return False
        
        required_fields = ['timestamp', 'open', 'high', 'low', 'close']
        
        for item in data:
            # Check required fields
            if not all(field in item for field in required_fields):
                return False
            
            # Check data types and values
            try:
                timestamp = int(item['timestamp'])
                open_price = float(item['open'])
                high_price = float(item['high'])
                low_price = float(item['low'])
                close_price = float(item['close'])
                
                # Basic validation
                if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                    return False
                
                if high_price < max(open_price, close_price, low_price):
                    return False
                
                if low_price > min(open_price, close_price, high_price):
                    return False
                    
            except (ValueError, TypeError):
                return False
        
        return True
    
    def get_price_statistics(self, data: List[Dict]) -> Dict[str, float]:
        """Get basic statistics from price data"""
        if not data:
            return {}
        
        closes = [float(item['close']) for item in data]
        highs = [float(item['high']) for item in data]
        lows = [float(item['low']) for item in data]
        
        # Calculate returns
        returns = []
        for i in range(1, len(closes)):
            returns.append((closes[i] - closes[i-1]) / closes[i-1])
        
        stats = {
            'count': len(data),
            'mean_price': np.mean(closes),
            'min_price': np.min(lows),
            'max_price': np.max(highs),
            'price_std': np.std(closes),
            'mean_return': np.mean(returns) if returns else 0,
            'return_volatility': np.std(returns) if returns else 0,
            'time_span_hours': (data[-1]['timestamp'] - data[0]['timestamp']) / 3600
        }
        
        return stats
