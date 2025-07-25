#!/usr/bin/env python3
"""
Real-time Bitcoin Monitoring Script

This script continuously fetches Bitcoin prices and creates predictions.
It's designed to work with the existing Bitcoin Monte Carlo AI Predictor system.

Usage:
    python realtime_monitor.py [--interval MINUTES] [--output_dir DIR]
"""

import argparse
import time
import datetime
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import sys
from typing import List, Dict, Optional

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.price_fetcher import BitcoinPriceFetcher
from models.monte_carlo import simulate_crypto_price_paths

class RealTimeMonitor:
    def __init__(self, 
                 fetch_interval: int = 5,  # minutes
                 output_dir: str = "realtime_data"):
        """
        Initialize the real-time monitor
        
        Args:
            fetch_interval: Minutes between price fetches
            output_dir: Directory to save outputs
        """
        self.fetch_interval = fetch_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.price_fetcher = BitcoinPriceFetcher()
        
        # Data storage
        self.price_history = []
        self.prediction_history = []
        
        # Setup logging
        self.setup_logging()
        
        # Load existing data if available
        self.load_existing_data()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"monitor_log_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_existing_data(self):
        """Load existing price history if available"""
        history_file = self.output_dir / "price_history.csv"
        
        if history_file.exists():
            try:
                df = pd.read_csv(history_file)
                self.price_history = df.to_dict('records')
                self.logger.info(f"Loaded {len(self.price_history)} historical price points")
            except Exception as e:
                self.logger.warning(f"Could not load existing data: {e}")
                self.price_history = []
        else:
            self.price_history = []
    
    def save_price_history(self):
        """Save current price history to CSV"""
        if self.price_history:
            df = pd.DataFrame(self.price_history)
            history_file = self.output_dir / "price_history.csv"
            df.to_csv(history_file, index=False)
            
            # Also save as training data format
            training_file = self.output_dir / "bitcoin_training_data.csv"
            df.to_csv(training_file, index=False)
            
            self.logger.info(f"Saved {len(self.price_history)} price points")
    
    def fetch_latest_price(self) -> Optional[Dict]:
        """Fetch the latest Bitcoin price and add to history"""
        try:
            price_data = self.price_fetcher.fetch_latest_bitcoin_price()
            if price_data[0] is None:
                self.logger.warning("Failed to fetch price data")
                return None
                
            price, timestamp = price_data
            dt = datetime.datetime.now()
            
            # Create price record (OHLC format for compatibility)
            price_record = {
                'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'open': price,
                'high': price * 1.001,  # Small variation for OHLC
                'low': price * 0.999,
                'close': price,
                'volume': 0.0
            }
            
            self.price_history.append(price_record)
            
            # Keep only recent data (last 2000 points)
            if len(self.price_history) > 2000:
                self.price_history = self.price_history[-2000:]
            
            self.logger.info(f"Fetched price: ${price:,.2f} at {dt.strftime('%H:%M:%S')}")
            return price_record
            
        except Exception as e:
            self.logger.error(f"Error fetching price: {e}")
            return None
    
    def calculate_volatility(self, lookback_periods: int = 20) -> float:
        """Calculate recent volatility from price history"""
        if len(self.price_history) < lookback_periods:
            return 0.02  # Default 2% daily volatility
        
        try:
            # Get recent prices
            recent_prices = [float(p['close']) for p in self.price_history[-lookback_periods:]]
            
            # Calculate returns
            returns = [np.log(recent_prices[i] / recent_prices[i-1]) 
                      for i in range(1, len(recent_prices))]
            
            # Calculate volatility (annualized from 5-minute intervals)
            vol = np.std(returns) * np.sqrt(288 * 365)  # 288 = 5-min intervals per day
            
            # Convert to daily volatility
            daily_vol = vol / np.sqrt(365)
            
            return max(0.005, min(0.1, daily_vol))  # Clamp between 0.5% and 10%
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility: {e}")
            return 0.02
    
    def make_prediction(self, current_price: float) -> Optional[Dict]:
        """Make prediction using Monte Carlo simulation"""
        try:
            # Calculate dynamic parameters based on recent data
            volatility = self.calculate_volatility()
            
            # Use adaptive AI parameters for the simulation
            ai_params = {
                'daily_sigma': volatility,
                'daily_drift': 0.0001,  # Small positive drift
                'skewness': 0.0,
                'kurtosis': 3.0
            }
            
            # Run Monte Carlo simulation for 24 hours
            simulation_results = simulate_crypto_price_paths(
                current_price=current_price,
                time_increment=1/24,  # Hourly increments
                time_length=1,  # 24 hours
                num_simulations=500,
                asset='BTC',
                ai_params=ai_params
            )
            
            # Calculate prediction statistics
            final_prices = [path[-1] for path in simulation_results]
            
            prediction = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': current_price,
                'predicted_mean': float(np.mean(final_prices)),
                'predicted_median': float(np.median(final_prices)),
                'predicted_std': float(np.std(final_prices)),
                'confidence_95_lower': float(np.percentile(final_prices, 2.5)),
                'confidence_95_upper': float(np.percentile(final_prices, 97.5)),
                'confidence_68_lower': float(np.percentile(final_prices, 16)),
                'confidence_68_upper': float(np.percentile(final_prices, 84)),
                'volatility_estimate': volatility,
                'monte_carlo_params': ai_params,
                'num_simulations': len(final_prices)
            }
            
            self.prediction_history.append(prediction)
            
            # Keep only recent predictions
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            self.logger.info(f"24h Prediction: ${prediction['predicted_mean']:,.2f} "
                           f"(95% CI: ${prediction['confidence_95_lower']:,.2f} - ${prediction['confidence_95_upper']:,.2f})")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return None
    
    def save_prediction(self, prediction: Dict):
        """Save prediction to file"""
        try:
            predictions_file = self.output_dir / "predictions.json"
            
            # Load existing predictions
            if predictions_file.exists():
                with open(predictions_file, 'r') as f:
                    all_predictions = json.load(f)
            else:
                all_predictions = []
            
            all_predictions.append(prediction)
            
            # Keep only recent predictions
            if len(all_predictions) > 500:
                all_predictions = all_predictions[-500:]
            
            # Save back to file
            with open(predictions_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
    
    def save_summary(self):
        """Save current summary to file"""
        try:
            summary_file = self.output_dir / "current_status.json"
            
            if self.price_history and self.prediction_history:
                current_price = self.price_history[-1]['close']
                latest_prediction = self.prediction_history[-1]
                
                summary = {
                    'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'current_price': current_price,
                    'price_history_count': len(self.price_history),
                    'prediction_count': len(self.prediction_history),
                    'latest_prediction': latest_prediction,
                    'volatility_estimate': latest_prediction.get('volatility_estimate', 0.02)
                }
                
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Error saving summary: {e}")
    
    def print_status(self, current_price: float, prediction: Optional[Dict]):
        """Print current status to console"""
        print("\n" + "="*80)
        print(f"REAL-TIME BITCOIN MONITOR - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"Current Price: ${current_price:,.2f}")
        print(f"Data Points Collected: {len(self.price_history)}")
        print(f"Predictions Made: {len(self.prediction_history)}")
        
        if prediction:
            print(f"24h Prediction: ${prediction['predicted_mean']:,.2f}")
            print(f"95% Confidence: ${prediction['confidence_95_lower']:,.2f} - ${prediction['confidence_95_upper']:,.2f}")
            print(f"Current Volatility: {prediction['volatility_estimate']:.4f} ({prediction['volatility_estimate']*100:.2f}%)")
        
        next_fetch = datetime.datetime.now() + datetime.timedelta(minutes=self.fetch_interval)
        print(f"Next Update: {next_fetch.strftime('%H:%M:%S')}")
        print(f"Data saved to: {self.output_dir}")
        print("="*80)
    
    def run(self):
        """Main loop for real-time monitoring"""
        self.logger.info("Starting real-time Bitcoin monitor...")
        print(f"Real-time monitor started. Fetching prices every {self.fetch_interval} minutes.")
        print(f"Data will be saved to: {self.output_dir}")
        print("Press Ctrl+C to stop.\n")
        
        try:
            while True:
                # Fetch latest price
                price_record = self.fetch_latest_price()
                
                if price_record:
                    current_price = price_record['close']
                    
                    # Make prediction
                    prediction = self.make_prediction(current_price)
                    
                    if prediction:
                        self.save_prediction(prediction)
                    
                    # Save all data
                    self.save_price_history()
                    self.save_summary()
                    
                    # Print status
                    self.print_status(current_price, prediction)
                
                # Wait for next fetch
                time.sleep(self.fetch_interval * 60)
                
        except KeyboardInterrupt:
            self.logger.info("Stopping real-time monitor...")
            print("\nStopping monitor. Saving final data...")
            self.save_price_history()
            self.save_summary()
            print(f"All data saved to: {self.output_dir}")
            print("Done! You can now use the collected data for training:")
            print(f"python train_model.py train --csv_file {self.output_dir}/bitcoin_training_data.csv")
        
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.save_price_history()
            self.save_summary()
            raise

def main():
    parser = argparse.ArgumentParser(description='Real-time Bitcoin Price Monitor')
    parser.add_argument('--interval', type=int, default=5, 
                       help='Minutes between price fetches (default: 5)')
    parser.add_argument('--output_dir', type=str, default='realtime_data',
                       help='Directory to save outputs (default: realtime_data)')
    
    args = parser.parse_args()
    
    # Create and run monitor
    monitor = RealTimeMonitor(
        fetch_interval=args.interval,
        output_dir=args.output_dir
    )
    
    monitor.run()

if __name__ == "__main__":
    main()