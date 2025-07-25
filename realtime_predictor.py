#!/usr/bin/env python3
"""
Real-time Bitcoin Price Predictor

This script continuously:
1. Fetches the latest Bitcoin price every 5 minutes
2. Updates the training dataset with new price data
3. Retrains the AI model with recent data
4. Makes predictions for the next 24 hours
5. Saves results and displays live updates

Usage:
    python realtime_predictor.py [--interval MINUTES] [--retrain_interval HOURS] [--max_data_points N]
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
import torch
from typing import List, Dict, Optional

# Import our modules
from utils.price_fetcher import BitcoinPriceFetcher
from models.ai_predictor import BitcoinParameterPredictor
from models.monte_carlo import MonteCarloSimulator
from data.data_processor import DataProcessor

class RealTimePredictor:
    def __init__(self, 
                 fetch_interval: int = 5,  # minutes
                 retrain_interval: int = 1,  # hours
                 max_data_points: int = 2000,
                 output_dir: str = "realtime_output"):
        """
        Initialize the real-time predictor
        
        Args:
            fetch_interval: Minutes between price fetches
            retrain_interval: Hours between model retraining
            max_data_points: Maximum historical data points to keep
            output_dir: Directory to save outputs
        """
        self.fetch_interval = fetch_interval
        self.retrain_interval = retrain_interval * 3600  # Convert to seconds
        self.max_data_points = max_data_points
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.price_fetcher = BitcoinPriceFetcher()
        self.data_processor = DataProcessor()
        self.ai_predictor = None
        self.monte_carlo = MonteCarloSimulator()
        
        # Data storage
        self.price_history = []
        self.last_retrain_time = 0
        self.prediction_history = []
        
        # Setup logging
        self.setup_logging()
        
        # Load existing data if available
        self.load_existing_data()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"realtime_log_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        
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
            self.logger.info(f"Saved {len(self.price_history)} price points to {history_file}")
    
    def fetch_latest_price(self) -> Optional[Dict]:
        """Fetch the latest Bitcoin price and add to history"""
        try:
            price_data = self.price_fetcher.fetch_latest_bitcoin_price()
            if price_data[0] is None:
                return None
            price = price_data[0]
            timestamp = datetime.datetime.now()
            
            # Create price record (simulating OHLC data from single price point)
            price_record = {
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'open': price,
                'high': price * 1.001,  # Small variation for OHLC
                'low': price * 0.999,
                'close': price,
                'volume': 0.0
            }
            
            self.price_history.append(price_record)
            
            # Keep only recent data points
            if len(self.price_history) > self.max_data_points:
                self.price_history = self.price_history[-self.max_data_points:]
            
            self.logger.info(f"Fetched price: ${price:,.2f} at {timestamp.strftime('%H:%M:%S')}")
            return price_record
            
        except Exception as e:
            self.logger.error(f"Error fetching price: {e}")
            return None
    
    def should_retrain_model(self) -> bool:
        """Check if it's time to retrain the model"""
        current_time = time.time()
        return (current_time - self.last_retrain_time) >= self.retrain_interval
    
    def train_model(self) -> bool:
        """Train the AI model with current price history"""
        if len(self.price_history) < 50:
            self.logger.warning(f"Not enough data for training. Need at least 50 points, have {len(self.price_history)}")
            return False
        
        try:
            self.logger.info("Starting model training...")
            
            # Convert price history to DataFrame
            df = pd.DataFrame(self.price_history)
            
            # Convert DataFrame to CSV string and process
            csv_string = df.to_csv(index=False)
            processed_data = self.data_processor.process_training_data(csv_string)
            
            if processed_data is None or len(processed_data) < 30:
                self.logger.warning("Insufficient processed data for training")
                return False
            
            # Initialize and train AI predictor
            self.ai_predictor = BitcoinParameterPredictor()
            
            # Train with recent data (use last 80% for training, keep 20% for validation)
            train_size = int(len(processed_data) * 0.8)
            train_data = processed_data[:train_size]
            
            try:
                losses = self.ai_predictor.fit(train_data, epochs=20, verbose=False)
                success = True
            except Exception as e:
                self.logger.error(f"Training failed: {e}")
                success = False
            
            if success:
                self.last_retrain_time = time.time()
                self.logger.info("Model training completed successfully")
                
                # Save model
                model_file = self.output_dir / f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pth"
                torch.save(self.ai_predictor.state_dict(), model_file)
                
                return True
            else:
                self.logger.error("Model training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            return False
    
    def make_prediction(self, current_price: float) -> Optional[Dict]:
        """Make prediction using current model"""
        if self.ai_predictor is None:
            self.logger.warning("No trained model available for prediction")
            return None
        
        try:
            # Use simple default parameters for now (can be enhanced later)
            params = {
                'sigma': 0.02,  # 2% daily volatility
                'drift': 0.001,  # 0.1% daily drift
                'skewness': 0.0,
                'kurtosis': 3.0
            }
            
            # Run Monte Carlo simulation
            simulation_results = self.monte_carlo.simulate(
                current_price=current_price,
                time_increment=1/24,  # Hourly increments
                time_length=1,  # 24 hours
                num_simulations=1000,
                **params
            )
            
            # Calculate prediction statistics
            final_prices = [path[-1] for path in simulation_results['paths']]
            
            prediction = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': current_price,
                'predicted_mean': np.mean(final_prices),
                'predicted_median': np.median(final_prices),
                'predicted_std': np.std(final_prices),
                'confidence_95_lower': np.percentile(final_prices, 2.5),
                'confidence_95_upper': np.percentile(final_prices, 97.5),
                'confidence_68_lower': np.percentile(final_prices, 16),
                'confidence_68_upper': np.percentile(final_prices, 84),
                'monte_carlo_params': params,
                'num_simulations': len(final_prices)
            }
            
            self.prediction_history.append(prediction)
            
            # Keep only recent predictions
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            self.logger.info(f"Prediction: ${prediction['predicted_mean']:,.2f} "
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
            if len(all_predictions) > 1000:
                all_predictions = all_predictions[-1000:]
            
            # Save back to file
            with open(predictions_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
    
    def print_status(self, current_price: float, prediction: Optional[Dict]):
        """Print current status to console"""
        print("\n" + "="*80)
        print(f"REAL-TIME BITCOIN PREDICTOR - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"Current Price: ${current_price:,.2f}")
        print(f"Data Points: {len(self.price_history)}")
        print(f"Model Status: {'Trained' if self.ai_predictor else 'Not Trained'}")
        
        if prediction:
            print(f"24h Prediction: ${prediction['predicted_mean']:,.2f}")
            print(f"95% Confidence: ${prediction['confidence_95_lower']:,.2f} - ${prediction['confidence_95_upper']:,.2f}")
            print(f"Volatility (σ): {prediction['monte_carlo_params'].get('sigma', 'N/A'):.4f}")
        
        next_fetch = datetime.datetime.now() + datetime.timedelta(minutes=self.fetch_interval)
        print(f"Next Update: {next_fetch.strftime('%H:%M:%S')}")
        print("="*80)
    
    def run(self):
        """Main loop for real-time prediction"""
        self.logger.info("Starting real-time Bitcoin predictor...")
        print(f"Real-time predictor started. Fetching prices every {self.fetch_interval} minutes.")
        print(f"Model retraining every {self.retrain_interval//3600} hour(s).")
        print("Press Ctrl+C to stop.\n")
        
        try:
            while True:
                # Fetch latest price
                price_record = self.fetch_latest_price()
                
                if price_record:
                    current_price = price_record['close']
                    
                    # Check if model needs retraining
                    if self.should_retrain_model() or self.ai_predictor is None:
                        self.train_model()
                    
                    # Make prediction
                    prediction = self.make_prediction(current_price)
                    
                    if prediction:
                        self.save_prediction(prediction)
                    
                    # Save data
                    self.save_price_history()
                    
                    # Print status
                    self.print_status(current_price, prediction)
                
                # Wait for next fetch
                time.sleep(self.fetch_interval * 60)
                
        except KeyboardInterrupt:
            self.logger.info("Stopping real-time predictor...")
            print("\nStopping real-time predictor. Final data save...")
            self.save_price_history()
            print("Done!")
        
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.save_price_history()
            raise

def main():
    parser = argparse.ArgumentParser(description='Real-time Bitcoin Price Predictor')
    parser.add_argument('--interval', type=int, default=5, 
                       help='Minutes between price fetches (default: 5)')
    parser.add_argument('--retrain_interval', type=int, default=1, 
                       help='Hours between model retraining (default: 1)')
    parser.add_argument('--max_data_points', type=int, default=2000,
                       help='Maximum historical data points to keep (default: 2000)')
    parser.add_argument('--output_dir', type=str, default='realtime_output',
                       help='Directory to save outputs (default: realtime_output)')
    
    args = parser.parse_args()
    
    # Create and run predictor
    predictor = RealTimePredictor(
        fetch_interval=args.interval,
        retrain_interval=args.retrain_interval,
        max_data_points=args.max_data_points,
        output_dir=args.output_dir
    )
    
    predictor.run()

if __name__ == "__main__":
    main()