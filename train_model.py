#!/usr/bin/env python3
"""
Bitcoin Monte Carlo AI Predictor - Command Line Training Interface

Usage:
    python train_model.py --csv_file path/to/data.csv --epochs 50 --learning_rate 0.001
    python train_model.py --csv_file data.csv --sequence_length 30 --batch_size 32
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from datetime import datetime

# Add project modules to path
sys.path.append(str(Path(__file__).parent))

from models.ai_predictor import BitcoinParameterPredictor
from data.data_processor import DataProcessor
from models.monte_carlo import simulate_crypto_price_paths
from utils.price_fetcher import BitcoinPriceFetcher
import plotly.graph_objects as go


def load_csv_data(csv_file_path: str) -> str:
    """Load CSV data from file"""
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    with open(csv_file_path, 'r') as file:
        csv_data = file.read()
    
    return csv_data


def train_model(csv_file: str, epochs: int = 50, learning_rate: float = 0.001, 
                sequence_length: int = 30, batch_size: int = 32, 
                hidden_size: int = 128, num_layers: int = 3,
                intervals_per_day: int = 288,
                model_save_path: str = "models/bitcoin_predictor.pth"):
    """Train the AI model"""
    
    print(f"Starting training with parameters:")
    print(f"  CSV file: {csv_file}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Model save path: {model_save_path}")
    print("-" * 50)
    
    # Load and process data
    print("Loading CSV data...")
    csv_data = load_csv_data(csv_file)
    
    data_processor = DataProcessor()
    print("Processing training data...")
    processed_data = data_processor.process_training_data(csv_data, sequence_length=sequence_length)
    
    print(f"Processed {len(processed_data)} data points")
    
    # Initialize model with upgraded architecture for interval prediction
    print("Initializing AI model for 5-minute interval prediction...")
    model = BitcoinParameterPredictor(
        input_size=4,  # OHLC data
        hidden_size=hidden_size,
        num_layers=num_layers,
        sequence_length=sequence_length,
        intervals_per_day=intervals_per_day  # 288 5-minute intervals = 24 hours
    )
    
    # Train model
    print("Starting training...")
    train_losses = model.train_model(
        processed_data,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    # Save model
    print(f"Saving model to {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_model(model_save_path)
    
    print("Training completed successfully!")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    
    return model, train_losses


def predict_parameters(model_path: str = "models/bitcoin_predictor.pth", 
                      days_history: int = 7):
    """Predict Monte Carlo parameters using trained model"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = BitcoinParameterPredictor.load_model(model_path)
    
    print(f"Fetching recent {days_history} days of Bitcoin price data...")
    price_fetcher = BitcoinPriceFetcher()
    recent_data = price_fetcher.get_recent_data(days=days_history)
    
    if not recent_data or len(recent_data) < model.sequence_length:
        raise ValueError(f"Insufficient recent data. Need at least {model.sequence_length} data points, got {len(recent_data)}")
    
    print("Predicting interval-based Monte Carlo parameters...")
    predicted_params = model.predict_interval_parameters(recent_data)
    
    print("Predicted parameters summary:")
    print(f"  Volatility array (288 values): min={np.min(predicted_params['sigma']):.6f}, max={np.max(predicted_params['sigma']):.6f}, mean={np.mean(predicted_params['sigma']):.6f}")
    print(f"  Skewness array (288 values): min={np.min(predicted_params['skewness']):.6f}, max={np.max(predicted_params['skewness']):.6f}, mean={np.mean(predicted_params['skewness']):.6f}")
    print(f"  Kurtosis array (288 values): min={np.min(predicted_params['kurtosis']):.6f}, max={np.max(predicted_params['kurtosis']):.6f}, mean={np.mean(predicted_params['kurtosis']):.6f}")
    
    return predicted_params


def run_simulation(model_path: str = "models/bitcoin_predictor.pth",
                  num_simulations: int = 1000, time_increment: int = 300,
                  time_length: int = 86400, use_ai_params: bool = True,
                  save_results: bool = True):
    """Run Monte Carlo simulation"""
    
    print("Fetching current Bitcoin price...")
    price_fetcher = BitcoinPriceFetcher()
    current_price, timestamp = price_fetcher.fetch_latest_bitcoin_price()
    
    if not current_price:
        raise ValueError("Failed to fetch current Bitcoin price")
    
    print(f"Current Bitcoin price: ${current_price:,.2f}")
    
    if use_ai_params and os.path.exists(model_path):
        print("Using AI-predicted parameters...")
        predicted_params = predict_parameters(model_path)
        ai_params = predicted_params
    else:
        print("Using default parameters...")
        predicted_params = None
        ai_params = None
    
    print(f"Running Monte Carlo simulation with {num_simulations} paths...")
    print(f"Time increment: {time_increment} seconds")
    print(f"Simulation length: {time_length} seconds ({time_length/3600:.1f} hours)")
    
    price_paths = simulate_crypto_price_paths(
        current_price=current_price,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
        asset='BTC',
        ai_params=ai_params
    )
    
    # Calculate statistics
    final_prices = price_paths[:, -1]
    
    print("\nSimulation Results:")
    print(f"  Mean final price: ${np.mean(final_prices):,.2f}")
    print(f"  Median final price: ${np.median(final_prices):,.2f}")
    print(f"  Min final price: ${np.min(final_prices):,.2f}")
    print(f"  Max final price: ${np.max(final_prices):,.2f}")
    print(f"  Standard deviation: ${np.std(final_prices):,.2f}")
    print(f"  5th percentile: ${np.percentile(final_prices, 5):,.2f}")
    print(f"  95th percentile: ${np.percentile(final_prices, 95):,.2f}")
    
    # Calculate price change statistics
    price_changes = (final_prices - current_price) / current_price * 100
    print(f"  Mean price change: {np.mean(price_changes):+.2f}%")
    print(f"  Probability of gain: {np.mean(price_changes > 0)*100:.1f}%")
    print(f"  Probability of loss: {np.mean(price_changes < 0)*100:.1f}%")
    
    if save_results:
        # Save results to CSV
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"simulation_results_{timestamp_str}.csv"
        
        results_df = pd.DataFrame({
            'simulation_id': range(num_simulations),
            'final_price': final_prices,
            'price_change_pct': price_changes
        })
        
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
    
    return price_paths, predicted_params if use_ai_params else None


def main():
    parser = argparse.ArgumentParser(description='Bitcoin Monte Carlo AI Predictor - Command Line Interface')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the AI model')
    train_parser.add_argument('--csv_file', required=True, help='Path to CSV training data file')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    train_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    train_parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length (default: 30)')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    train_parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size (default: 64)')
    train_parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers (default: 2)')
    train_parser.add_argument('--model_save_path', default='models/bitcoin_predictor.pth', help='Model save path')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict Monte Carlo parameters')
    predict_parser.add_argument('--model_path', default='models/bitcoin_predictor.pth', help='Path to trained model')
    predict_parser.add_argument('--days_history', type=int, default=7, help='Days of historical data to use (default: 7)')
    
    # Simulate command
    simulate_parser = subparsers.add_parser('simulate', help='Run Monte Carlo simulation')
    simulate_parser.add_argument('--model_path', default='models/bitcoin_predictor.pth', help='Path to trained model')
    simulate_parser.add_argument('--num_simulations', type=int, default=1000, help='Number of simulations (default: 1000)')
    simulate_parser.add_argument('--time_increment', type=int, default=300, help='Time increment in seconds (default: 300)')
    simulate_parser.add_argument('--time_length', type=int, default=86400, help='Simulation length in seconds (default: 86400)')
    simulate_parser.add_argument('--no_ai', action='store_true', help='Use default parameters instead of AI predictions')
    simulate_parser.add_argument('--no_save', action='store_true', help="Don't save simulation results")
    
    # All-in-one command
    all_parser = subparsers.add_parser('all', help='Train model and run simulation')
    all_parser.add_argument('--csv_file', required=True, help='Path to CSV training data file')
    all_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    all_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    all_parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length (default: 30)')
    all_parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    all_parser.add_argument('--num_simulations', type=int, default=1000, help='Number of simulations (default: 1000)')
    all_parser.add_argument('--time_increment', type=int, default=300, help='Time increment in seconds (default: 300)')
    all_parser.add_argument('--time_length', type=int, default=86400, help='Simulation length in seconds (default: 86400)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            train_model(
                csv_file=args.csv_file,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                model_save_path=args.model_save_path
            )
            
        elif args.command == 'predict':
            predict_parameters(
                model_path=args.model_path,
                days_history=args.days_history
            )
            
        elif args.command == 'simulate':
            run_simulation(
                model_path=args.model_path,
                num_simulations=args.num_simulations,
                time_increment=args.time_increment,
                time_length=args.time_length,
                use_ai_params=not args.no_ai,
                save_results=not args.no_save
            )
            
        elif args.command == 'all':
            print("Step 1: Training model...")
            train_model(
                csv_file=args.csv_file,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size
            )
            
            print("\nStep 2: Running simulation...")
            run_simulation(
                num_simulations=args.num_simulations,
                time_increment=args.time_increment,
                time_length=args.time_length,
                use_ai_params=True,
                save_results=True
            )
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()