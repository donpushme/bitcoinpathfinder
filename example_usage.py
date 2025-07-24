#!/usr/bin/env python3
"""
Example usage of the Bitcoin Monte Carlo AI Predictor command line interface

This script shows how to use the train_model.py script for different scenarios.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and display the results"""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    # Uncomment the line below to actually run the command
    # subprocess.run(command, shell=True)
    print("(Command not executed - uncomment subprocess.run to execute)")

def main():
    print("Bitcoin Monte Carlo AI Predictor - Command Line Examples")
    print("=" * 60)
    
    # Example 1: Train a model
    run_command(
        "python train_model.py train --csv_file data/bitcoin_data.csv --epochs 100 --learning_rate 0.001",
        "Train model with 100 epochs using custom learning rate"
    )
    
    # Example 2: Train with custom parameters
    run_command(
        "python train_model.py train --csv_file data/bitcoin_data.csv --epochs 50 --sequence_length 20 --batch_size 64 --hidden_size 128",
        "Train model with custom architecture and batch size"
    )
    
    # Example 3: Predict parameters only
    run_command(
        "python train_model.py predict --model_path models/bitcoin_predictor.pth --days_history 14",
        "Predict Monte Carlo parameters using 14 days of history"
    )
    
    # Example 4: Run simulation with AI parameters
    run_command(
        "python train_model.py simulate --num_simulations 2000 --time_increment 300 --time_length 86400",
        "Run 2000 simulations over 24 hours with 5-minute intervals"
    )
    
    # Example 5: Run simulation with default parameters (no AI)
    run_command(
        "python train_model.py simulate --num_simulations 1000 --no_ai --no_save",
        "Run simulation with default parameters and don't save results"
    )
    
    # Example 6: Complete workflow (train + simulate)
    run_command(
        "python train_model.py all --csv_file data/bitcoin_data.csv --epochs 75 --num_simulations 1500",
        "Complete workflow: train model then run simulation"
    )
    
    print(f"\n{'='*60}")
    print("QUICK START GUIDE:")
    print("1. Prepare your CSV file with columns: timestamp,open,high,low,close,volume")
    print("2. Train a model: python train_model.py train --csv_file your_data.csv")
    print("3. Run simulation: python train_model.py simulate")
    print("4. Or do both: python train_model.py all --csv_file your_data.csv")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()