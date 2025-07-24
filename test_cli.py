#!/usr/bin/env python3
"""
Test script for the command line interface
"""

import subprocess
import sys
import os
from pathlib import Path

def test_help():
    """Test that help commands work"""
    print("Testing help commands...")
    
    commands = [
        "python train_model.py --help",
        "python train_model.py train --help",
        "python train_model.py predict --help", 
        "python train_model.py simulate --help",
        "python train_model.py all --help"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✓ Help command works")
            else:
                print(f"✗ Help command failed: {result.stderr}")
        except Exception as e:
            print(f"✗ Error running command: {e}")
        print()

def create_sample_data():
    """Create a small sample CSV file for testing"""
    sample_data = """timestamp,open,high,low,close,volume
2022-02-03 14:50:00,36551.864,36618.271,36500.254,36500.254,0.0
2022-02-03 14:55:00,36500.254,36580.123,36490.100,36545.678,0.0
2022-02-03 15:00:00,36545.678,36620.456,36520.789,36598.123,0.0
2022-02-03 15:05:00,36598.123,36650.234,36580.567,36625.890,0.0
2022-02-03 15:10:00,36625.890,36680.345,36600.123,36655.234,0.0
"""
    
    with open('sample_data.csv', 'w') as f:
        f.write(sample_data)
    
    print("Created sample_data.csv for testing")

def test_predict_without_model():
    """Test prediction without a trained model"""
    print("Testing prediction without trained model...")
    
    cmd = "python train_model.py predict"
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
        if "Model file not found" in result.stdout or "Model file not found" in result.stderr:
            print("✓ Correctly handles missing model file")
        else:
            print(f"✗ Unexpected output: {result.stdout}")
    except Exception as e:
        print(f"✗ Error: {e}")

def test_simulate_without_model():
    """Test simulation with default parameters (no AI model)"""
    print("Testing simulation with default parameters...")
    
    cmd = "python train_model.py simulate --no_ai --num_simulations 10 --no_save"
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ Simulation with default parameters works")
            print("Sample output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        else:
            print(f"✗ Simulation failed: {result.stderr}")
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    print("Bitcoin Monte Carlo AI Predictor - CLI Test Suite")
    print("=" * 60)
    
    # Test 1: Help commands
    test_help()
    
    # Test 2: Create sample data
    create_sample_data()
    
    # Test 3: Predict without model
    test_predict_without_model()
    
    # Test 4: Simulate without model
    test_simulate_without_model()
    
    print("\nTest Summary:")
    print("- Help commands: Should show usage information")
    print("- Sample data: Created for testing")
    print("- Prediction: Should handle missing model gracefully")
    print("- Simulation: Should work with default parameters")
    
    print("\nNext steps:")
    print("1. Test with real data: python train_model.py train --csv_file your_data.csv --epochs 5")
    print("2. Run full simulation: python train_model.py simulate")
    print("3. Complete workflow: python train_model.py all --csv_file your_data.csv --epochs 5")

if __name__ == "__main__":
    main()