#!/usr/bin/env python3
"""
Quick Start Script for Bitcoin Monte Carlo AI Predictor

This script helps you get started quickly with the application.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'torch', 'streamlit', 'pandas', 'numpy', 
        'plotly', 'requests', 'sklearn', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with:")
        if 'torch' in missing_packages:
            print("pip install torch")
        if 'sklearn' in missing_packages:
            print("pip install scikit-learn")
        for pkg in missing_packages:
            if pkg not in ['torch', 'sklearn']:
                print(f"pip install {pkg}")
        return False
    
    print("\n✓ All dependencies are installed!")
    return True

def create_sample_data():
    """Create a sample CSV file for testing"""
    sample_data = """timestamp,open,high,low,close,volume
2022-02-03 14:50:00,36551.864,36618.271,36500.254,36500.254,0.0
2022-02-03 14:55:00,36500.254,36580.123,36490.100,36545.678,0.0
2022-02-03 15:00:00,36545.678,36620.456,36520.789,36598.123,0.0
2022-02-03 15:05:00,36598.123,36650.234,36580.567,36625.890,0.0
2022-02-03 15:10:00,36625.890,36680.345,36600.123,36655.234,0.0
2022-02-03 15:15:00,36655.234,36700.567,36640.890,36682.345,0.0
2022-02-03 15:20:00,36682.345,36720.678,36660.123,36705.456,0.0
2022-02-03 15:25:00,36705.456,36750.789,36685.234,36728.567,0.0
2022-02-03 15:30:00,36728.567,36780.890,36710.345,36751.678,0.0
2022-02-03 15:35:00,36751.678,36800.123,36735.456,36774.789,0.0
"""
    
    filename = 'sample_bitcoin_data.csv'
    with open(filename, 'w') as f:
        f.write(sample_data)
    
    print(f"✓ Created {filename} for testing")
    return filename

def show_usage_options():
    """Show usage options to the user"""
    print("\n" + "="*60)
    print("QUICK START OPTIONS:")
    print("="*60)
    
    print("\n1. WEB INTERFACE (Recommended for beginners):")
    print("   streamlit run app.py --server.port 5000")
    print("   Then open: http://localhost:5000")
    
    print("\n2. COMMAND LINE (Advanced users):")
    print("   # Train model:")
    print("   python train_model.py train --csv_file sample_bitcoin_data.csv --epochs 10")
    print("   ")
    print("   # Run simulation:")
    print("   python train_model.py simulate --num_simulations 100")
    print("   ")
    print("   # Do both:")
    print("   python train_model.py all --csv_file sample_bitcoin_data.csv --epochs 10")
    
    print("\n3. HELP COMMANDS:")
    print("   python train_model.py --help")
    print("   python train_model.py train --help")
    print("   python train_model.py simulate --help")
    
    print("\n4. DOCUMENTATION:")
    print("   - SETUP.md: Complete setup guide")
    print("   - CLI_README.md: Command line documentation")
    print("   - replit.md: Technical architecture")

def main():
    print("Bitcoin Monte Carlo AI Predictor - Quick Start")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies first.")
        return
    
    # Create sample data
    sample_file = create_sample_data()
    
    # Show usage options
    show_usage_options()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("1. Choose either web interface or command line")
    print("2. For real data, replace sample_bitcoin_data.csv with your CSV file")
    print("3. Adjust parameters as needed (epochs, simulations, etc.)")
    print("4. Check the documentation files for detailed information")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()