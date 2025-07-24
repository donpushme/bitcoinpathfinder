# Bitcoin Monte Carlo AI Predictor - Setup Guide

This guide will help you set up and run the Bitcoin Monte Carlo AI Predictor application on your local machine.

## Prerequisites

- Python 3.8 or higher
- Internet connection (for fetching real-time Bitcoin prices)
- At least 4GB RAM (recommended 8GB+ for large datasets)
- 1GB free disk space

## Installation

### 1. Clone or Download the Project

If you're using this on Replit, the files are already available. For local setup:

```bash
# Download all project files to your local directory
# Ensure you have all these files:
# - app.py (Streamlit web interface)
# - train_model.py (Command line interface)
# - models/ (AI predictor and Monte Carlo simulation)
# - data/ (Data processing utilities)
# - utils/ (Price fetcher and visualization)
```

### 2. Install Python Dependencies

#### Option A: Using pip (Standard Python)
```bash
pip install streamlit torch numpy pandas plotly requests scikit-learn scipy
```

#### Option B: Using conda (Anaconda/Miniconda)
```bash
conda install -c conda-forge streamlit pytorch numpy pandas plotly requests scikit-learn scipy
```

#### Option C: Using requirements.txt (if provided)
```bash
pip install -r requirements.txt
```

### 3. Verify Installation

Test that all dependencies are installed correctly:

```bash
python -c "import torch, streamlit, pandas, numpy, plotly, requests, sklearn, scipy; print('All dependencies installed successfully!')"
```

## Quick Start Options

### Option 1: Web Interface (Recommended for Beginners)

1. **Start the web application:**
   ```bash
   streamlit run app.py --server.port 5000
   ```

2. **Open your browser:**
   - Go to `http://localhost:5000`
   - The application will open in your browser

3. **Upload your CSV data:**
   - Click "Upload Bitcoin price data (CSV)" in the sidebar
   - Select your CSV file with format: `timestamp,open,high,low,close,volume`

4. **Train the model:**
   - Select "Train New" in the sidebar
   - Adjust training parameters if needed
   - Click "Start Training"

5. **Run simulations:**
   - Click "Fetch Current Bitcoin Price"
   - Click "Run AI-Powered Monte Carlo Simulation"

### Option 2: Command Line Interface (Advanced Users)

1. **Prepare your data:**
   - Ensure your CSV file has columns: `timestamp,open,high,low,close,volume`
   - Example format:
     ```csv
     timestamp,open,high,low,close,volume
     2022-02-03 14:50:00,36551.864,36618.271,36500.254,36500.254,0.0
     ```

2. **Train and run simulation (one command):**
   ```bash
   python train_model.py all --csv_file your_data.csv
   ```

3. **Or step by step:**
   ```bash
   # Train model
   python train_model.py train --csv_file your_data.csv --epochs 50
   
   # Run simulation
   python train_model.py simulate --num_simulations 1000
   ```

## Data Requirements

### CSV Data Format
Your Bitcoin price data must be in CSV format with these exact columns:

```csv
timestamp,open,high,low,close,volume
2022-02-03 14:50:00,36551.864,36618.271,36500.254,36500.254,0.0
2022-02-03 14:55:00,36500.254,36580.123,36490.100,36545.678,0.0
```

**Column Descriptions:**
- `timestamp`: Date and time (YYYY-MM-DD HH:MM:SS format)
- `open`: Opening price for the time period
- `high`: Highest price during the time period
- `low`: Lowest price during the time period
- `close`: Closing price for the time period
- `volume`: Trading volume (ignored by the model but must be present)

### Data Quality Recommendations

- **Minimum data points:** 10,000+ for basic training, 50,000+ for good results
- **Time intervals:** 5-minute intervals work well (1-minute to 1-hour also supported)
- **Data span:** At least 30 days, preferably 90+ days
- **Completeness:** Minimize missing data points

## Configuration Options

### Web Interface Configuration

The Streamlit app can be configured by editing `.streamlit/config.toml`:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000

[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Command Line Parameters

**Training Parameters:**
- `--epochs`: Number of training iterations (default: 50)
- `--learning_rate`: Learning speed (default: 0.001)
- `--sequence_length`: Input sequence length (default: 30)
- `--batch_size`: Training batch size (default: 32)
- `--hidden_size`: Neural network size (default: 64)

**Simulation Parameters:**
- `--num_simulations`: Number of price paths (default: 1000)
- `--time_increment`: Time step in seconds (default: 300 = 5 minutes)
- `--time_length`: Total simulation time in seconds (default: 86400 = 24 hours)

## Troubleshooting

### Common Installation Issues

**1. PyTorch Installation Issues**
```bash
# For CPU-only version (smaller download)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For systems with CUDA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. Streamlit Port Issues**
```bash
# Use a different port if 5000 is busy
streamlit run app.py --server.port 8501
```

**3. Memory Issues**
```bash
# Reduce batch size for training
python train_model.py train --csv_file data.csv --batch_size 16

# Reduce number of simulations
python train_model.py simulate --num_simulations 500
```

### Common Runtime Issues

**"Model file not found"**
- Solution: Train a model first using either the web interface or CLI

**"Failed to fetch Bitcoin price"**
- Solution: Check your internet connection
- Alternative: Use `--no_ai` flag to run with default parameters

**"Insufficient data after cleaning"**
- Solution: Use a larger dataset or reduce `--sequence_length`

**"Training loss not decreasing"**
- Solution: Try different learning rates (0.0001 to 0.01)
- Check data quality and remove outliers

## Performance Optimization

### For Large Datasets (100K+ points)
```bash
# Use larger batch size and hidden size
python train_model.py train --csv_file large_data.csv --batch_size 64 --hidden_size 128
```

### For Quick Testing
```bash
# Use fewer epochs and smaller model
python train_model.py train --csv_file data.csv --epochs 10 --hidden_size 32
```

### For Production Use
```bash
# Use more epochs and larger model
python train_model.py train --csv_file data.csv --epochs 100 --hidden_size 128 --num_layers 3
```

## File Structure After Setup

```
bitcoin-monte-carlo-predictor/
├── app.py                     # Web interface
├── train_model.py            # Command line interface
├── SETUP.md                  # This setup guide
├── CLI_README.md             # CLI documentation
├── models/
│   ├── ai_predictor.py       # AI model
│   ├── monte_carlo.py        # Simulation engine
│   └── bitcoin_predictor.pth # Trained model (after training)
├── data/
│   └── data_processor.py     # Data processing
├── utils/
│   ├── price_fetcher.py      # Price data fetching
│   └── visualization.py     # Chart creation
├── .streamlit/
│   └── config.toml           # Streamlit configuration
└── simulation_results_*.csv  # Simulation outputs
```

## Next Steps

1. **Start with the web interface** if you're new to the application
2. **Use command line** for automated training and batch processing
3. **Experiment with parameters** to find optimal settings for your data
4. **Save your trained models** for reuse
5. **Analyze simulation results** to understand Bitcoin price forecasts

## Getting Help

- Check the `CLI_README.md` for detailed command line usage
- Review error messages carefully - they often indicate the exact issue
- Ensure your CSV data format matches the requirements exactly
- Start with small datasets and fewer epochs for initial testing

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.9+ |
| RAM | 4GB | 8GB+ |
| Storage | 1GB | 2GB+ |
| Internet | Required for price data | Stable connection |
| CPU | Any modern CPU | Multi-core preferred |
| GPU | Not required | CUDA-compatible for faster training |