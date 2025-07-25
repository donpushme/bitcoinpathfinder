# Real-time Bitcoin Price Monitor and Predictor

This document describes the real-time monitoring script that continuously fetches Bitcoin prices and generates predictions using Monte Carlo simulation.

## Overview

The `realtime_monitor.py` script provides:
- **Continuous Price Fetching**: Fetches Bitcoin prices every 5 minutes (configurable)
- **Dynamic Volatility Calculation**: Estimates volatility from recent price data
- **Monte Carlo Predictions**: Generates 24-hour price forecasts with confidence intervals
- **Data Persistence**: Saves all data for later analysis and model training
- **Live Monitoring**: Real-time console output with status updates

## Quick Start

### Basic Usage
```bash
# Start monitoring with default settings (5-minute intervals)
python realtime_monitor.py

# Custom interval (every 2 minutes)
python realtime_monitor.py --interval 2

# Custom output directory
python realtime_monitor.py --output_dir my_bitcoin_data
```

### Example Output
```
================================================================================
REAL-TIME BITCOIN MONITOR - 2025-07-24 22:22:30
================================================================================
Current Price: $118,535.60
Data Points Collected: 1
Predictions Made: 1
24h Prediction: $118,535.97
95% Confidence: $118,511.93 - $118,557.47
Current Volatility: 0.0200 (2.00%)
Next Update: 22:23:30
Data saved to: realtime_data
================================================================================
```

## Features

### 1. Price Data Collection
- Fetches real-time Bitcoin prices using the existing price fetcher
- Creates OHLC (Open, High, Low, Close) format for compatibility
- Automatically saves data in CSV format for training
- Maintains a rolling history of the last 2000 data points

### 2. Dynamic Volatility Estimation
- Calculates volatility from the last 20 price points
- Converts from 5-minute intervals to daily volatility
- Uses logarithmic returns for accurate calculation
- Clamps volatility between 0.5% and 10% for stability

### 3. Monte Carlo Predictions
- Generates 500 simulation paths for 24-hour forecasts
- Uses adaptive parameters based on recent market data
- Provides comprehensive statistics:
  - Mean and median predictions
  - 95% and 68% confidence intervals
  - Standard deviation of predictions

### 4. Data Persistence
- **Price History**: `realtime_data/price_history.csv`
- **Training Data**: `realtime_data/bitcoin_training_data.csv`
- **Predictions**: `realtime_data/predictions.json`
- **Current Status**: `realtime_data/current_status.json`
- **Log Files**: `realtime_data/monitor_log_YYYYMMDD.log`

## Command Line Options

```bash
python realtime_monitor.py [OPTIONS]

Options:
  --interval INT        Minutes between price fetches (default: 5)
  --output_dir TEXT     Directory to save outputs (default: realtime_data)
  --help               Show help message and exit
```

## Integration with Existing System

### Training Models with Real-time Data
Once you've collected data, you can train the AI model:

```bash
# Train using collected real-time data
python train_model.py train --csv_file realtime_data/bitcoin_training_data.csv --epochs 50

# Run complete workflow with real-time data
python train_model.py all --csv_file realtime_data/bitcoin_training_data.csv
```

### Using with Web Interface
The collected data is automatically saved in the correct format for the Streamlit web interface:

```bash
# Start web interface
streamlit run app.py --server.port 5000
# Then upload realtime_data/bitcoin_training_data.csv for training
```

## Data Format

### Price History CSV
```csv
timestamp,open,high,low,close,volume
2025-07-24 22:22:30,118535.60,118543.60,118527.60,118535.60,0.0
```

### Predictions JSON
```json
{
  "timestamp": "2025-07-24 22:22:30",
  "current_price": 118535.60,
  "predicted_mean": 118535.97,
  "confidence_95_lower": 118511.93,
  "confidence_95_upper": 118557.47,
  "volatility_estimate": 0.02,
  "monte_carlo_params": {
    "daily_sigma": 0.02,
    "daily_drift": 0.0001,
    "skewness": 0.0,
    "kurtosis": 3.0
  }
}
```

## Performance and Reliability

### Resource Usage
- **Memory**: Keeps only the last 2000 price points in memory
- **Storage**: Approximately 1MB per day of continuous monitoring
- **CPU**: Minimal usage, Monte Carlo simulation runs in ~0.5 seconds

### Error Handling
- Automatic retry for failed price fetches
- Graceful handling of network errors
- Continues operation even if individual predictions fail
- All errors logged to daily log files

### Data Integrity
- Uses authentic price data from Pyth Network and Binance APIs
- Validates price data before processing
- Automatic data backup and recovery
- Consistent file format for integration

## Stopping and Restarting

### Graceful Shutdown
Press `Ctrl+C` to stop the monitor. It will:
1. Save all current data
2. Create final summary
3. Display instructions for using collected data

### Resuming Monitoring
The script automatically:
- Loads existing price history on startup
- Continues from where it left off
- Appends new data to existing files

## Troubleshooting

### Common Issues

**Price Fetch Failures**
- Check internet connection
- API rate limits (script includes appropriate delays)
- Check log files for detailed error messages

**Storage Issues**
- Ensure sufficient disk space
- Check write permissions for output directory
- Monitor log files for I/O errors

**Prediction Errors**
- Requires minimum data points for volatility calculation
- Check log files for Monte Carlo simulation errors
- Verify price data format

### Log Files
Daily log files contain detailed information:
```bash
# View today's log
tail -f realtime_data/monitor_log_20250724.log

# Search for errors
grep "ERROR" realtime_data/monitor_log_*.log
```

## Advanced Usage

### Custom Parameters
You can modify the volatility calculation and prediction parameters by editing the script:

```python
# In calculate_volatility()
lookback_periods = 20  # Number of periods for volatility calculation

# In make_prediction()
num_simulations = 500  # Number of Monte Carlo paths
time_length = 1        # Forecast horizon (days)
```

### Integration with Other Systems
The JSON output format makes it easy to integrate with:
- Trading systems
- Alert mechanisms
- Dashboard applications
- Analysis tools

## Next Steps

1. **Start Monitoring**: Run the script to begin collecting data
2. **Let it Run**: Allow several hours or days of data collection
3. **Train Models**: Use collected data to train the AI predictor
4. **Compare Results**: Analyze prediction accuracy over time
5. **Optimize Parameters**: Adjust intervals and parameters based on results

The real-time monitor provides a solid foundation for continuous Bitcoin price analysis and prediction using the Monte Carlo AI system.