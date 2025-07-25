# Bitcoin Monte Carlo AI Predictor - Command Line Interface

This guide shows how to use the Bitcoin Monte Carlo AI Predictor without the web interface, directly from the command line using Python scripts.

## Quick Start

### 1. Prepare Your Data
Ensure your CSV file has the following format:
```csv
timestamp,open,high,low,close,volume
2022-02-03 14:50:00,36551.864,36618.271,36500.254,36500.254,0.0
2022-02-03 14:55:00,36500.254,36580.123,36490.100,36545.678,0.0
```

### 2. Train and Run (All-in-One)
```bash
python train_model.py all --csv_file your_data.csv
```

### 3. Or Step by Step

#### Train Model
```bash
python train_model.py train --csv_file your_data.csv --epochs 50
```

#### Run Simulation
```bash
python train_model.py simulate --num_simulations 1000
```

## Available Commands

### `train` - Train the AI Model
Train a new LSTM model using your historical Bitcoin price data.

**Basic Usage:**
```bash
python train_model.py train --csv_file data.csv
```

**Advanced Usage:**
```bash
python train_model.py train \
    --csv_file data.csv \
    --epochs 100 \
    --learning_rate 0.001 \
    --sequence_length 30 \
    --output_steps 288 \
    --batch_size 32 \
    --hidden_size 64 \
    --num_layers 2 \
    --model_save_path models/my_model.pth
```

**Parameters:**
- `--csv_file`: Path to your CSV training data (required)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate for training (default: 0.001)
- `--sequence_length`: Input sequence length (default: 30)
- `--output_steps`: Number of output steps (default: 288, for 24h at 5-min intervals)
- `--batch_size`: Training batch size (default: 32)
- `--hidden_size`: LSTM hidden layer size (default: 64)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--model_save_path`: Where to save the trained model (default: models/bitcoin_predictor.pth)

### `predict` - Predict Monte Carlo Parameters
Use a trained model to predict 288-step arrays for sigma, skewness, and kurtosis for Monte Carlo simulation.

**Usage:**
```bash
python train_model.py predict
```

**Advanced Usage:**
```bash
python train_model.py predict \
    --model_path models/bitcoin_predictor.pth \
    --days_history 7
```

**Parameters:**
- `--model_path`: Path to trained model file (default: models/bitcoin_predictor.pth)
- `--days_history`: Days of recent data to use for prediction (default: 7)

### `simulate` - Run Monte Carlo Simulation
Run a Monte Carlo simulation to forecast Bitcoin price paths using the 288-step parameter arrays.

**Usage:**
```bash
python train_model.py simulate
```

**Advanced Usage:**
```bash
python train_model.py simulate \
    --model_path models/bitcoin_predictor.pth \
    --num_simulations 2000 \
    --time_increment 300 \
    --time_length 86400 \
    --no_ai \
    --no_save
```

**Parameters:**
- `--model_path`: Path to trained model file (default: models/bitcoin_predictor.pth)
- `--num_simulations`: Number of simulation paths (default: 1000)
- `--time_increment`: Time step in seconds (default: 300 = 5 minutes)
- `--time_length`: Total simulation time in seconds (default: 86400 = 24 hours)
- `--no_ai`: Use default parameters instead of AI predictions (optional)
- `--no_save`: Don't save simulation results to CSV (optional)

### `all` - Complete Workflow
Train a model and immediately run a simulation.

**Usage:**
```bash
python train_model.py all --csv_file data.csv
```

**Advanced Usage:**
```bash
python train_model.py all \
    --csv_file data.csv \
    --epochs 75 \
    --learning_rate 0.001 \
    --sequence_length 30 \
    --output_steps 288 \
    --batch_size 32 \
    --num_simulations 1500 \
    --time_increment 300 \
    --time_length 86400
```

## Output Examples

### Training Output
```
Starting training with parameters:
  CSV file: data.csv
  Epochs: 50
  Learning rate: 0.001
  Sequence length: 30
  Batch size: 32
  Hidden size: 64
  Num layers: 2
  Model save path: models/bitcoin_predictor.pth
--------------------------------------------------
Loading CSV data...
Processing training data...
Converted 361376 data points to DataFrame
Calculated technical indicators
After cleaning: 361356 data points
Processed 361356 data points
Initializing AI model...
Starting training...
Epoch 10/50, Train Loss: 0.124563, Val Loss: 0.128745
Epoch 20/50, Train Loss: 0.089234, Val Loss: 0.092156
Epoch 30/50, Train Loss: 0.076543, Val Loss: 0.078923
Epoch 40/50, Train Loss: 0.071234, Val Loss: 0.073456
Epoch 50/50, Train Loss: 0.068912, Val Loss: 0.070234
Training completed!
Final training loss: 0.068912
Saving model to models/bitcoin_predictor.pth...
Training completed successfully!
```

### Prediction Output
```
Loading model from models/bitcoin_predictor.pth...
Fetching recent 7 days of Bitcoin price data...
Predicting Monte Carlo parameters...
Predicted parameter arrays (first 5 values):
  sigma: [0.0412 0.0413 0.0411 0.0410 0.0409] ... (total 288)
  skewness: [-0.15 -0.14 -0.13 -0.13 -0.12] ... (total 288)
  kurtosis: [3.45 3.44 3.43 3.42 3.41] ... (total 288)
```

### Simulation Output
```
Fetching current Bitcoin price...
Current Bitcoin price: $43,250.75
Using AI-predicted parameters...
Running Monte Carlo simulation with 1000 paths...
Time increment: 300 seconds
Simulation length: 86400 seconds (24.0 hours)

Simulation Results:
  Mean final price: $43,425.32
  Median final price: $43,380.15
  Min final price: $39,876.23
  Max final price: $47,234.89
  Standard deviation: $1,456.78
  5th percentile: $41,023.45
  95th percentile: $45,789.23
  Mean price change: +0.40%
  Probability of gain: 62.3%
  Probability of loss: 37.7%

Results saved to: simulation_results_20250724_143052.csv
```

## File Structure

After training and running simulations, your project will have:

```
├── models/
│   └── bitcoin_predictor.pth          # Trained AI model
├── simulation_results_YYYYMMDD_HHMMSS.csv  # Simulation results
├── train_model.py                     # Main CLI script
├── example_usage.py                   # Usage examples
└── CLI_README.md                      # This documentation
```

## CSV Data Requirements

Your input CSV file must have these columns:
- `timestamp`: Date and time (e.g., "2022-02-03 14:50:00")
- `open`: Opening price
- `high`: Highest price in the period
- `low`: Lowest price in the period
- `close`: Closing price
- `volume`: Trading volume (ignored but must be present)

## Tips for Best Results

1. **Data Quality**: Use at least 30,000 data points for good training results
2. **Training Time**: More epochs generally improve accuracy but take longer
3. **Sequence Length**: 20-50 works well for most datasets
4. **Validation**: Check training loss decreases consistently
5. **Real-time Data**: The model fetches live Bitcoin prices automatically

## Troubleshooting

### Common Issues:

**"Model file not found"**
- Make sure you've trained a model first using the `train` command

**"Insufficient recent data"**
- Check your internet connection
- Try increasing `--days_history` parameter

**"Invalid CSV data format"**
- Verify your CSV has the required columns: timestamp,open,high,low,close,volume
- Check for missing or invalid data points

**Training takes too long**
- Reduce `--epochs` or `--batch_size`
- Use a smaller dataset for testing

**Out of memory errors**
- Reduce `--batch_size`
- Reduce `--hidden_size` or `--num_layers`
- Use a smaller dataset