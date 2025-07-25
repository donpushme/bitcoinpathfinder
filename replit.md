# Bitcoin Monte Carlo AI Predictor

## Overview

This repository contains a real-time Bitcoin price forecasting application that combines AI-powered parameter prediction with Monte Carlo simulations. The system uses an LSTM neural network to predict Monte Carlo simulation parameters based on historical price data, then runs Monte Carlo simulations to generate probabilistic price forecasts.

The application supports both web-based interaction through Streamlit and command-line operation for automated training and simulation workflows.

## User Preferences

Preferred communication style: Simple, everyday language.

## Quick Setup

### Prerequisites
- Python 3.8+
- Internet connection for real-time Bitcoin price data

### Installation
```bash
# Install dependencies (on Replit, already available)
pip install streamlit torch numpy pandas plotly requests scikit-learn scipy
```

### Quick Start - UPGRADED FOR 5-MINUTE INTERVAL PREDICTION
```bash
# Option 1: Web Interface (Now with 288-interval prediction)
streamlit run app.py --server.port 5000

# Option 2: Real-time Monitoring (5-minute intervals)
python realtime_monitor.py --interval 5

# Option 3: Command Line (train + simulate with interval arrays)
python train_model.py all --csv_file your_data.csv

# Option 4: Run setup helper
python quickstart.py
```

### NEW: 5-Minute Interval Architecture
The system now predicts 288 values for each parameter (volatility, skewness, kurtosis) representing 5-minute intervals over 24 hours. This enables much more granular and accurate forecasting.

### Data Format
CSV file with columns: `timestamp,open,high,low,close,volume`
```csv
timestamp,open,high,low,close,volume
2022-02-03 14:50:00,36551.864,36618.271,36500.254,36500.254,0.0
```

**📋 See SETUP.md for complete installation and setup instructions**
**🔄 See REALTIME_README.md for real-time monitoring guide**

## System Architecture

The application follows a modular architecture with clear separation of concerns:

### Frontend Architecture
- **Streamlit Web Interface**: Interactive dashboard for model training, real-time predictions, and visualization
- **Command Line Interface**: Python script (`train_model.py`) for headless training and simulation without UI
- **Session State Management**: Maintains model instances, data processors, and simulation results across user interactions
- **Responsive Layout**: Wide layout configuration optimized for data visualization and controls

### Backend Architecture
- **AI Model Layer**: LSTM-based neural network for predicting Monte Carlo parameters
- **Monte Carlo Engine**: Statistical simulation engine for generating price paths
- **Data Processing Layer**: Handles price data validation, transformation, and feature engineering
- **External API Integration**: Real-time price fetching from multiple cryptocurrency exchanges

## Key Components

### 1. Real-time Monitor (`realtime_monitor.py`) - NEW!
- **Purpose**: Continuous Bitcoin price monitoring with live predictions
- **Features**: Auto-fetches prices every 5 minutes, calculates dynamic volatility, runs Monte Carlo predictions, saves training data
- **Output**: Live console updates, CSV price history, JSON predictions, comprehensive logging
- **Usage**: `python realtime_monitor.py --interval 5` for continuous monitoring

### 2. Command Line Interface (`train_model.py`)
- **Purpose**: Headless training and simulation without web interface
- **Commands**: train, predict, simulate, all (complete workflow)
- **Features**: Flexible parameter configuration, automated result saving, comprehensive logging
- **Usage**: Direct Python execution with command-line arguments

### 3. AI Predictor (`models/ai_predictor.py`) - UPGRADED FOR INTERVAL PREDICTION
- **Purpose**: Enhanced LSTM neural network that predicts 5-minute interval parameters for 24-hour forecasts
- **Architecture**: Deeper 3-layer LSTM (128 hidden units) with batch normalization and dropout
- **Outputs**: 864 total parameters - 288 volatility values, 288 skewness values, 288 kurtosis values (5-minute intervals for 24 hours)
- **Features**: Interval-based parameter generation, advanced time-pattern modeling, enhanced capacity for complex temporal predictions

### 4. Monte Carlo Simulator (`models/monte_carlo.py`) - UPGRADED FOR INTERVAL SIMULATION
- **Purpose**: Generates probabilistic price paths using interval-specific statistical distributions
- **Supported Assets**: Bitcoin (BTC), Ethereum (ETH), Gold (XAU)
- **Features**: Interval-based parameter handling, time-varying volatility/skewness/kurtosis, enhanced distribution modeling
- **Integration**: Supports both legacy single-value parameters and new 288-element interval arrays from AI predictor

### 5. Data Processor (`data/data_processor.py`)
- **Purpose**: Validates and transforms price data for model consumption
- **Input Format**: CSV with columns: timestamp,open,high,low,close,volume (volume column ignored)
- **Features**: Data validation, DataFrame conversion, technical indicator calculation, and CSV parsing

### 6. Price Fetcher (`utils/price_fetcher.py`)
- **Purpose**: Retrieves real-time Bitcoin price data from external APIs
- **Primary Source**: Pyth Network API (free, no API key required)
- **Fallback Sources**: Binance and Coinbase APIs
- **Features**: Error handling, retry logic, and timestamp synchronization

### 7. Visualization (`utils/visualization.py`)
- **Purpose**: Creates interactive charts for simulation results and parameter analysis
- **Technology**: Plotly for interactive web-based visualizations
- **Features**: Monte Carlo path plotting, percentile bands, and parameter trend analysis

## Data Flow - UPGRADED FOR INTERVAL PREDICTION

1. **Training Data Upload**: Users upload CSV files with Bitcoin historical price data (timestamp,open,high,low,close,volume format)
2. **Data Processing**: CSV data is validated, parsed, and transformed with technical indicators for 5-minute interval modeling
3. **AI Model Training**: Enhanced LSTM neural network learns from historical patterns to predict 288 parameter arrays for each forecast
4. **Real-time Data Ingestion**: Price fetcher retrieves current Bitcoin price from external APIs for live predictions
5. **AI Interval Prediction**: Trained model analyzes recent price patterns to predict 864 total parameters (288 × 3 types)
6. **Interval-Based Monte Carlo Simulation**: Statistical engine generates multiple price paths using time-varying parameters for each 5-minute interval
7. **Enhanced Visualization**: Results are rendered as interactive charts showing granular price distributions and confidence intervals
8. **User Interaction**: Streamlit interface allows users to train models, run simulations, and analyze interval-based results

### MAJOR ARCHITECTURAL UPGRADE (December 2024)
- **Before**: Single parameter prediction (4 values: daily sigma, drift, skewness, kurtosis)
- **After**: Interval-based prediction (864 values: 288 sigma + 288 skewness + 288 kurtosis for 5-minute intervals)
- **Impact**: Much more granular and accurate 24-hour Bitcoin price forecasting with time-varying risk parameters

## External Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework for LSTM model implementation
- **Streamlit**: Web application framework for interactive dashboard
- **Plotly**: Interactive visualization library for charts and graphs
- **NumPy/Pandas**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning utilities and preprocessing

### External APIs
- **Pyth Network**: Primary source for real-time cryptocurrency price data
- **Binance API**: Backup price data source
- **Coinbase Pro API**: Additional backup price data source

### Statistical Libraries
- **SciPy**: Statistical distributions for Monte Carlo simulations
- **Requests**: HTTP client for API communication

## Deployment Strategy

### Current Setup
- **Local Development**: Streamlit development server for testing and iteration
- **Session Persistence**: In-memory state management for model instances and data
- **File-based Storage**: Model checkpoints and training data stored locally

### Recommended Production Setup
- **Cloud Hosting**: Deploy to cloud platform supporting Python/Streamlit applications
- **Database Integration**: Add persistent storage for historical data and model artifacts
- **API Rate Limiting**: Implement caching and rate limiting for external API calls
- **Monitoring**: Add logging and error tracking for production reliability

### Scalability Considerations
- **Model Serving**: Separate model inference from web interface for better performance
- **Data Pipeline**: Implement automated data collection and preprocessing
- **Caching Layer**: Add Redis or similar for caching API responses and simulation results
- **Load Balancing**: Support multiple concurrent users with proper resource management

The architecture prioritizes modularity and extensibility, making it easy to add new assets, improve models, or integrate additional data sources. The separation between AI prediction and Monte Carlo simulation allows for independent testing and optimization of each component.