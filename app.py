import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

from models.ai_predictor import BitcoinParameterPredictor
from models.monte_carlo import simulate_crypto_price_paths
from utils.price_fetcher import BitcoinPriceFetcher
from utils.visualization import create_simulation_plot, create_parameter_plot, create_multistep_parameter_plot
from data.data_processor import DataProcessor

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'price_fetcher' not in st.session_state:
    st.session_state.price_fetcher = BitcoinPriceFetcher()
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'last_simulation' not in st.session_state:
    st.session_state.last_simulation = None

st.set_page_config(
    page_title="Bitcoin Monte Carlo AI Predictor",
    page_icon="₿",
    layout="wide"
)

st.title("₿ Bitcoin Monte Carlo AI Predictor")
st.markdown("Real-time Bitcoin price forecasting using AI-powered Monte Carlo simulations")

# Sidebar for controls
st.sidebar.header("Configuration")

# Model section
st.sidebar.subheader("🤖 AI Model")
model_action = st.sidebar.radio(
    "Model Action:",
    ["Load Existing", "Train New", "Real-time Prediction"]
)

if model_action == "Train New":
    st.sidebar.subheader("Training Data")
    
    # File upload for training data
    uploaded_file = st.sidebar.file_uploader(
        "Upload Bitcoin price data (CSV)",
        type=['csv'],
        help="Upload historical Bitcoin price data in CSV format with columns: timestamp,open,high,low,close,volume"
    )
    
    if uploaded_file is not None:
        try:
            csv_data = uploaded_file.read().decode('utf-8')
            st.session_state.training_data = csv_data
            # Count data points by counting lines minus header
            data_points = len(csv_data.strip().split('\n')) - 1
            st.sidebar.success(f"Loaded {data_points} data points")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}")
    
    # Training parameters
    epochs = st.sidebar.slider("Training Epochs", 10, 200, 50)
    sequence_length = st.sidebar.slider("Sequence Length", 10, 100, 30)
    learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    
    if st.sidebar.button("Start Training") and st.session_state.training_data:
        with st.spinner("Training AI model..."):
            try:
                # Initialize model
                st.session_state.model = BitcoinParameterPredictor(
                    input_size=4,  # OHLC data
                    hidden_size=64,
                    num_layers=2,
                    sequence_length=sequence_length
                )
                
                # Process training data
                processed_data = st.session_state.data_processor.process_training_data(
                    st.session_state.training_data,
                    sequence_length=sequence_length
                )
                
                # Train model
                train_losses = st.session_state.model.train_model(
                    processed_data,
                    epochs=epochs,
                    learning_rate=learning_rate
                )
                
                # Save model
                model_path = "models/bitcoin_predictor.pth"
                os.makedirs("models", exist_ok=True)
                st.session_state.model.save_model(model_path)
                
                st.sidebar.success("Model trained and saved successfully!")
                
                # Display training progress
                fig = px.line(y=train_losses, title="Training Loss")
                fig.update_xaxis(title="Epoch")
                fig.update_yaxis(title="Loss")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.sidebar.error(f"Training failed: {e}")

elif model_action == "Load Existing":
    model_path = "models/bitcoin_predictor.pth"
    if st.sidebar.button("Load Model"):
        if os.path.exists(model_path):
            try:
                st.session_state.model = BitcoinParameterPredictor.load_model(model_path)
                st.sidebar.success("Model loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading model: {e}")
        else:
            st.sidebar.error("No trained model found. Please train a new model first.")

# Simulation parameters
st.sidebar.subheader("🎲 Simulation Parameters")
num_simulations = st.sidebar.slider("Number of Simulations", 100, 2000, 1000)
time_increment = st.sidebar.selectbox(
    "Time Increment",
    [300, 600, 900, 1800, 3600],  # 5min, 10min, 15min, 30min, 1hr
    index=0,
    format_func=lambda x: f"{x//60} minutes" if x < 3600 else f"{x//3600} hour(s)"
)
time_length = st.sidebar.selectbox(
    "Simulation Length",
    [3600, 7200, 14400, 28800, 86400],  # 1hr, 2hr, 4hr, 8hr, 24hr
    index=4,
    format_func=lambda x: f"{x//3600} hour(s)"
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Real-time Analysis")
    
    # Current Bitcoin price
    if st.button("Fetch Current Bitcoin Price"):
        with st.spinner("Fetching current Bitcoin price..."):
            price, timestamp = st.session_state.price_fetcher.fetch_latest_bitcoin_price()
            if price and timestamp:
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                st.success(f"Current Bitcoin Price: ${price:,.2f}")
                st.info(f"Last updated: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                st.session_state.current_price = price
                st.session_state.current_timestamp = timestamp
            else:
                st.error("Failed to fetch Bitcoin price")
    
    # Real-time prediction and simulation
    if st.button("Run AI-Powered Monte Carlo Simulation") and st.session_state.model:
        if hasattr(st.session_state, 'current_price'):
            with st.spinner("Running AI prediction and Monte Carlo simulation..."):
                try:
                    # Get recent price data for prediction
                    recent_data = st.session_state.price_fetcher.get_recent_data(days=7)
                    if recent_data:
                        # Predict parameters using AI model
                        predicted_params = st.session_state.model.predict_parameters(recent_data)
                        
                        # Run Monte Carlo simulation with AI-predicted parameters
                        price_paths = simulate_crypto_price_paths(
                            current_price=st.session_state.current_price,
                            time_increment=time_increment,
                            time_length=time_length,
                            num_simulations=num_simulations,
                            asset='BTC',
                            ai_params=predicted_params
                        )
                        
                        st.session_state.last_simulation = {
                            'price_paths': price_paths,
                            'predicted_params': predicted_params,
                            'current_price': st.session_state.current_price,
                            'time_increment': time_increment,
                            'time_length': time_length,
                            'num_simulations': num_simulations
                        }
                        
                        # Create visualization
                        fig = create_simulation_plot(
                            price_paths,
                            st.session_state.current_price,
                            time_increment,
                            time_length
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display statistics
                        st.subheader("📈 Simulation Statistics")
                        final_prices = price_paths[:, -1]
                        
                        stats_col1, stats_col2, stats_col3 = st.columns(3)
                        with stats_col1:
                            st.metric("Mean Final Price", f"${np.mean(final_prices):,.2f}")
                            st.metric("Median Final Price", f"${np.median(final_prices):,.2f}")
                        with stats_col2:
                            st.metric("Min Final Price", f"${np.min(final_prices):,.2f}")
                            st.metric("Max Final Price", f"${np.max(final_prices):,.2f}")
                        with stats_col3:
                            std_dev = np.std(final_prices)
                            st.metric("Standard Deviation", f"${std_dev:,.2f}")
                            st.metric("Coefficient of Variation", f"{std_dev/np.mean(final_prices):.2%}")
                        
                    else:
                        st.error("Failed to fetch recent data for prediction")
                        
                except Exception as e:
                    st.error(f"Simulation failed: {e}")
        else:
            st.warning("Please fetch current Bitcoin price first")
    
    # Display predicted parameters if available
    if st.session_state.last_simulation and 'predicted_params' in st.session_state.last_simulation:
        st.subheader("🔮 AI-Predicted Parameters (24h, 5-min steps)")
        params = st.session_state.last_simulation['predicted_params']
        # Show summary stats for first, median, last step
        st.write(f"**First Sigma:** {params['sigma'][0]:.4f} | **Median Sigma:** {np.median(params['sigma']):.4f} | **Last Sigma:** {params['sigma'][-1]:.4f}")
        st.write(f"**First Skewness:** {params['skewness'][0]:.3f} | **Median Skewness:** {np.median(params['skewness']):.3f} | **Last Skewness:** {params['skewness'][-1]:.3f}")
        st.write(f"**First Kurtosis:** {params['kurtosis'][0]:.3f} | **Median Kurtosis:** {np.median(params['kurtosis']):.3f} | **Last Kurtosis:** {params['kurtosis'][-1]:.3f}")
        # Plot the full arrays
        fig_params = create_multistep_parameter_plot(params, time_increment=st.session_state.last_simulation['time_increment'])
        st.plotly_chart(fig_params, use_container_width=True)

with col2:
    st.header("🎯 Model Status")
    
    if st.session_state.model:
        st.success("✅ AI Model Loaded")
        
        # Model information
        st.subheader("Model Info")
        model_info = st.session_state.model.get_model_info()
        for key, value in model_info.items():
            st.write(f"**{key}:** {value}")
    else:
        st.warning("⚠️ No AI Model Loaded")
        st.info("Please train a new model or load an existing one from the sidebar.")
    
    # Price history chart
    st.subheader("📊 Recent Price History")
    if st.button("Show Price History"):
        with st.spinner("Fetching price history..."):
            history_data = st.session_state.price_fetcher.get_recent_data(days=7)
            if history_data and len(history_data) > 0:
                df = pd.DataFrame(history_data)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df['datetime'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Bitcoin Price'
                ))
                
                fig.update_layout(
                    title="Bitcoin Price History (7 Days)",
                    xaxis_title="Time",
                    yaxis_title="Price (USD)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to fetch price history")

# Footer
st.markdown("---")
st.markdown(
    "**Disclaimer:** This tool is for educational and research purposes only. "
    "Cryptocurrency investments carry significant risk. Always do your own research "
    "and consult with financial advisors before making investment decisions."
)
