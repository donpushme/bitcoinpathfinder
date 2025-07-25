import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta

def create_simulation_plot(price_paths: np.ndarray, current_price: float, 
                          time_increment: int, time_length: int,
                          show_percentiles: bool = True, max_paths: int = 100) -> go.Figure:
    """
    Create an interactive plot of Monte Carlo simulation results.
    
    Args:
        price_paths: Array of simulated price paths (num_simulations, num_steps)
        current_price: Starting price
        time_increment: Time step in seconds
        time_length: Total simulation time in seconds
        show_percentiles: Whether to show percentile bands
        max_paths: Maximum number of individual paths to display
        
    Returns:
        Plotly figure object
    """
    num_simulations, num_steps = price_paths.shape
    
    # Create time axis
    time_points = np.arange(0, num_steps) * time_increment / 3600  # Convert to hours
    
    # Create figure
    fig = go.Figure()
    
    # Add individual simulation paths (sample to avoid overcrowding)
    if num_simulations > max_paths:
        indices = np.random.choice(num_simulations, max_paths, replace=False)
        sample_paths = price_paths[indices]
    else:
        sample_paths = price_paths
    
    for i, path in enumerate(sample_paths):
        fig.add_trace(go.Scatter(
            x=time_points,
            y=path,
            mode='lines',
            line=dict(width=0.5, color='rgba(100, 100, 100, 0.3)'),
            showlegend=False,
            hovertemplate='Path %{fullData.name}<br>Time: %{x:.1f}h<br>Price: $%{y:,.2f}<extra></extra>',
            name=f'Path {i+1}'
        ))
    
    # Add percentile bands if requested
    if show_percentiles:
        percentiles = [5, 25, 50, 75, 95]
        colors = ['rgba(255, 0, 0, 0.3)', 'rgba(255, 165, 0, 0.3)', 
                 'rgba(0, 128, 0, 0.8)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 0, 0, 0.3)']
        
        for i, (percentile, color) in enumerate(zip(percentiles, colors)):
            percentile_values = np.percentile(price_paths, percentile, axis=0)
            
            line_width = 3 if percentile == 50 else 2
            name = f'{percentile}th Percentile' if percentile != 50 else 'Median'
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=percentile_values,
                mode='lines',
                line=dict(width=line_width, color=color.replace('0.3', '0.8')),
                name=name,
                hovertemplate=f'{name}<br>Time: %{{x:.1f}}h<br>Price: $%{{y:,.2f}}<extra></extra>'
            ))
    
    # Add starting point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[current_price],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Starting Price',
        hovertemplate='Starting Price<br>$%{y:,.2f}<extra></extra>'
    ))
    
    # Calculate final price statistics
    final_prices = price_paths[:, -1]
    mean_final = np.mean(final_prices)
    
    # Add mean final price line
    fig.add_hline(
        y=mean_final,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Mean Final: ${mean_final:,.2f}",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Bitcoin Monte Carlo Simulation<br>'
                 f'<span style="font-size: 12px;">{num_simulations:,} simulations over {time_length/3600:.1f} hours</span>',
            x=0.5
        ),
        xaxis_title='Time (Hours)',
        yaxis_title='Bitcoin Price (USD)',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=600,
        template='plotly_white'
    )
    
    # Format y-axis as currency
    fig.update_yaxis(tickformat='$,.0f')
    
    return fig

def create_parameter_plot(historical_params: List[Dict], predicted_params: Dict) -> go.Figure:
    """
    Create a plot showing historical vs predicted parameters.
    
    Args:
        historical_params: List of historical parameter dictionaries
        predicted_params: Dictionary of predicted parameters
        
    Returns:
        Plotly figure object
    """
    if not historical_params:
        # Create simple bar chart for predicted parameters only
        fig = go.Figure()
        
        params = list(predicted_params.keys())
        values = list(predicted_params.values())
        
        fig.add_trace(go.Bar(
            x=params,
            y=values,
            name='Predicted Parameters',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='AI-Predicted Monte Carlo Parameters',
            xaxis_title='Parameter',
            yaxis_title='Value',
            template='plotly_white'
        )
        
        return fig
    
    # Create subplots for each parameter
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Daily Volatility', 'Daily Drift', 'Skewness', 'Kurtosis'],
        vertical_spacing=0.1
    )
    
    # Convert historical data to DataFrame
    df = pd.DataFrame(historical_params)
    
    param_mapping = {
        'daily_sigma': (1, 1, 'Daily Volatility'),
        'daily_drift': (1, 2, 'Daily Drift'),
        'skewness': (2, 1, 'Skewness'),
        'kurtosis': (2, 2, 'Kurtosis')
    }
    
    for param, (row, col, title) in param_mapping.items():
        if param in df.columns:
            # Historical data
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[param],
                mode='lines',
                name=f'Historical {title}',
                line=dict(color='blue'),
                showlegend=(row == 1 and col == 1)
            ), row=row, col=col)
            
            # Predicted value as horizontal line
            if param in predicted_params:
                fig.add_hline(
                    y=predicted_params[param],
                    line_dash="dash",
                    line_color="red",
                    row=row, col=col,
                    annotation_text=f"Predicted: {predicted_params[param]:.4f}",
                    annotation_position="top right"
                )
    
    fig.update_layout(
        title='Historical vs Predicted Parameters',
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_price_distribution_plot(final_prices: np.ndarray, current_price: float) -> go.Figure:
    """
    Create a histogram of final prices from Monte Carlo simulation.
    
    Args:
        final_prices: Array of final prices from simulation
        current_price: Starting price for reference
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Create histogram
    fig.add_trace(go.Histogram(
        x=final_prices,
        nbinsx=50,
        name='Final Price Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Add vertical lines for key statistics
    mean_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    
    fig.add_vline(
        x=current_price,
        line_dash="solid",
        line_color="green",
        annotation_text=f"Starting: ${current_price:,.2f}",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=mean_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ${mean_price:,.2f}",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=median_price,
        line_dash="dot",
        line_color="blue",
        annotation_text=f"Median: ${median_price:,.2f}",
        annotation_position="top"
    )
    
    # Update layout
    fig.update_layout(
        title='Distribution of Final Prices',
        xaxis_title='Final Price (USD)',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400
    )
    
    # Format x-axis as currency
    fig.update_xaxis(tickformat='$,.0f')
    
    return fig

def create_training_loss_plot(train_losses: List[float], val_losses: Optional[List[float]] = None) -> go.Figure:
    """
    Create a plot of training and validation losses.
    
    Args:
        train_losses: List of training losses
        val_losses: Optional list of validation losses
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Training loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses,
        mode='lines',
        name='Training Loss',
        line=dict(color='blue')
    ))
    
    # Validation loss if available
    if val_losses and len(val_losses) == len(train_losses):
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_losses,
            mode='lines',
            name='Validation Loss',
            line=dict(color='red')
        ))
    
    fig.update_layout(
        title='Training Progress',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_correlation_heatmap(data: pd.DataFrame, features: List[str]) -> go.Figure:
    """
    Create a correlation heatmap of features.
    
    Args:
        data: DataFrame containing features
        features: List of feature names to include
        
    Returns:
        Plotly figure object
    """
    # Calculate correlation matrix
    available_features = [f for f in features if f in data.columns]
    corr_matrix = data[available_features].corr(numeric_only=True)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=500,
        template='plotly_white'
    )
    
    return fig
