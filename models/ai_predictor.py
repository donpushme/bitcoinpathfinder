import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pickle
import os

class BitcoinParameterPredictor(nn.Module):
    """
    LSTM-based neural network for predicting Bitcoin Monte Carlo simulation parameters.
    Predicts: 288 volatility values (5-min intervals for 24h), 288 skewness values, 288 kurtosis values
    Total output: 864 parameters (288 * 3)
    """
    
    def __init__(self, input_size: int = 4, hidden_size: int = 128, num_layers: int = 3, 
                 sequence_length: int = 30, intervals_per_day: int = 288):
        super(BitcoinParameterPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.intervals_per_day = intervals_per_day
        self.output_size = intervals_per_day * 3  # 288 sigma + 288 skewness + 288 kurtosis
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected layers for time-series prediction
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, self.output_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
        # Parameter bounds for 5-minute intervals
        self.param_bounds = {
            'interval_sigma': (0.0001, 0.05),    # 0.01% to 5% per 5-minute interval  
            'skewness': (-2.0, 2.0),             # Skewness range
            'kurtosis': (0.1, 10.0)              # Kurtosis range
        }
        
        # Scalers for input and output normalization
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.fitted = False
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers with batch normalization
        out = self.relu(self.batch_norm1(self.fc1(lstm_out)))
        out = self.dropout(out)
        out = self.relu(self.batch_norm2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        
        # Apply parameter bounds using activation functions
        out = self._apply_parameter_bounds(out)
        
        return out
    
    def _apply_parameter_bounds(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter-specific bounds using activation functions"""
        # Split the output into sigma, skewness, and kurtosis arrays
        n_intervals = self.intervals_per_day
        
        # Sigma values (288 values): sigmoid scaled to (0.0001, 0.05)
        sigma_raw = x[:, :n_intervals]
        sigma = torch.sigmoid(sigma_raw) * 0.0499 + 0.0001
        
        # Skewness values (288 values): tanh scaled to (-2.0, 2.0)
        skewness_raw = x[:, n_intervals:2*n_intervals]
        skewness = torch.tanh(skewness_raw) * 2.0
        
        # Kurtosis values (288 values): sigmoid scaled to (0.1, 10.0)
        kurtosis_raw = x[:, 2*n_intervals:3*n_intervals]
        kurtosis = torch.sigmoid(kurtosis_raw) * 9.9 + 0.1
        
        return torch.cat([sigma, skewness, kurtosis], dim=1)
    
    def prepare_sequences(self, data: List[Dict], sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training sequences from price data.
        
        Args:
            data: List of price dictionaries with OHLC data
            sequence_length: Length of input sequences
            
        Returns:
            X: Input sequences (N, sequence_length, 4)
            y: Target parameters (N, 864) - 288 sigma + 288 skewness + 288 kurtosis
        """
        if len(data) < sequence_length + 1:
            raise ValueError(f"Not enough data points. Need at least {sequence_length + 1}, got {len(data)}")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Calculate features from OHLC data
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=5, min_periods=1).std()
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['open_close_range'] = abs(df['open'] - df['close']) / df['close']
        
        # Prepare input features (OHLC normalized)
        features = ['open', 'high', 'low', 'close']
        X_data = df[features].values
        
        # Calculate target parameters for 5-minute intervals (288 intervals = 24 hours)
        targets = []
        for i in range(sequence_length, len(df)):
            # Calculate 5-minute volatility, skewness, and kurtosis for next 288 intervals
            window_data = df.iloc[i-sequence_length:i]
            
            # Create 288 target values for each parameter
            target_sigma = self._calculate_interval_volatilities(window_data, 288)
            target_skewness = self._calculate_interval_skewness(window_data, 288)
            target_kurtosis = self._calculate_interval_kurtosis(window_data, 288)
            
            # Combine all targets: [sigma_288, skewness_288, kurtosis_288]
            combined_targets = np.concatenate([target_sigma, target_skewness, target_kurtosis])
            targets.append(combined_targets)
        
        # Create sequences
        X_sequences = []
        for i in range(len(targets)):
            sequence = X_data[i:i+sequence_length]
            X_sequences.append(sequence)
        
        X = np.array(X_sequences)
        y = np.array(targets)
        
        return X, y
    
    def _calculate_interval_volatilities(self, window_data: pd.DataFrame, n_intervals: int) -> np.ndarray:
        """Calculate volatility forecasts for n future 5-minute intervals"""
        returns = window_data['returns'].dropna()
        
        if len(returns) < 5:
            # Default volatility pattern: higher during market hours, lower at night
            base_vol = 0.01  # 1% per 5-minute interval
            pattern = np.sin(np.linspace(0, 4*np.pi, n_intervals)) * 0.005 + base_vol
            return np.clip(pattern, 0.0001, 0.05)
        
        # Calculate rolling volatility from recent data
        recent_vol = returns.rolling(window=5, min_periods=1).std().iloc[-1]
        if np.isnan(recent_vol):
            recent_vol = 0.01
            
        # Create volatility pattern with some variation
        base_vol = np.clip(recent_vol, 0.001, 0.05)
        
        # Add time-of-day pattern (higher volatility during market hours)
        time_pattern = np.sin(np.linspace(0, 4*np.pi, n_intervals)) * 0.3 + 1.0
        volatilities = base_vol * time_pattern
        
        # Add some random variation
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(1.0, 0.1, n_intervals)
        volatilities = volatilities * noise
        
        return np.clip(volatilities, 0.0001, 0.05)
    
    def _calculate_interval_skewness(self, window_data: pd.DataFrame, n_intervals: int) -> np.ndarray:
        """Calculate skewness forecasts for n future 5-minute intervals"""
        returns = window_data['returns'].dropna()
        
        if len(returns) < 10:
            # Default skewness pattern with some variation
            base_skew = -0.1  # Slightly negative (crash risk)
            pattern = np.sin(np.linspace(0, 2*np.pi, n_intervals)) * 0.5 + base_skew
            return np.clip(pattern, -2.0, 2.0)
        
        # Calculate rolling skewness
        recent_skew = returns.rolling(window=10, min_periods=5).skew().iloc[-1]
        if np.isnan(recent_skew):
            recent_skew = -0.1
            
        # Create skewness pattern
        base_skew = np.clip(recent_skew, -2.0, 2.0)
        
        # Add variation around the base skewness
        np.random.seed(43)  # For reproducibility  
        variations = np.random.normal(base_skew, 0.3, n_intervals)
        
        return np.clip(variations, -2.0, 2.0)
    
    def _calculate_interval_kurtosis(self, window_data: pd.DataFrame, n_intervals: int) -> np.ndarray:
        """Calculate kurtosis forecasts for n future 5-minute intervals"""
        returns = window_data['returns'].dropna()
        
        if len(returns) < 10:
            # Default kurtosis pattern (fat tails)
            base_kurt = 4.0  # Higher than normal (3.0)
            pattern = np.sin(np.linspace(0, 3*np.pi, n_intervals)) * 2.0 + base_kurt
            return np.clip(pattern, 0.1, 10.0)
        
        # Calculate rolling kurtosis
        recent_kurt = returns.rolling(window=10, min_periods=5).kurt().iloc[-1] + 3.0
        if np.isnan(recent_kurt):
            recent_kurt = 4.0
            
        # Create kurtosis pattern
        base_kurt = np.clip(recent_kurt, 1.0, 10.0)
        
        # Add variation around the base kurtosis
        np.random.seed(44)  # For reproducibility
        variations = np.random.normal(base_kurt, 1.0, n_intervals)
        
        return np.clip(variations, 0.1, 10.0)
    
    def train_model(self, data: List[Dict], epochs: int = 50, learning_rate: float = 0.001, 
                   batch_size: int = 32, validation_split: float = 0.2) -> List[float]:
        """
        Train the model on historical Bitcoin price data.
        
        Args:
            data: List of price dictionaries
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            
        Returns:
            List of training losses
        """
        print(f"Preparing training data from {len(data)} price points...")
        
        # Prepare sequences
        X, y = self.prepare_sequences(data, self.sequence_length)
        print(f"Created {len(X)} training sequences")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, shuffle=False
        )
        
        # Fit and transform data
        X_train_scaled = self._fit_transform_input(X_train)
        X_val_scaled = self._transform_input(X_val)
        y_train_scaled = self._fit_transform_output(y_train)
        y_val_scaled = self._transform_output(y_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        train_losses = []
        val_losses = []
        
        print("Starting training...")
        self.train()
        
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            num_batches = 0
            
            # Training phase
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            self.train()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.fitted = True
        print("Training completed!")
        
        return train_losses
    
    def predict_interval_parameters(self, recent_data: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Predict parameters for 288 5-minute intervals (24 hours)
        
        Args:
            recent_data: Recent price data for prediction
            
        Returns:
            Dictionary with arrays of sigma, skewness, kurtosis (each 288 elements)
        """
        if not self.fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare input sequence
        df = pd.DataFrame(recent_data)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=5, min_periods=1).std()
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['open_close_range'] = abs(df['open'] - df['close']) / df['close']
        
        # Get the last sequence for prediction
        features = ['open', 'high', 'low', 'close']
        if len(df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
        
        X = df[features].values[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        X_scaled = self._transform_input(X)
        
        # Convert to tensor and predict
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.eval()
        with torch.no_grad():
            predictions = self.forward(X_tensor)
            predictions_np = predictions.cpu().numpy()[0]  # Remove batch dimension
        
        # Split predictions into sigma, skewness, and kurtosis arrays
        n_intervals = self.intervals_per_day
        sigma_array = predictions_np[:n_intervals]
        skewness_array = predictions_np[n_intervals:2*n_intervals]  
        kurtosis_array = predictions_np[2*n_intervals:3*n_intervals]
        
        return {
            'sigma': sigma_array,
            'skewness': skewness_array, 
            'kurtosis': kurtosis_array
        }
    
    def _fit_transform_input(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform input data"""
        # Reshape for scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.input_scaler.fit_transform(X_reshaped)
        return X_scaled.reshape(original_shape)
    
    def _transform_input(self, X: np.ndarray) -> np.ndarray:
        """Transform input data using fitted scaler"""
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.input_scaler.transform(X_reshaped)
        return X_scaled.reshape(original_shape)
    
    def _fit_transform_output(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform output data"""
        return self.output_scaler.fit_transform(y)
    
    def _transform_output(self, y: np.ndarray) -> np.ndarray:
        """Transform output data using fitted scaler"""
        return self.output_scaler.transform(y)
    
    def _inverse_transform_output(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform output data"""
        return self.output_scaler.inverse_transform(y)
    
    def predict_parameters(self, recent_data: List[Dict]) -> Dict[str, float]:
        """
        Predict parameters for Monte Carlo simulation from recent price data.
        
        Args:
            recent_data: List of recent price dictionaries
            
        Returns:
            Dictionary with predicted parameters
        """
        if not self.fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if len(recent_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
        
        # Take the most recent sequence
        recent_sequence = recent_data[-self.sequence_length:]
        
        # Convert to array format
        df = pd.DataFrame(recent_sequence)
        features = ['open', 'high', 'low', 'close']
        X_data = df[features].values
        
        # Normalize and reshape
        X_reshaped = X_data.reshape(1, self.sequence_length, -1)
        X_scaled = self._transform_input(X_reshaped)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make prediction
        self.eval()
        with torch.no_grad():
            prediction = self.forward(X_tensor)
            prediction_np = prediction.cpu().numpy()
        
        # Convert back to parameter dictionary
        params = {
            'daily_sigma': float(prediction_np[0, 0]),
            'daily_drift': float(prediction_np[0, 1]),
            'skewness': float(prediction_np[0, 2]),
            'kurtosis': float(prediction_np[0, 3])
        }
        
        return params
    
    def save_model(self, filepath: str):
        """Save model and scalers"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler,
            'model_params': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'sequence_length': self.sequence_length,
                'output_size': self.output_size
            },
            'fitted': self.fitted
        }
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BitcoinParameterPredictor':
        """Load model and scalers"""
        try:
            # Try loading with weights_only=False for backward compatibility
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception as e:
            # If that fails, try with safe globals for sklearn objects
            import torch.serialization
            torch.serialization.add_safe_globals(['sklearn.preprocessing._data.StandardScaler'])
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
        
        # Handle different model parameter formats
        model_params = checkpoint.get('model_params', {})
        
        # Ensure all required parameters are present with defaults
        required_params = {
            'input_size': 4,
            'hidden_size': 128,
            'num_layers': 3,
            'sequence_length': 30,
            'intervals_per_day': 288
        }
        
        # Update with saved params, keeping defaults for missing ones
        for key, default_value in required_params.items():
            if key not in model_params:
                model_params[key] = default_value
                print(f"Warning: Missing parameter '{key}', using default: {default_value}")
        
        # Create model instance
        model = cls(**model_params)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.input_scaler = checkpoint['input_scaler']
        model.output_scaler = checkpoint['output_scaler']
        model.fitted = checkpoint['fitted']
        
        return model
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information"""
        return {
            'Input Size': str(self.input_size),
            'Hidden Size': str(self.hidden_size),
            'Num Layers': str(self.num_layers),
            'Sequence Length': str(self.sequence_length),
            'Output Size': str(self.output_size),
            'Device': str(self.device),
            'Fitted': str(self.fitted),
            'Total Parameters': str(sum(p.numel() for p in self.parameters()))
        }
