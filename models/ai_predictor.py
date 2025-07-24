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
    Predicts: daily_sigma, daily_drift, skewness, kurtosis
    """
    
    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, 
                 sequence_length: int = 30, output_size: int = 4):
        super(BitcoinParameterPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
        # Parameter bounds and scalers
        self.param_bounds = {
            'daily_sigma': (0.001, 0.15),    # 0.1% to 15% daily volatility
            'daily_drift': (-0.1, 0.1),     # -10% to +10% daily drift
            'skewness': (-2.0, 2.0),         # Skewness range
            'kurtosis': (0.1, 10.0)          # Kurtosis range
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
        
        # Fully connected layers
        out = self.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Apply parameter bounds using sigmoid/tanh activation
        out = self._apply_parameter_bounds(out)
        
        return out
    
    def _apply_parameter_bounds(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter-specific bounds using activation functions"""
        # daily_sigma: sigmoid scaled to (0.001, 0.15)
        daily_sigma = torch.sigmoid(x[:, 0:1]) * 0.149 + 0.001
        
        # daily_drift: tanh scaled to (-0.1, 0.1)
        daily_drift = torch.tanh(x[:, 1:2]) * 0.1
        
        # skewness: tanh scaled to (-2.0, 2.0)
        skewness = torch.tanh(x[:, 2:3]) * 2.0
        
        # kurtosis: sigmoid scaled to (0.1, 10.0)
        kurtosis = torch.sigmoid(x[:, 3:4]) * 9.9 + 0.1
        
        return torch.cat([daily_sigma, daily_drift, skewness, kurtosis], dim=1)
    
    def prepare_sequences(self, data: List[Dict], sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training sequences from price data.
        
        Args:
            data: List of price dictionaries with OHLC data
            sequence_length: Length of input sequences
            
        Returns:
            X: Input sequences (N, sequence_length, 4)
            y: Target parameters (N, 4)
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
        
        # Calculate target parameters using rolling windows
        targets = []
        for i in range(sequence_length, len(df)):
            window_data = df.iloc[i-sequence_length:i]
            
            # Calculate parameters from the window
            returns = window_data['returns'].dropna()
            if len(returns) > 5:  # Need minimum data for stable estimates
                daily_sigma = returns.std()
                daily_drift = returns.mean()
                skewness = returns.skew() if not np.isnan(returns.skew()) else 0.0
                kurtosis = returns.kurtosis() + 3  # Convert to absolute kurtosis
                
                # Bound the parameters
                daily_sigma = np.clip(daily_sigma, 0.001, 0.15)
                daily_drift = np.clip(daily_drift, -0.1, 0.1)
                skewness = np.clip(skewness, -2.0, 2.0)
                kurtosis = np.clip(kurtosis, 0.1, 10.0)
                
                targets.append([daily_sigma, daily_drift, skewness, kurtosis])
            else:
                # Use default values if insufficient data
                targets.append([0.0366, 0.0, 0.09, 1.34])
        
        # Create sequences
        X_sequences = []
        for i in range(len(targets)):
            sequence = X_data[i:i+sequence_length]
            X_sequences.append(sequence)
        
        X = np.array(X_sequences)
        y = np.array(targets)
        
        return X, y
    
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
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create model instance
        model = cls(**checkpoint['model_params'])
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
