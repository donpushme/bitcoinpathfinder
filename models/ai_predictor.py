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
    Predicts: 288-step arrays for sigma (volatility), skewness, kurtosis
    """
    
    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, 
                 sequence_length: int = 30, output_steps: int = 288, output_size: int = 3):
        super(BitcoinParameterPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.output_steps = output_steps
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected layers for each time step
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
        # Parameter bounds and scalers
        self.param_bounds = {
            'sigma': (0.001, 0.15),    # 0.1% to 15% daily volatility
            'skewness': (-2.0, 2.0),   # Skewness range
            'kurtosis': (0.1, 10.0)    # Kurtosis range
        }
        
        # Scalers for input and output normalization
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.fitted = False
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network (multi-step output)"""
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # Use the last hidden state as the context for all output steps
        last_hidden = lstm_out[:, -1, :].unsqueeze(1).repeat(1, self.output_steps, 1)
        # Optionally, you could use a sequence decoder for more complex tasks
        out = self.relu(self.fc1(last_hidden))
        out = self.dropout(out)
        out = self.fc2(out)  # (batch, output_steps, output_size)
        out = self._apply_parameter_bounds(out)
        return out  # (batch, 288, 3)

    def _apply_parameter_bounds(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter-specific bounds using activation functions for each step"""
        # x shape: (batch, output_steps, output_size)
        sigma = torch.sigmoid(x[..., 0]) * 0.149 + 0.001  # (batch, output_steps)
        skewness = torch.tanh(x[..., 1]) * 2.0            # (batch, output_steps)
        kurtosis = torch.sigmoid(x[..., 2]) * 9.9 + 0.1   # (batch, output_steps)
        return torch.stack([sigma, skewness, kurtosis], dim=-1)  # (batch, output_steps, 3)

    def prepare_sequences(self, data: List[Dict], sequence_length: int, output_steps: int = 288) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training sequences from price data for multi-step prediction.
        Args:
            data: List of price dictionaries with OHLC data
            sequence_length: Length of input sequences
            output_steps: Number of steps to predict (default 288)
        Returns:
            X: Input sequences (N, sequence_length, 4)
            y: Target parameter arrays (N, output_steps, 3)
        """
        if len(data) < sequence_length + output_steps:
            raise ValueError(f"Not enough data points. Need at least {sequence_length + output_steps}, got {len(data)}")
        df = pd.DataFrame(data)
        df['returns'] = df['close'].pct_change()
        X_data = df[['open', 'high', 'low', 'close']].values
        X_sequences = []
        y_targets = []
        for i in range(sequence_length, len(df) - output_steps + 1):
            # Input sequence
            X_seq = X_data[i-sequence_length:i]
            X_sequences.append(X_seq)
            # Output: for each of the next output_steps, calculate sigma, skewness, kurtosis
            y_seq = []
            for j in range(output_steps):
                window = df.iloc[i+j-sequence_length+1:i+j+1]['returns'].dropna()
                if len(window) > 5:
                    sigma = np.clip(window.std(), 0.001, 0.15)
                    skew = np.clip(window.skew() if not np.isnan(window.skew()) else 0.0, -2.0, 2.0)
                    kurt = np.clip(window.kurtosis() + 3, 0.1, 10.0)
                else:
                    sigma, skew, kurt = 0.0366, 0.09, 1.34
                y_seq.append([sigma, skew, kurt])
            y_targets.append(y_seq)
        X = np.array(X_sequences)
        y = np.array(y_targets)
        return X, y

    def train_model(self, data: List[Dict], epochs: int = 50, learning_rate: float = 0.001, 
                   batch_size: int = 32, validation_split: float = 0.2) -> List[float]:
        print(f"Preparing training data from {len(data)} price points...")
        X, y = self.prepare_sequences(data, self.sequence_length, self.output_steps)
        print(f"Created {len(X)} training sequences")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, shuffle=False
        )
        X_train_scaled = self._fit_transform_input(X_train)
        X_val_scaled = self._transform_input(X_val)
        y_train_scaled = self._fit_transform_output(y_train)
        y_val_scaled = self._transform_output(y_val)
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).to(self.device)
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
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item()
                num_batches += 1
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
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
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.input_scaler.fit_transform(X_reshaped)
        return X_scaled.reshape(original_shape)

    def _transform_input(self, X: np.ndarray) -> np.ndarray:
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.input_scaler.transform(X_reshaped)
        return X_scaled.reshape(original_shape)

    def _fit_transform_output(self, y: np.ndarray) -> np.ndarray:
        original_shape = y.shape
        y_reshaped = y.reshape(-1, y.shape[-1])
        y_scaled = self.output_scaler.fit_transform(y_reshaped)
        return y_scaled.reshape(original_shape)

    def _transform_output(self, y: np.ndarray) -> np.ndarray:
        original_shape = y.shape
        y_reshaped = y.reshape(-1, y.shape[-1])
        y_scaled = self.output_scaler.transform(y_reshaped)
        return y_scaled.reshape(original_shape)

    def _inverse_transform_output(self, y: np.ndarray) -> np.ndarray:
        original_shape = y.shape
        y_reshaped = y.reshape(-1, y.shape[-1])
        y_inv = self.output_scaler.inverse_transform(y_reshaped)
        return y_inv.reshape(original_shape)

    def predict_parameters(self, recent_data: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Predict 288-step arrays for sigma, skewness, kurtosis from recent price data.
        Args:
            recent_data: List of recent price dictionaries
        Returns:
            Dict with arrays for 'sigma', 'skewness', 'kurtosis' (each shape: (288,))
        """
        if not self.fitted:
            raise ValueError("Model must be trained before making predictions")
        if len(recent_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
        recent_sequence = recent_data[-self.sequence_length:]
        df = pd.DataFrame(recent_sequence)
        features = ['open', 'high', 'low', 'close']
        X_data = df[features].values
        X_reshaped = X_data.reshape(1, self.sequence_length, -1)
        X_scaled = self._transform_input(X_reshaped)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        self.eval()
        with torch.no_grad():
            prediction = self.forward(X_tensor)
            prediction_np = prediction.cpu().numpy()[0]  # shape: (288, 3)
        # Return as dict of arrays
        return {
            'sigma': prediction_np[:, 0],
            'skewness': prediction_np[:, 1],
            'kurtosis': prediction_np[:, 2]
        }

    def save_model(self, filepath: str):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler,
            'model_params': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'sequence_length': self.sequence_length,
                'output_steps': self.output_steps,
                'output_size': self.output_size
            },
            'fitted': self.fitted
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> 'BitcoinParameterPredictor':
        checkpoint = torch.load(filepath, map_location='cpu')
        model = cls(**checkpoint['model_params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.input_scaler = checkpoint['input_scaler']
        model.output_scaler = checkpoint['output_scaler']
        model.fitted = checkpoint['fitted']
        return model

    def get_model_info(self) -> Dict[str, str]:
        return {
            'Input Size': str(self.input_size),
            'Hidden Size': str(self.hidden_size),
            'Num Layers': str(self.num_layers),
            'Sequence Length': str(self.sequence_length),
            'Output Steps': str(self.output_steps),
            'Output Size': str(self.output_size),
            'Device': str(self.device),
            'Fitted': str(self.fitted),
            'Total Parameters': str(sum(p.numel() for p in self.parameters()))
        }
