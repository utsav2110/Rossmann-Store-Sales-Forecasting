"""
Custom N-HiTS layers for Keras - Implements the actual N-HiTS architecture.

This matches the training implementation from the Colab notebook.
"""

from tensorflow import keras
from tensorflow.keras import layers


class NHiTSBlock(layers.Layer):
    """N-HiTS block: MaxPool → Dense stack → backcast + forecast.
    
    This is the exact architecture used in the Colab training notebook.
    """
    
    def __init__(self, input_size, forecast_size, pool_size, n_layers=2, units=256, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.pool_size = pool_size
        self.n_layers = n_layers
        self.units = units
        
        # MaxPooling layer
        self.pool = layers.MaxPooling1D(pool_size=pool_size, strides=1, padding='same')
        
        # MLP stack
        mlp = []
        for _ in range(n_layers):
            mlp += [
                layers.Dense(units, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.1)
            ]
        self.mlp = keras.Sequential(mlp)
        
        # Output layers
        self.backcast_l = layers.Dense(input_size)
        self.forecast_l = layers.Dense(forecast_size)
        
        # Track if built
        self._is_built = False
    
    def build(self, input_shape):
        """Build the layer layers."""
        if not self._is_built:
            # Sublayers are already created in __init__, just mark as built
            super().build(input_shape)
            self._is_built = True
    
    def call(self, x, training=False):
        """Forward pass returns (backcast, forecast) tuple."""
        pooled    = self.pool(x)
        flat      = layers.Flatten()(pooled)
        h         = self.mlp(flat, training=training)
        backcast  = self.backcast_l(h)
        forecast  = self.forecast_l(h)
        return backcast, forecast
    
    def get_config(self):
        """Serialize layer configuration for model saving."""
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'forecast_size': self.forecast_size,
            'pool_size': self.pool_size,
            'n_layers': self.n_layers,
            'units': self.units,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Deserialize layer from config."""
        return cls(**config)


class NHiTS(layers.Layer):
    """N-HiTS full architecture with stacked blocks."""
    
    def __init__(self, input_size, forecast_size, n_stacks=30, pool_size=2, n_layers=2, units=256, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.n_stacks = n_stacks
        self.pool_size = pool_size
        self.n_layers = n_layers
        self.units = units
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'forecast_size': self.forecast_size,
            'n_stacks': self.n_stacks,
            'pool_size': self.pool_size,
            'n_layers': self.n_layers,
            'units': self.units,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
