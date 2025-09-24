# Define a simple feed-forward MLP using Keras: Input -> Dense(32, ReLU) -> Dropout(0.2) -> Dense(16, ReLU) -> Dense(1)
# Compile with MSE loss, Adam optimizer
# Include a function to build the model
from tensorflow.keras import layers, models

def build_mlp(input_shape):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
