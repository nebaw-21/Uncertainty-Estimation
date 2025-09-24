# Function for MC Dropout inference: Take a model with dropout, perform N=30 stochastic forward passes on input data
# Return mean prediction and standard deviation (uncertainty)
import numpy as np
import tensorflow as tf

def mc_dropout_predict(model, X, n_samples=30):
    preds = []
    for _ in range(n_samples):
        # Enable dropout at inference by setting training=True
        pred = model(X, training=True).numpy().flatten()
        preds.append(pred)
    preds = np.stack(preds, axis=0)
    mean_pred = np.mean(preds, axis=0)
    std_pred = np.std(preds, axis=0)
    return mean_pred, std_pred
