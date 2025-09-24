# Function to train K=5 independent MLPs with different random seeds
# Return list of trained models
from models.simple_mlp import build_mlp
import numpy as np
import tensorflow as tf

def train_ensemble(X_train, y_train, k=5, epochs=100, batch_size=32):
    models_list = []
    for seed in range(k):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        model = build_mlp(X_train.shape[1])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        models_list.append(model)
    return models_list

# Function for ensemble inference: Predict with all models, return mean and std across ensemble preds
def ensemble_predict(models_list, X):
    preds = [model.predict(X).flatten() for model in models_list]
    preds = np.stack(preds, axis=0)
    mean_pred = np.mean(preds, axis=0)
    std_pred = np.std(preds, axis=0)
    return mean_pred, std_pred
