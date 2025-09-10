import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

DATA_PATH = "data/train_2015_2023.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_train_data(path=DATA_PATH):
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna(how="any")
    return returns


def build_autoencoder(input_dim, lr=1e-3):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(25, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(25, activation="relu"),
        layers.Dense(input_dim, activation="linear")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse"
    )
    return model



def train_autoencoder(returns, epochs=50, batch_size=32, lr=1e-3):
    X = returns.values

    # Стандартизация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Модель
    print(X.shape[1])
    model = build_autoencoder(input_dim=X.shape[1], lr=lr)
    model.summary()

    # Обучение
    history = model.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    model.save(os.path.join(MODEL_DIR, "autoencoder.keras"))
    np.save(os.path.join(MODEL_DIR, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(MODEL_DIR, "scaler_scale.npy"), scaler.scale_)

    print(f"Модель сохранена в {MODEL_DIR}/autoencoder.keras")

    return model, scaler, history


if __name__ == "__main__":
    returns = load_train_data()
    print("Размер данных:", returns.shape)

    model, scaler, history = train_autoencoder(
        returns, epochs=100, batch_size=32, lr=1e-3
    )
    print("Обучение завершено и модель сохранена")
