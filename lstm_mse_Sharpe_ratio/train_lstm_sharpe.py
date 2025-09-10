import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "data"
MODEL_DIR = "models"

def load_data(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.dropna(how="any")

def normalize_prices(df):
    price_cols = [c for c in df.columns if c.endswith("_price")]
    df_norm = df.copy()
    for col in price_cols:
        df_norm[col] = df_norm[col] / df_norm[col].iloc[0]
    return df_norm

def create_windows(df, lookback=50):
    """
    обучающие окна признаков для LSTM
    X: последовательности (lookback x features)
    y: доходности на следующий день
    """
    price_cols = [c for c in df.columns if c.endswith("_price")]
    ret_cols = [c for c in df.columns if c.endswith("_ret")]

    X, y = [], []
    data = df[price_cols + ret_cols].values
    returns = df[ret_cols].values

    for i in range(lookback, len(df) - 1):
        X.append(data[i - lookback:i])    # окно (lookback дней)
        y.append(returns[i + 1])          # доходности на следующий день

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y



def sharpe_loss(y_true, w, eps=1e-8):
    """
    y_true: фактические доходности активов
    w: веса для активов портфеля, нормированные softmax
    Loss = -Sharpe ratio (максимизация Sharpe ratio)
    """
    # Доходности портфеля
    portf_returns = tf.reduce_sum(y_true * w, axis=1)

    mean = tf.reduce_mean(portf_returns)
    std = tf.math.reduce_std(portf_returns) + eps

    sharpe = mean / std
    return -sharpe



def build_model(input_shape, num_assets):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64),
        layers.Dense(num_assets, activation="softmax")
    ])
    return model



if __name__ == "__main__":
    # обучающие данные
    df = load_data("train_2015_2023.csv")

    # нормализация
    df = normalize_prices(df)

    # окна признаков
    X_train, y_train = create_windows(df, lookback=50)

    print("X_train:", X_train.shape)  # (samples, 50, features)
    print("y_train:", y_train.shape)  # (samples, assets)

    # модель
    num_assets = y_train.shape[1]
    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_model(input_shape, num_assets)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=sharpe_loss)

    # обучение
    history = model.fit(
        X_train, y_train,
        epochs=485,
        batch_size=64,
        verbose=1
    )

    model.save(os.path.join(MODEL_DIR, "lstm_sharpe_portfolio_model.keras"))
    print("Модель сохранена.")