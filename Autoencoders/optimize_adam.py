import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

from data import get_returns

TICKERS = ['SBER', 'TATN', 'SBERP', 'PIKK', 'PLZL', 'AFKS', 'RUAL', 'GAZP', 'GMKN', 'MOEX']
TRAIN_PATH = "data/train_2015_2023.csv"


def build_autoencoder(input_dim, lr=1e-3):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(5, activation="relu"),
        layers.Dense(input_dim, activation="linear")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse"
    )
    return model


if __name__ == "__main__":
    prices = pd.read_csv(TRAIN_PATH, index_col=0, parse_dates=True)
    prices = prices[TICKERS].dropna()
    prices = prices[prices.index.year.isin([2023])]

    # дневные доходности
    returns = get_returns(prices, freq="D")
    X = returns.values

    # Стандартизация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # автоэнкодер
    model = build_autoencoder(input_dim=X.shape[1], lr=1e-3)
    model.summary()
    model.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)

    # предсказания
    X_hat = model.predict(X_scaled, verbose=0)

    # Ошибки по каждому активу
    errors = np.mean((X_scaled - X_hat) ** 2, axis=0)

    # Нормировка
    weights = errors / errors.sum()

    # Результаты
    df = pd.DataFrame({
        "Ticker": TICKERS,
        "Reconstruction_Error": errors,
        "Weight": weights
    }).sort_values("Reconstruction_Error", ascending=False)

    print("\nОшибки реконструкции и веса портфеля (Autoencoder):")
    print(df.to_string(index=False))

    print("\nИтоговые веса (в порядке тикеров):")
    print(dict(zip(TICKERS, weights.round(4))))

