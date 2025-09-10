import os
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from train import load_train_data

MODEL_DIR = "models"
DATA_PATH = "data/train_2015_2023.csv"


def load_model_and_scaler():
    model = keras.models.load_model(os.path.join(MODEL_DIR, "autoencoder.keras"))

    mean = np.load(os.path.join(MODEL_DIR, "scaler_mean.npy"))
    scale = np.load(os.path.join(MODEL_DIR, "scaler_scale.npy"))

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale

    return model, scaler


def evaluate_assets(model, scaler, returns: pd.DataFrame):
    X = returns.values
    X_scaled = scaler.transform(X)

    X_recon = model.predict(X_scaled, verbose=0)

    # Ошибка по каждому активу (среднее квадратичное отклонение)
    errors = np.mean((X_scaled - X_recon) ** 2, axis=0)

    df_errors = pd.DataFrame({
        "Ticker": returns.columns,
        "ReconstructionError": errors
    }).sort_values("ReconstructionError")

    return df_errors


if __name__ == "__main__":
    # Загружаем данные и модель
    returns = load_train_data(DATA_PATH)
    model, scaler = load_model_and_scaler()

    # ошибки
    df_errors = evaluate_assets(model, scaler, returns)

    print("\nОшибки реконструкции по активам")
    print(df_errors)

    # Выбираем 8 лучших + 2 худших
    best_8 = df_errors.head(8)["Ticker"].tolist()
    worst_2 = df_errors.tail(2)["Ticker"].tolist()
    selected = best_8 + worst_2

    print("\nВыбранные активы (8 + 2 ):")
    print(selected)
