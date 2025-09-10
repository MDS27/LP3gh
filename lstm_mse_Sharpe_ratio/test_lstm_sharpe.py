import os
import pandas as pd
import numpy as np
import tensorflow as tf

DATA_DIR = "data"
RESULTS_DIR = "data_results"
MODEL_DIR = "models"
LOOKBACK = 50

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
    окна признаков для LSTM
    X: последовательности (lookback x features)
    """
    price_cols = [c for c in df.columns if c.endswith("_price")]
    ret_cols = [c for c in df.columns if c.endswith("_ret")]

    X = []
    data = df[price_cols + ret_cols].values

    for i in range(lookback, len(df) - 1):
        X.append(data[i - lookback:i])    # окно (lookback дней)

    X = np.array(X, dtype=np.float32)
    return X,  df.index[lookback+1:]



if __name__ == "__main__":
    # тестовые данные
    df = load_data("test_2024.csv")
    df = normalize_prices(df)

    # окна
    X_test, dates = create_windows(df, lookback=LOOKBACK)
    mask = (dates >= "2024-01-01")
    X_test, dates = X_test[mask], dates[mask]

    print("Форма X_test:", X_test.shape)
    print("Количество дат:", len(dates))

    # модель
    model_path = os.path.join(MODEL_DIR, "lstm_sharpe_portfolio_model.keras")
    model = tf.keras.models.load_model(model_path, compile=False)

    # веса активов портфеля для тестового периода
    weights = model.predict(X_test, verbose=0)

    # Собираем данные о портфеле в DataFrame: даты, цены (close), веса
    price_cols = [c for c in df.columns if c.endswith("_price")]
    tickers = [c.replace("_price", "") for c in price_cols]


    # цены
    close_test = df.loc[dates, price_cols].reset_index(drop=True)

    # веса
    weights_df = pd.DataFrame(weights, columns=[f"{t}_weight" for t in tickers])

    # объединяем
    result = pd.concat([pd.Series(dates, name="date"), close_test.reset_index(drop=True), weights_df], axis=1)

    save_path = os.path.join(RESULTS_DIR, "lstm_sharpe_test_results.csv")
    result.to_csv(save_path, index=False)
    print(f"Результаты сохранены: {save_path}")

    print(result.head())