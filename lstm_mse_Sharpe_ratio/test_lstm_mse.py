import os
import pandas as pd
import numpy as np
import cvxpy as cp
import tensorflow as tf

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "data_results"
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



def optimize_portfolio(pred_returns, cov_matrix):
    """
    На основе предсказанных доходностей pred_returns и ковариации cov_matrix
    ищем портфель, максимизирующий ожидаемую доходность при ограничении риска.
    """
    n = len(pred_returns)
    w = cp.Variable(n)

    max_daily_variance = 1**2/252#(1.0 / np.sqrt(252)) ** 2

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        cp.quad_form(w, cov_matrix) <= max_daily_variance
    ]

    objective = cp.Maximize(pred_returns @ w)
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.SCS, verbose=False)
        if w.value is None:
            return np.ones(n) / n
        w_opt = np.array(w.value).clip(min=0)
        return w_opt / w_opt.sum()
    except Exception as e:
        print("CVXPY error:", e)
        return np.ones(n) / n



if __name__ == "__main__":
    df = load_data("test_2024.csv")
    df = normalize_prices(df)

    # окна
    X_test, dates = create_windows(df, lookback=LOOKBACK)

    mask = (dates >= "2024-01-01")
    X_test, dates = X_test[mask], dates[mask]

    print("Форма X_test:", X_test.shape)
    print("Количество дат:", len(dates))

    # модель
    model_path = os.path.join(MODEL_DIR, "lstm_mse_portfolio_model.keras")
    model = tf.keras.models.load_model(model_path, compile=False)

    # прогнозы доходностей
    preds = model.predict(X_test, verbose=0)  # (samples, n_assets)

    # Ковариация на основе последних LOOKBACK дней
    ret_cols = [c for c in df.columns if c.endswith("_ret")]
    returns = df[ret_cols].values

    weights = []
    for i, date in enumerate(dates):
        window_returns = returns[i:i+LOOKBACK]  # окно для ковариации
        Sigma = np.cov(window_returns.T)
        w = optimize_portfolio(preds[i], Sigma)
        weights.append(w)

    weights = np.array(weights)

    # Собираем данные о портфеле в DataFrame: даты, цены (close), веса
    price_cols = [c for c in df.columns if c.endswith("_price")]
    tickers = [c.replace("_price", "") for c in price_cols]

    close_test = df.loc[dates, price_cols].reset_index(drop=True)
    weights_df = pd.DataFrame(weights, columns=[f"{t}_weight" for t in tickers])

    result = pd.concat([pd.Series(dates, name="date"), close_test, weights_df], axis=1)

    save_path = os.path.join(RESULTS_DIR, "lstm_mse_test_results.csv")
    result.to_csv(save_path, index=False)
    print(f"Результаты сохранены: {save_path}")

