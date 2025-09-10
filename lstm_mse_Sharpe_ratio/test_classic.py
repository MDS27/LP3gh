import os
import pandas as pd
import numpy as np
import cvxpy as cp


DATA_DIR = "data"
RESULTS_DIR = "data_results"
LOOKBACK = 50



def load_data(filename):

    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.dropna(how="any")

def create_windows(df, lookback=50):
    """
    окна (lookback x n_assets) только по доходностям
    """
    ret_cols = [c for c in df.columns if c.endswith("_ret")]
    returns = df[ret_cols].values

    X, dates = [], []
    for i in range(lookback, len(df) - 1):
        X.append(returns[i - lookback:i])  # окно доходностей
        dates.append(df.index[i + 1])      # дата следующего дня

    return np.array(X, dtype=np.float32), pd.to_datetime(dates)


def optimize_markowitz(returns_window):
    mu = np.mean(returns_window, axis=0)
    Sigma = np.cov(returns_window.T)

    n = len(mu)
    w = cp.Variable(n)

    max_daily_variance =  1**2/252

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        cp.quad_form(w, Sigma) <= max_daily_variance
    ]

    objective = cp.Maximize(mu @ w)
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

    # окна
    X_test, dates = create_windows(df, lookback=LOOKBACK)

    mask = (dates >= "2024-01-01")
    X_test, dates = X_test[mask], dates[mask]

    print("Форма X_test:", X_test.shape)
    print("Количество дат:", len(dates))

    # веса
    weights = []
    for window in X_test:
        w = optimize_markowitz(window)
        weights.append(w / w.sum())  # нормируем, чтобы сумма = 1

    weights = np.array(weights)

    # Собираем данные о портфеле в DataFrame: даты, цены (close), веса
    price_cols = [c for c in df.columns if c.endswith("_price")]
    tickers = [c.replace("_price", "") for c in price_cols]

    close_test = df.loc[dates, price_cols].reset_index(drop=True)
    weights_df = pd.DataFrame(weights, columns=[f"{t}_weight" for t in tickers])

    result = pd.concat([pd.Series(dates, name="date"), close_test, weights_df], axis=1)

    save_path = os.path.join(RESULTS_DIR, "classic_test_results.csv")
    result.to_csv(save_path, index=False)
    print(f"Результаты сохранены: {save_path}")

    print(result.head())
