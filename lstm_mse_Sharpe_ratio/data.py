import pandas as pd
import requests
import time
import os
from apimoex import get_board_history

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def load_moex_data(session, ticker, start, end, retries=3, delay=1.0):
    for attempt in range(retries):
        try:
            data = get_board_history(session, ticker, start=start, end=end)
            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError(f"Нет данных для {ticker}")
            return df.set_index("TRADEDATE")["CLOSE"].rename(ticker)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            print(f"[{ticker}] Ошибка SSL/сети, попытка {attempt+1}/{retries}: {e}")
            time.sleep(delay)
    print(f"[{ticker}] Не удалось загрузить данные")
    return None


def prepare_data(tickers, start, end, save_name, batch_size=5, delay=0.5):
    dfs = []

    with requests.Session() as session:
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            print(f"\nТикеры: {batch}")
            for ticker in batch:
                df = load_moex_data(session, ticker, start, end)
                if df is not None:
                    dfs.append(df)
            time.sleep(delay)  # пауза между батчами

    prices = pd.concat(dfs, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index)

    # дневные доходности
    returns = prices.pct_change(fill_method=None).dropna(how="all")

    prices = prices.drop(prices.index[0])

    # данные по тикерам с MOEX (цены + доходности)
    combined = pd.concat(
        [prices.add_suffix("_price"), returns.add_suffix("_ret")],
        axis=1
    )

    path = os.path.join(DATA_DIR, save_name)
    combined.to_csv(path)
    print(f"\nДанные для обучения (цены + доходности): {path}")

    return combined

def filter_by_nans(prices: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    nan_ratio = prices.isna().mean()
    bad_tickers = nan_ratio[nan_ratio > threshold].index.tolist()
    if bad_tickers:
        print(f"Удалены активы с NaN > {threshold*100:.0f}%: {bad_tickers}")
    return prices.drop(columns=bad_tickers)

if __name__ == "__main__":
    tickers = ['SBER', 'TATN', 'SBERP', 'PIKK', 'PLZL', 'AFKS', 'RUAL', 'GAZP', 'GMKN', 'MOEX'] # 5+5

    train_prices = prepare_data(tickers, "2015-01-01", "2023-12-31", save_name="train_2015_2023.csv")
    path = os.path.join(DATA_DIR, "train_2015_2023.csv")
    prices = pd.read_csv(path, index_col=0, parse_dates=True)

    # фильтрация NaN
    prices = filter_by_nans(prices, threshold=0.1)

    # данные для обучения 2015-2023
    path = os.path.join(DATA_DIR, "train_2015_2023.csv")
    prices.to_csv(path)

    # тестовые данные за 2024
    test_prices = prepare_data(tickers, "2023-09-01", "2024-12-31", save_name="test_2024.csv")
