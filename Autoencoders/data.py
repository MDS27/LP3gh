import pandas as pd
import requests
import datetime
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


def prepare_data(tickers, start, end, save_name=None, batch_size=5, delay=0.5):
    dfs = []

    with requests.Session() as session:
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            print(f"\nЗагружаем тикеры: {batch}")
            for ticker in batch:
                df = load_moex_data(session, ticker, start, end)
                if df is not None:
                    dfs.append(df)
            time.sleep(delay)  # пауза между батчами

    if not dfs:
        raise RuntimeError("Не удалось загрузить данные ни по одному тикеру")

    result = pd.concat(dfs, axis=1).sort_index()
    result.index = pd.to_datetime(result.index)  # преобразуем строки в даты

    if save_name:
        path = os.path.join(DATA_DIR, save_name)
        result.to_csv(path)
        print(f"\nДанные сохранены: {path}")

    return result


def get_returns(prices: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    returns = prices.resample(freq).last().pct_change().dropna(how="all")
    return returns


def get_test_returns(tickers: list) -> (pd.DataFrame, pd.Series):
    start, end = "2023-12-01", "2024-12-31"
    prices = prepare_data(tickers, start, end, save_name="test_2024.csv")

    monthly_returns = get_returns(prices, freq="ME")
    annual_returns = (prices.iloc[-1] / prices.iloc[0] - 1)

    return monthly_returns, annual_returns



def filter_by_nans(prices: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    nan_ratio = prices.isna().mean()
    bad_tickers = nan_ratio[nan_ratio > threshold].index.tolist()
    if bad_tickers:
        print(f"Удалены активы с NaN > {threshold*100:.0f}%: {bad_tickers}")
    return prices.drop(columns=bad_tickers)



if __name__ == "__main__":
    tickers = ["AFKS", "AFLT", "ALRS", "CHMF", "DSKY", "ENPG", "FIVE", "FIXP", "GAZP", "GLTR", "GMKN", "HYDR", "IRAO",
               "MAGN", "MGNT", "MOEX", "MTSS", "NLMK", "NVTK", "OZON", "PHOR", "PIKK", "PLZL", "POLY", "ROSN", "RTKM",
               "RUAL", "SBER", "SBERP", "SNGS", "SNGSP", "TATN", "TATNP", "TCSG", "TRNFP", "VKCO", "VTBR", "YNDX"]

    train_prices = prepare_data(tickers, "2015-01-01", "2023-12-31", save_name="train_2015_2023.csv")
    path = os.path.join(DATA_DIR, "train_2015_2023.csv")
    prices = pd.read_csv(path, index_col=0, parse_dates=True)

    # фильтрация NaN
    prices = filter_by_nans(prices, threshold=0.1)

    # пересохраняем очищенные данные
    path = os.path.join(DATA_DIR, "train_2015_2023.csv")
    prices.to_csv(path)

    test = prepare_data(['SBER', 'TATN', 'SBERP', 'PIKK', 'PLZL', 'AFKS', 'RUAL', 'GAZP', 'GMKN', 'MOEX'], "2024-01-01", "2024-12-31", save_name="test_2024.csv")