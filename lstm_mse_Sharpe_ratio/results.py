import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_RESULTS = "data_results"



def compute_portfolio_value(df):
    """
    df: DataFrame вида [date, *_price, *_weight]
    выход: Series с портфельной стоимостью
    """
    price_cols = [c for c in df.columns if c.endswith("_price")]
    weight_cols = [c for c in df.columns if c.endswith("_weight")]

    # Доходности активов
    returns = df[price_cols].pct_change().fillna(0).values
    weights = df[weight_cols].values

    # Доходности портфеля
    portf_returns = np.sum(weights * returns, axis=1)

    # изменение стоимости портфеля (начиная с 1.0)
    nav = (1 + portf_returns).cumprod()
    return pd.Series(nav, index=df["date"])


def annualized_return(portf_returns, periods_per_year=252):
    mean_ret = np.mean(portf_returns)
    return mean_ret * periods_per_year


def annualized_volatility(portf_returns, periods_per_year=252):
    return np.std(portf_returns) * np.sqrt(periods_per_year)


def sharpe_ratio(portf_returns, periods_per_year=252, rf=0.0):
    ret = annualized_return(portf_returns, periods_per_year)
    vol = annualized_volatility(portf_returns, periods_per_year)
    return (ret - rf) / (vol + 1e-8)


def sortino_ratio(portf_returns, periods_per_year=252):
    mean_ret = np.mean(portf_returns)
    downside = portf_returns[portf_returns < 0]
    downside_std = np.std(downside) * np.sqrt(periods_per_year)
    return mean_ret * periods_per_year / (downside_std + 1e-8)



if __name__ == "__main__":
    files = [f for f in os.listdir(DATA_RESULTS) if f.endswith(".csv")]
    if not files:
        raise RuntimeError("Нет файлов в директории data_results")

    results = {}

    plt.figure(figsize=(10, 6))

    for fname in files:
        path = os.path.join(DATA_RESULTS, fname)
        df = pd.read_csv(path, parse_dates=["date"])

        # NAV и доходности
        nav = compute_portfolio_value(df)
        portf_returns = nav.pct_change().dropna().values

        # Метрики
        ann_ret = annualized_return(portf_returns)
        ann_vol = annualized_volatility(portf_returns)
        sharpe = sharpe_ratio(portf_returns)
        sortino = sortino_ratio(portf_returns)

        results[fname] = {
            "Annualized Return": ann_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino
        }

        # график NAV
        plt.plot(nav.index, nav.values, label=fname.replace(".csv", ""))

    # Вывод метрик
    print("\n------------- Метрики по портфелям -------------")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # График
    plt.title("Динамика стоимости портфелей (NAV)")
    plt.xlabel("Дата")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
