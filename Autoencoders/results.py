import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import prepare_data, get_returns

TEST_START, TEST_END = "2024-01-03", "2024-12-30"
TEST_PATH = "data/test_2024.csv"

def evaluate_portfolio(portfolio: dict, returns: pd.DataFrame):
    weights = np.array([portfolio[t] for t in returns.columns])
    port_ret = returns @ weights  # доходности портфеля
    cum_value = (1 + port_ret).cumprod()
    print(cum_value)

    annual_return = cum_value.iloc[-1] - 1
    volatility = port_ret.std() * np.sqrt(12)  # месячные -> годовые
    sharpe = annual_return / volatility if volatility > 0 else np.nan

    metrics = {
        "Annual Return": float(annual_return),
        "Volatility": float(volatility),
        "Sharpe Ratio": float(sharpe)
    }
    return port_ret, cum_value, metrics

def main():

    portfolios = [
        {'SBER': np.float64(1.6235463315854972e-06), 'TATN': np.float64(0.8152598783330348),
         'SBERP': np.float64(1.6526772010904293e-06), 'PIKK': np.float64(1.3751326718732018e-06),
         'PLZL': np.float64(2.1475853185082616e-06), 'AFKS': np.float64(1.2035025132721334e-06),
         'RUAL': np.float64(2.1844464637342144e-06), 'GAZP': np.float64(2.3166768714787064e-06),
         'GMKN': np.float64(2.4249512105071664e-06), 'MOEX': np.float64(0.1847251931483831)}
,
        {'SBER': np.float64(0.0445), 'TATN': np.float64(0.1296), 'SBERP': np.float64(0.0531), 'PIKK': np.float64(0.1035),
         'PLZL': np.float64(0.1145), 'AFKS': np.float64(0.0963), 'RUAL': np.float64(0.1279), 'GAZP': np.float64(0.1096),
         'GMKN': np.float64(0.089), 'MOEX': np.float64(0.132)}
    ]

    tickers = sorted(portfolios[0].keys())

    # цены и доходности за 2024
    prices = prepare_data(tickers, TEST_START, TEST_END, save_name="test_2024.csv")
    returns = get_returns(prices, freq="ME")  # месячные доходности

    # Оценка портфелей
    results = {}
    plt.figure(figsize=(10,6))

    for i, pf in enumerate(portfolios):
        name = f"Portfolio_{i+1}"
        port_ret, cum_val, metrics = evaluate_portfolio(pf, returns)
        results[name] = metrics


        plt.plot(cum_val.index, cum_val.values, label=name)

    # метрики
    df_results = pd.DataFrame(results).T
    print("\nМетрики портфелей (2024):")
    print(df_results)

    plt.title("Изменение стоимости портфелей (2024)")
    plt.xlabel("Дата")
    plt.ylabel("Стоимость (начало = 1)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
