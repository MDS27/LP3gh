import pandas as pd
import numpy as np
import cvxpy as cp


TICKERS =  ['SBER', 'TATN', 'SBERP', 'PIKK', 'PLZL', 'AFKS', 'RUAL', 'GAZP', 'GMKN', 'MOEX'] # 5+5
TRAIN_PRICES_PATH = "data/train_2015_2023.csv"
TRADING_DAYS_IN_YEAR = 252

def optimize_markowitz(returns_window):
    mu = np.mean(returns_window, axis=0).values
    Sigma = np.cov(returns_window.T)*252

    n = len(mu)
    w = cp.Variable(n)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        cp.quad_form(w, Sigma) <= 0.2**2
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
    prices = pd.read_csv(TRAIN_PRICES_PATH, index_col=0, parse_dates=True)
    prices = prices[TICKERS].dropna(how="any")  # важно убрать все NaN
    prices = prices[prices.index.year.isin([2023])]
    print(prices)

    ret_d = prices.pct_change().dropna()

    weight = optimize_markowitz(ret_d)
    markowitz_weight_dict = {t: np.float64(w_) for t, w_ in zip(TICKERS, weight)}
    print("\nВеса Марковица (в порядке тикеров):")
    print(markowitz_weight_dict)