import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

DATA_PATH = "data/train_2015_2023.csv"

def load_train_data(path=DATA_PATH):
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna(how="all")
    return returns



def build_autoencoder(input_dim, lr=1e-3):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(25, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(25, activation="relu"),
        layers.Dense(input_dim, activation="linear")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse"
    )
    return model


def run_grid_search(returns, param_grid, test_size=0.2):
    X = returns.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val = train_test_split(X_scaled, test_size=test_size, shuffle=False)

    results = []
    keys, values = zip(*param_grid.items())

    for comb in itertools.product(*values):
        params = dict(zip(keys, comb))
        print(f"\n=== Запуск: {params} ===")

        model = build_autoencoder(input_dim=X.shape[1], lr=params["lr"])

        history = model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0
        )

        val_loss = history.history["val_loss"][-1]
        results.append({**params, "val_loss": val_loss})

    df_results = pd.DataFrame(results)
    best = df_results.loc[df_results["val_loss"].idxmin()]
    print("\nЛучшие гиперпараметры:")
    print(best)

    return df_results, best


if __name__ == "__main__":
    returns = load_train_data()

    param_grid = {
        "epochs": [20, 30, 50],
        "batch_size": [16, 32],
        "lr": [1e-3, 5e-4]
    }

    results, best = run_grid_search(returns, param_grid)
