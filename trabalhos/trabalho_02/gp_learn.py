from datetime import datetime

import graphviz
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from numba import jit
from sklearn.metrics import confusion_matrix

df_treino = pd.read_csv("treino.csv").replace(["DOWN", "STABLE", "UP"], [-1, 0, 1])
df_treino["datetime"] = pd.to_datetime(
    df_treino["datetime"], format="%Y-%m-%d %H:%M:%S"
).map(datetime.timestamp)
X_treino = df_treino.drop(["class"], axis=1).values
y_treino = df_treino["class"].values

df_teste = pd.read_csv("teste.csv").replace(["DOWN", "STABLE", "UP"], [-1, 0, 1])
df_teste["datetime"] = pd.to_datetime(
    df_teste["datetime"], format="%Y-%m-%d %H:%M:%S"
).map(datetime.timestamp)
X_teste = df_treino.drop(["class"], axis=1).values
y_teste = df_treino["class"].values


def prepare_data(df, X_columns, y_columns, window, future_n):
    data = np.zeros((len(df) - window, len(X_columns) * window))
    for i in range(0, len(df) - window):
        aux = np.zeros((len(X_columns), window))
        for j in range(0, len(X_columns)):
            aux[j, :] = df[X_columns[j]].iloc[i : i + window].values
        # print(df.index.values[i:i+window])
        data[i, :] = aux.reshape(1, len(X_columns) * window)

    future_data = data[len(data) - future_n :,].astype(np.float64)
    X = data[: len(data) - future_n,].astype(np.float64)
    y = df[y_columns].values[future_n + window :].astype(np.float64).reshape(-1,)
    return (X, future_data, y)


window = 3

n_future = 0

X_columns = ["sma", "wma", "macd", "rsi", "mom", "class"]

y_columns = ["class"]

X_treino, _, y_treino = prepare_data(df_treino, X_columns, y_columns, window, n_future)

X_teste, _, y_teste = prepare_data(df_teste, X_columns, y_columns, window, n_future)

X_treino = X_treino[:, -1:]
X_teste = X_teste[:, -1:]


est_gp = SymbolicRegressor(
    population_size=30_000,
    generations=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.00001,
    random_state=0,
    metric="mse",
    n_jobs=-1,
    function_set=("add", "sub", "mul", "div", "sin", "cos"),
)

est_gp.fit(X_treino, y_treino)

est_gp.fit(X_treino, y_treino)

y_hat = est_gp.predict(X_treino)
y_preds = est_gp.predict(X_teste)

pd.DataFrame(y_hat).to_csv("y_hat.csv")
pd.DataFrame(y_preds).to_csv("y_preds.csv")


print(confusion_matrix(y_teste, y_hat))


dot_data = est_gp._program.export_graphviz()
graph = graphviz.Source(dot_data)
graph
print(est_gp._program)
