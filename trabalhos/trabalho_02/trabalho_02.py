import math
import operator

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools
from sklearn.metrics import mean_squared_error

df_treino = pd.read_csv("treino.csv").replace(["DOWN", "STABLE", "UP"], [-1, 0, 1])
X_treino = df_treino.drop(["datetime", "class"], axis=1).values
y_treino = df_treino["class"].values

df_teste = pd.read_csv("teste.csv").replace(["DOWN", "STABLE", "UP"], [-1, 0, 1])
X_teste = df_treino.drop(["datetime", "class"], axis=1).values
y_teste = df_treino["class"].values


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(np.round, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)

pset.renameArguments(ARG0="sma")
pset.renameArguments(ARG1="wma")
pset.renameArguments(ARG2="macd")
pset.renameArguments(ARG3="rsi")
pset.renameArguments(ARG4="mom")

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evaluate(individual):
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    func = toolbox.compile(individual)

    # Yp = np.array(list(map(func, X)))
    Yp = np.round(
        np.array(
            list(
                map(
                    func,
                    X_treino[:, 0],
                    X_treino[:, 1],
                    X_treino[:, 2],
                    X_treino[:, 3],
                    X_treino[:, 4],
                )
            )
        )
    )
    # return the MSE as we are evaluating on it anyway - then the stats are more fun to watch...
    return (mean_squared_error(y_treino, Yp),)


toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(
    pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=True
)
