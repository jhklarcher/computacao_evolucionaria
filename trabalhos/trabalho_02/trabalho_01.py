import operator
import random

import geppy as gep
import numpy as np
import pandas as pd
from deap import base, creator, tools
from numba import jit

# for reproduction
s = 0
random.seed(s)
np.random.seed(s)

df_treino = pd.read_csv("treino.csv").replace(["DOWN", "STABLE", "UP"], [-1, 0, 1])
X_treino = df_treino.drop(["datetime", "class"], axis=1).values
y_treino = df_treino["class"].values

df_teste = pd.read_csv("teste.csv").replace(["DOWN", "STABLE", "UP"], [-1, 0, 1])
X_teste = df_treino.drop(["datetime", "class"], axis=1).values
y_teste = df_treino["class"].values


def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2


pset = gep.PrimitiveSet("Main", input_names=["sma", "wma", "macd", "rsi", "mom"])

pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
# pset.add_function(np.sin, 1)        # I tested adding my own functions
# pset.add_function(np.cos, 1)
# pset.add_function(np.tan, 1)
pset.add_rnc_terminal()

creator.create(
    "FitnessMin", base.Fitness, weights=(-1,)
)  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

h = 7  # head length
n_genes = 2  # number of genes in a chromosome
r = 5  # length of the RNC array

toolbox = gep.Toolbox()
toolbox.register(
    "rnc_gen", random.randint, a=-10, b=10
)  # each RNC is random integer within [-5, 5]
toolbox.register(
    "gene_gen",
    gep.GeneDc,
    pset=pset,
    head_length=h,
    rnc_gen=toolbox.rnc_gen,
    rnc_array_length=r,
)
toolbox.register(
    "individual",
    creator.Individual,
    gene_gen=toolbox.gene_gen,
    n_genes=n_genes,
    linker=operator.add,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register("compile", gep.compile_, pset=pset)


@jit
def evaluate(individual):
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    func = toolbox.compile(individual)

    # Yp = np.array(list(map(func, X)))

    Yp = np.array(
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
    # return the MSE as we are evaluating on it anyway - then the stats are more fun to watch...

    return (np.mean((y_treino - Yp) ** 2),)


toolbox.register("evaluate", evaluate)

toolbox.register("select", tools.selTournament, tournsize=3)
# 1. general operators
toolbox.register("mut_uniform", gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register("mut_invert", gep.invert, pb=0.1)
toolbox.register("mut_is_transpose", gep.is_transpose, pb=0.1)
toolbox.register("mut_ris_transpose", gep.ris_transpose, pb=0.1)
toolbox.register("mut_gene_transpose", gep.gene_transpose, pb=0.1)
toolbox.register("cx_1p", gep.crossover_one_point, pb=0.3)
toolbox.register("cx_2p", gep.crossover_two_point, pb=0.2)
toolbox.register("cx_gene", gep.crossover_gene, pb=0.1)

# 2. Dc-specific operators
toolbox.register("mut_dc", gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
toolbox.register("mut_invert_dc", gep.invert_dc, pb=0.1)
toolbox.register("mut_transpose_dc", gep.transpose_dc, pb=0.1)

# for some uniform mutations, we can also assign the ind_pb a string to indicate our expected number of point mutations in an individual
toolbox.register(
    "mut_rnc_array_dc", gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb="0.5p"
)
toolbox.pbs[
    "mut_rnc_array_dc"
] = 1  # we can also give the probability via the pbs property


stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# size of population and number of generations
n_pop = 1000
n_gen = 100

# 100 3000

champs = 3

pop = toolbox.population(n=n_pop)  #
hof = tools.HallOfFame(
    champs
)  # only record the best three individuals ever found in all generations

# start evolution
pop, log = gep.gep_simple(
    pop,
    toolbox,
    n_generations=n_gen,
    n_elites=1,
    stats=stats,
    hall_of_fame=hof,
    verbose=True,
)

best_ind = hof[0]
symplified_best = gep.simplify(best_ind)

print(symplified_best)
