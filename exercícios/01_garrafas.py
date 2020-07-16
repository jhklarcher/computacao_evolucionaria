#%%capture
#!pip install deap
import pandas as pd
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools
random.seed(42)

tipos = [
    ["Leite", 1, 6.0/100, 10, 5.0],
    ["Suco", 1, 5.0/100, 20, 4.5]
    ]

max_leite = 800
cap_max = 15000
max_tempo = 60

def n_por_carga():
    return random.choices(range(0, 2), k = 20)

def evaluate(individual):

    individual = individual[0]

    leite = int("".join(str(x) for x in individual[0:10]), 2)
    suco = int("".join(str(x) for x in individual[10:19]), 2)

    vol = leite * tipos[0][1] * tipos[0][3] + suco * tipos[1][1] * tipos[1][3]

    lucro = leite * tipos[0][1] * tipos[0][4] + suco * tipos[1][1] * tipos[1][4]

    tempo = leite * tipos[0][2] + suco * tipos[1][2]

    n_leite = leite * tipos[0][1]

    penalidade = 0

    if((tempo > max_tempo) or (n_leite > max_leite) or (vol > cap_max)):
        #return 0.0,
        penalidade = 0.5

    return lucro*(1-penalidade),

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

toolbox.register("n_por_carga", n_por_carga)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.n_por_carga, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=100)
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # CXPB  is the probability with which two individuals
    #       are crossed

    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.1
    
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    
    # Variable keeping track of the number of generations
    g = 0
    stats = []
    # Begin the evolution
    while g < 1000:
        # A new generation
        g = g + 1
        # print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1[0], child2[0])
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values
            
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        stats.append([mean, max(fits)]) #min(fits), , std
    
    best = pop[np.argmax([toolbox.evaluate(x) for x in pop])]
    return best, stats

best_solution = main()
print(evaluate(best_solution[0]))
print(best_solution[0])

individual = best_solution[0][0]

leite = int("".join(str(x) for x in individual[0:10]), 2)
suco = int("".join(str(x) for x in individual[10:19]), 2)

vol = leite * tipos[0][1] * tipos[0][3] + suco * tipos[1][1] * tipos[1][3]
lucro = leite * tipos[0][1] * tipos[0][4] + suco * tipos[1][1] * tipos[1][4]
tempo = leite * tipos[0][2] + suco * tipos[1][2]
n_leite = leite * tipos[0][1]

print("Volume: {}".format(vol))
print("Tempo de produção: {}".format(tempo))
print("Garrafas de leite: {}".format(n_leite))
print("Lucro: {}".format(lucro))

# %%
