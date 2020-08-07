# %%capture
#!pip install deap
import pandas as pd
import numpy as np
import random

from deap import base
from deap import creator
from deap import tools
random.seed(42)

#%%
div = 10000

cargas = [
    ["Carga 1", 66000/div, 7.75/div, 0.82 * 1000/div],
    ["Carga 2", 55000/div, 10.60/div, 1.15 * 1000/div],
    ["Carga 3", 85000/div, 8.36/div, 0.92 * 1000/div],
    ["Carga 4", 40000/div, 6.30/div, 0.75 * 1000/div]
    ]

compartimentos = [
    ["Dianteiro", 77500, 450],
    ["Central", 115000, 545],
    ["Traseiro", 57500, 305]
    ]

#%%
def n_por_carga():
    return random.choices( range(0, 10000), k = 12)

#%%
def evaluate(individual):
    individual = individual[0]
    
    peso1 = individual[0] * cargas[0][1] + individual[1] * cargas[1][1] + individual[2] * cargas[2][1]+ individual[3] * cargas[3][1]
    peso2 = individual[4] * cargas[0][1] + individual[5] * cargas[1][1] + individual[6] * cargas[2][1]+ individual[7] * cargas[3][1]
    peso3 = individual[8] * cargas[0][1] + individual[9] * cargas[1][1] + individual[10] * cargas[2][1]+ individual[11] * cargas[3][1]
    
    vol1 = individual[0] * cargas[0][2] * cargas[0][1] + individual[1] * cargas[1][2] * cargas[1][1] + individual[2] * cargas[2][2] * cargas[2][1] + individual[3] * cargas[3][2] * cargas[3][1]
    vol2 = individual[4] * cargas[0][2] * cargas[0][1] + individual[5] * cargas[1][2] * cargas[1][1] + individual[6] * cargas[2][2] * cargas[2][1] + individual[7] * cargas[3][2] * cargas[3][1]
    vol3 = individual[8] * cargas[0][2] * cargas[0][1] + individual[9] * cargas[1][2] * cargas[1][1] + individual[10] * cargas[2][2] * cargas[2][1] + individual[11] * cargas[3][2] * cargas[3][1]
    
    lucro1 = individual[0] * cargas[0][3] * cargas[0][1] + individual[1] * cargas[1][3] * cargas[1][1] + individual[2] * cargas[2][3] * cargas[2][1] + individual[3] * cargas[3][3] * cargas[3][1]
    lucro2 = individual[4] * cargas[0][3] * cargas[0][1] + individual[5] * cargas[1][3] * cargas[1][1] + individual[6] * cargas[2][3] * cargas[2][1] + individual[7] * cargas[3][3] * cargas[3][1]
    lucro3 = individual[8] * cargas[0][3] * cargas[0][1] + individual[9] * cargas[1][3] * cargas[1][1] + individual[10] * cargas[2][3] * cargas[2][1] + individual[11] * cargas[3][3] * cargas[3][1]

    if(peso1 > compartimentos[0][1] or peso2 > compartimentos[1][1] or peso3 > compartimentos[2][1] or vol1 > compartimentos[0][2] or vol2 > compartimentos[1][2] or vol3 > compartimentos[2][2]):
        return 0.0, 99999999.9

    lucro =  lucro1 + lucro2 + lucro3

    return lucro,

#%%
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)
toolbox = base.Toolbox()
toolbox.register("n_por_carga", n_por_carga)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.n_por_carga, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

#%%
def main():
    pop = toolbox.population(n=300)
    
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
    while g < 600:
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
        
        stats.append([min(fits), max(fits), mean, std])
    
    best = pop[np.argmax([toolbox.evaluate(x) for x in pop])]
    return best, stats

#%%
best_solution = main()
print(evaluate(best_solution[0]))
print(np.asmatrix(best_solution[0]).reshape(3, 4))

#%%
ans = []
individual = best_solution[0][0]

peso1 = ["peso1", individual[0] * cargas[0][1], individual[1] * cargas[1][1], individual[2] * cargas[2][1], individual[3] * cargas[3][1]]
peso2 = ["peso2", individual[4] * cargas[0][1], individual[5] * cargas[1][1], individual[6] * cargas[2][1], individual[7] * cargas[3][1]]
peso3 = ["peso3", individual[8] * cargas[0][1], individual[9] * cargas[1][1], individual[10] * cargas[2][1], individual[11] * cargas[3][1]]
    
vol1 = ["vol1", individual[0] * cargas[0][2] * cargas[0][1], individual[1] * cargas[1][2] * cargas[1][1], individual[2] * cargas[2][2] * cargas[2][1], individual[3] * cargas[3][2] * cargas[3][1]]
vol2 = ["vol2", individual[4] * cargas[0][2] * cargas[0][1], individual[5] * cargas[1][2] * cargas[1][1], individual[6] * cargas[2][2] * cargas[2][1], individual[7] * cargas[3][2] * cargas[3][1]]
vol3 = ["vol3", individual[8] * cargas[0][2] * cargas[0][1], individual[9] * cargas[1][2] * cargas[1][1], individual[10] * cargas[2][2] * cargas[2][1], individual[11] * cargas[3][2] * cargas[3][1]]
    
lucro1 = ["lucro1", individual[0] * cargas[0][3] * cargas[0][1], individual[1] * cargas[1][3] * cargas[1][1], individual[2] * cargas[2][3] * cargas[2][1], individual[3] * cargas[3][3] * cargas[3][1]]
lucro2 = ["lucro2", individual[4] * cargas[0][3] * cargas[0][1], individual[5] * cargas[1][3] * cargas[1][1], individual[6] * cargas[2][3] * cargas[2][1], individual[7] * cargas[3][3] * cargas[3][1]]
lucro3 = ["lucro3", individual[8] * cargas[0][3] * cargas[0][1], individual[9] * cargas[1][3] * cargas[1][1], individual[10] * cargas[2][3] * cargas[2][1], individual[11] * cargas[3][3] * cargas[3][1]]

ans.append(peso1)
ans.append(peso2)
ans.append(peso3)
ans.append(vol1)
ans.append(vol2)
ans.append(vol3)
ans.append(lucro1)
ans.append(lucro2)
ans.append(lucro3)

ans = pd.DataFrame(ans, columns=["tipo", "carga1", "carga2", "carga3", "carga4"])
ans.index = ans["tipo"]
ans = ans.drop(["tipo"], axis = 1)
ans["total"] = ans.sum(axis = 1)

ans
