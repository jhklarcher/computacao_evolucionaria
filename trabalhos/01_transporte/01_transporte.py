import pandas as pd
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools
random.seed(42)

# Dados do problema
demanda = pd.read_csv('demanda.csv', index_col='origem').fillna(0)
custos = pd.read_csv('custos.csv', index_col='origem').fillna(0)
distancias = pd.read_csv('distancias.csv', index_col='origem').fillna(0)
remuneracao = pd.read_csv('remuneracao.csv', index_col='origem').fillna(0)

veiculos_carga = 11
vel_c_carga	=	55
vel_s_carga	=	75
t_carga	=	2
t_descarga	=	2
tamanho_frota = 68
cobertura_min = 72
t_max_viagem = 30 * 24


def n_caminhoes():
  return random.choices(range(0, 68), k = 27)

def evaluate(individual):

  penalidade = 1.0

  individual = individual[0]
  caminhoes = pd.DataFrame([
    [0, individual[0], individual[1], individual[2], individual[3], individual[4], 0],
    [individual[5], individual[6], 0, individual[7], 0, individual[8], individual[9]],
    [0, individual[10], 0, 0, individual[11], 0, individual[12]],
    [individual[13], 0, individual[14], 0, individual[15], individual[16], 0],
    [0, individual[17], individual[18], 0, 0, 0, individual[19]],
    [0, 0, 0, individual[20], individual[21], individual[22], individual[23]],
    [individual[24], 0, 0, individual[25], individual[26], 0, 0]
  ])

  t_viagem_ind = t_carga + t_descarga + vel_c_carga * distancias.values + vel_s_carga * distancias.values
  
  n_viagens_ind = np.floor(t_max_viagem / t_viagem_ind)
  veiculos_ind = n_viagens_ind * veiculos_carga

  remuneracao_ind = np.sum( np.multiply(remuneracao.values, n_viagens_ind) )
  custos_ind = np.sum( np.multiply(custos.values, n_viagens_ind) )

  if np.any(veiculos_ind > demanda.values):
    penalidade = 0.4

  lucro = remuneracao_ind - custos_ind

  return lucro * penalidade, 



#%%
creator.create("Fitness", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

toolbox.register("n_por_carga", n_caminhoes)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.n_por_carga, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

#%%
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
    while g < 100:
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
print(np.asmatrix(best_solution[0]))

individual = best_solution[0][0]
caminhoes = pd.DataFrame([
  [0, individual[0], individual[1], individual[2], individual[3], individual[4], 0],
  [individual[5], individual[6], 0, individual[7], 0, individual[8], individual[9]],
  [0, individual[10], 0, 0, individual[11], 0, individual[12]],
  [individual[13], 0, individual[14], 0, individual[15], individual[16], 0],
  [0, individual[17], individual[18], 0, 0, 0, individual[19]],
  [0, 0, 0, individual[20], individual[21], individual[22], individual[23]],
  [individual[24], 0, 0, individual[25], individual[26], 0, 0]
])

print(caminhoes)

caminhoes.to_csv("ans.csv")





  


  








