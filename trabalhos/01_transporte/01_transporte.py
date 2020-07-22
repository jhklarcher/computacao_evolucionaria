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
custos = pd.read_csv('custos.csv', index_col='origem').fillna(0)

veiculos_carga = 11
vel_c_carga	=	55
vel_s_carga	=	75
t_carga	=	2
t_descarga	=	2
tamanho_frota = 68
cobertura_min = 72

def n_caminhoes():
    return random.choices(range(0, 10000), k = 12)

def evaluate(individual):
    individual = individual[0]
    

