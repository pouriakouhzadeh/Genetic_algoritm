import csv
import random
import numpy as np
import pandas as pd
from Train_model import TrainModels

def read_data(file_name, tail_size):
    chunk_size = 1000
    chunks = pd.read_csv(file_name, chunksize=chunk_size)
    tail_data = pd.concat([chunk for chunk in chunks]).tail(tail_size)
    if tail_data.empty:
        return None
    return tail_data

def model_prediction(parameters, currency_files):
    depth, page, feature, iter, threshold, tail_size = parameters
    total_acc = 0
    for currency_file in currency_files:
        currency_data = read_data(currency_file, tail_size)
        if currency_data is None:
            continue
        ACC, wins, loses = TrainModels().Train(currency_data, depth, page, feature, iter, threshold)

        if (wins + loses) >= (len(currency_data)*10)/100:
            total_acc += ACC
    average_acc = total_acc / len(currency_files) if len(currency_files) > 0 else 0
    return average_acc

def generate_population(size, param_space):
    population = []
    for _ in range(size):
        individual = [random.randint(param[0], param[1]) for param in param_space]
        population.append(individual)
    return population

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(individual, mutation_rate, param_space):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = random.randint(param_space[i][0], param_space[i][1])
    return mutated_individual

def genetic_algorithm(param_space, population_size, generations, mutation_rate, currency_files):
    population = generate_population(population_size, param_space)
    for generation in range(generations):
        fitness_scores = [model_prediction(individual, currency_files) for individual in population]
        max_fitness_idx = np.argmax(fitness_scores)
        best_individual = population[max_fitness_idx]
        best_fitness = fitness_scores[max_fitness_idx]
        print(f"Generation {generation + 1}: Best Fitness - {best_fitness}")
        new_population = [best_individual]
        for _ in range(1, population_size):
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            mutated_child = mutate(child, mutation_rate, param_space)
            new_population.append(mutated_child)
        population = new_population
    return best_individual, best_fitness

if __name__ == "__main__":
    file_names = ["EURUSD60.csv", "AUDCAD60.csv", "AUDCHF60.csv", "AUDNZD60.csv", "AUDUSD60.csv",
                  "EURAUD60.csv", "EURCHF60.csv", "EURGBP60.csv", "GBPUSD60.csv", "USDCAD60.csv", "USDCHF60.csv"]
    param_space = [(2, 16), (2, 20), (30, 500), (100, 5000), (55, 80), (500, 6000)]
    population_size = 20
    generations = 50
    mutation_rate = 0.2

    best_params, best_fitness = genetic_algorithm(param_space, population_size, generations, mutation_rate, file_names)

    print("Best Parameters for All Currency Pairs:", best_params)
    print("Best Average Performance:", best_fitness)

    result_file = 'genetic_algorithm_results_overall.csv'
    with open(result_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Best Parameters', 'Best Average Performance'])
        writer.writerow([best_params, best_fitness])
