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

def model_prediction(parameters, currency_file):
    depth, page, feature, iter, threshold, tail_size = parameters
    currency_data = read_data(currency_file, tail_size)
    if currency_data is None:
        return 0  # بازگشت 0 به عنوان دقت اگر داده‌ای نباشد
    ACC, wins, loses = TrainModels().Train(currency_data, depth, page, feature, iter, threshold)
    return ACC

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

def genetic_algorithm(param_space, population_size, generations, mutation_rate, currency_file):
    population = generate_population(population_size, param_space)
    for generation in range(generations):
        fitness_scores = [model_prediction(individual, currency_file) for individual in population]
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
    currency_file = "EURUSD60.csv"  # تغییر به استفاده از یک جفت ارز
    param_space = [(4, 8), (5, 10), (50, 200), (200, 1000), (60, 70), (1000, 3000)]
    population_size = 40
    generations = 150
    mutation_rate = 0.2

    best_params, best_fitness = genetic_algorithm(param_space, population_size, generations, mutation_rate, currency_file)

    print("Best Parameters for EUR/USD Pair:", best_params)
    print("Best Performance:", best_fitness)

    result_file = 'genetic_algorithm_results_eurusd.csv'
    with open(result_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Best Parameters', 'Best Performance'])
        writer.writerow([best_params, best_fitness])
UU