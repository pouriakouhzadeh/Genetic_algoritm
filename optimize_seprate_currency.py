import csv
import random
import numpy as np
import pandas as pd

class TrainModels:
    def Train(self, currency_data, depth, page, feature, iter, threshold):
        # Dummy implementation for the sake of completeness. Replace with actual training logic.
        return random.random(), random.randint(0, 100), random.randint(0, 100)

def read_data(file_name, tail_size):
    chunk_size = 1000
    chunks = pd.read_csv(file_name, chunksize=chunk_size)
    tail_data = pd.concat([chunk for chunk in chunks]).tail(tail_size)
    if tail_data.empty:
        return None
    return tail_data

def model_prediction(parameters, currency_data):
    depth, page, feature, iter, threshold, tail_size = parameters

    if currency_data is None:
        return 0

    model_results = TrainModels().Train(currency_data, depth, page, feature, iter, threshold)
    ACC, wins, loses = model_results

    if (wins + loses < 100):
        return 0
    else:
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

def genetic_algorithm(param_space, population_size, generations, mutation_rate, currency_file, tail_size):
    population = generate_population(population_size, param_space)
    currency_data = read_data(currency_file, tail_size)
    if currency_data is None:
        return None, 0
    
    for generation in range(generations):
        fitness_scores = [model_prediction(individual + [tail_size], currency_data) for individual in population]
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

def run_genetic_algorithm_for_all_currencies(param_space, population_size, generations, mutation_rate, currency_files, tail_size):
    results = {}
    for currency_file in currency_files:
        best_params, best_fitness = genetic_algorithm(param_space, population_size, generations, mutation_rate, currency_file, tail_size)
        results[currency_file] = (best_params, best_fitness)
    return results

if __name__ == "__main__":
    file_names = ["EURUSD60.csv", "AUDCAD60.csv", "AUDCHF60.csv", "AUDNZD60.csv", "AUDUSD60.csv",
                  "EURAUD60.csv", "EURCHF60.csv", "EURGBP60.csv", "GBPUSD60.csv", "USDCAD60.csv", "USDCHF60.csv"]
    param_space = [(2, 16), (2, 20), (30, 500), (100, 5000), (55, 80)]
    population_size = 50
    generations = 1000
    mutation_rate = 0.1
    tail_size = 1000

    results = run_genetic_algorithm_for_all_currencies(param_space, population_size, generations, mutation_rate, file_names, tail_size)

    for currency_file, (best_params, best_fitness) in results.items():
        print(f"Best Parameters for {currency_file}: {best_params}")
        print(f"Best Fitness for {currency_file}: {best_fitness}")

    result_file = 'genetic_algorithm_results_separate.csv'
    with open(result_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Currency File', 'Best Parameters', 'Best Fitness'])
        for currency_file, (best_params, best_fitness) in results.items():
            writer.writerow([currency_file, best_params, best_fitness])
import csv
import random
import numpy as np
import pandas as pd

class TrainModels:
    def Train(self, currency_data, depth, page, feature, iter, threshold):
        # Dummy implementation for the sake of completeness. Replace with actual training logic.
        return random.random(), random.randint(0, 100), random.randint(0, 100)

def read_data(file_name, tail_size):
    chunk_size = 1000
    chunks = pd.read_csv(file_name, chunksize=chunk_size)
    tail_data = pd.concat([chunk for chunk in chunks]).tail(tail_size)
    if tail_data.empty:
        return None
    return tail_data

def model_prediction(parameters, currency_data):
    depth, page, feature, iter, threshold, tail_size = parameters

    if currency_data is None:
        return 0

    model_results = TrainModels().Train(currency_data, depth, page, feature, iter, threshold)
    ACC, wins, loses = model_results

    if (wins + loses < 100):
        return 0
    else:
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

def genetic_algorithm(param_space, population_size, generations, mutation_rate, currency_file, tail_size):
    population = generate_population(population_size, param_space)
    currency_data = read_data(currency_file, tail_size)
    if currency_data is None:
        return None, 0
    
    for generation in range(generations):
        fitness_scores = [model_prediction(individual + [tail_size], currency_data) for individual in population]
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

def run_genetic_algorithm_for_all_currencies(param_space, population_size, generations, mutation_rate, currency_files, tail_size):
    results = {}
    for currency_file in currency_files:
        best_params, best_fitness = genetic_algorithm(param_space, population_size, generations, mutation_rate, currency_file, tail_size)
        results[currency_file] = (best_params, best_fitness)
    return results

if __name__ == "__main__":
    file_names = ["EURUSD60.csv", "AUDCAD60.csv", "AUDCHF60.csv", "AUDNZD60.csv", "AUDUSD60.csv",
                  "EURAUD60.csv", "EURCHF60.csv", "EURGBP60.csv", "GBPUSD60.csv", "USDCAD60.csv", "USDCHF60.csv"]
    param_space = [(2, 16), (2, 20), (30, 500), (100, 5000), (55, 80)]
    population_size = 50
    generations = 1000
    mutation_rate = 0.1
    tail_size = 1000

    results = run_genetic_algorithm_for_all_currencies(param_space, population_size, generations, mutation_rate, file_names, tail_size)

    for currency_file, (best_params, best_fitness) in results.items():
        print(f"Best Parameters for {currency_file}: {best_params}")
        print(f"Best Fitness for {currency_file}: {best_fitness}")

    result_file = 'genetic_algorithm_results_separate.csv'
    with open(result_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Currency File', 'Best Parameters', 'Best Fitness'])
        for currency_file, (best_params, best_fitness) in results.items():
            writer.writerow([currency_file, best_params, best_fitness])
