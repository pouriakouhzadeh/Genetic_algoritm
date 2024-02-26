import csv
import random
import numpy as np
import pandas as pd
from TR_MODEL import TrainModels  # فرض بر این است که تابع TrainModels برای کار با allowed_hours به‌روزرسانی شده است

currency_files = ["EURUSD60.csv", "AUDCAD60.csv", "AUDCHF60.csv", "AUDNZD60.csv", "AUDUSD60.csv",
                  "EURAUD60.csv", "EURCHF60.csv", "EURGBP60.csv", "GBPUSD60.csv", "USDCAD60.csv", "USDCHF60.csv"]

def read_data(file_name, tail_size):
    chunk_size = 1000
    chunks = pd.read_csv(file_name, chunksize=chunk_size)
    tail_data = pd.concat([chunk for chunk in chunks]).tail(tail_size)
    if tail_data.empty:
        return None
    return tail_data

def generate_individual(param_space, min_allowed_hours=8, max_allowed_hours=18):
    individual = [random.randint(param[0], param[1]) for param in param_space[:-1]]  # ایجاد هایپرپارامترهای قبلی
    hours = list(range(24))
    num_allowed_hours = random.randint(min_allowed_hours, max_allowed_hours)
    allowed_hours = random.sample(hours, num_allowed_hours)
    individual.append(allowed_hours)  # اضافه کردن allowed_hours به عنوان هایپرپارامتر جدید
    return individual

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    child[-1] = list(set(child[-1]))  # حذف تکراری‌ها
    if len(child[-1]) > 18:
        child[-1] = random.sample(child[-1], 18)  # اطمینان از حداکثر تعداد ساعات مجاز
    return child

def mutate(individual, mutation_rate, param_space):
    for i in range(len(individual) - 1):
        if random.random() < mutation_rate:
            individual[i] = random.randint(param_space[i][0], param_space[i][1])
    if random.random() < mutation_rate:
        possible_hours = list(set(range(24)) - set(individual[-1]))
        if possible_hours:
            random_hour = random.choice(possible_hours)
            if len(individual[-1]) < 18:
                individual[-1].append(random_hour)
            else:
                individual[-1][random.randint(0, len(individual[-1]) - 1)] = random_hour
    return individual

def save_to_file(file_name, data):
    with open(file_name, "a") as file:
        file.write(data + "\n")

def model_prediction(individual, currency_files):
    total_acc = 0
    valid_models = 0
    for currency_file in currency_files:
        currency_data = read_data(currency_file, 3000)
        if currency_data is not None:
            print(f"ACC : {currency_file}")
            acc, wins, loses = TrainModels().Train(currency_data, *individual[:-1], primit_hours=individual[-1])
            if wins + loses >= 0.2 * (len(currency_data)*0.1):
                total_acc += acc
                valid_models += 1
            else:
                print(f"Model for {currency_file} ignored due to insufficient wins/loses.")
    avg_acc = total_acc / valid_models if valid_models > 0 else 0
    return avg_acc

def genetic_algorithm_for_all_currencies(currency_files, population_size, generations, mutation_rate, param_space, results_file="ga_results.txt"):
    best_results = {}  # برای ذخیره بهترین نتایج برای هر جفت ارز

    for currency_file in currency_files:
        population = [generate_individual(param_space) for _ in range(population_size)]
        best_fitness = -1
        best_individual = None

        for generation in range(generations):
            fitness_scores = [model_prediction(individual, [currency_file]) for individual in population]
            total_fitness = sum(fitness_scores)
            save_to_file(results_file, f"Currency: {currency_file}, Generation {generation + 1}: Total Fitness = {total_fitness}")

            population_fitness = list(zip(population, fitness_scores))
            population_fitness.sort(key=lambda x: x[1], reverse=True)
            best_individual_gen, best_fitness_gen = population_fitness[0]
            if best_fitness_gen > best_fitness:
                best_fitness = best_fitness_gen
                best_individual = best_individual_gen
            save_to_file(results_file, f"Currency: {currency_file}, Generation {generation + 1}: Best Fitness = {best_fitness_gen}")

            selected = population_fitness[:len(population) // 2]
            population = [ind for ind, _ in selected]
            while len(population) < population_size:
                parent1, parent2 = random.sample(selected, 2)
                child = crossover(parent1[0], parent2[0])
                child = mutate(child, mutation_rate, param_space)
                population.append(child)

        best_results[currency_file] = (best_individual, best_fitness)

    save_to_file(results_file, f"Best Results for All Currencies:")
    for currency_file, (best_individual, best_fitness) in best_results.items():
        save_to_file(results_file, f"Currency: {currency_file}, Best Individual: {best_individual}, Fitness: {best_fitness}")

# استفاده از تابع genetic_algorithm_for_all_currencies برای اجرای الگوریتم ژنتیک بر روی تمامی جفت‌های ارز

param_space = [(2, 16), (2, 20), (30, 500), (500, 6000), (100, 5000), (55, 80), (0, 23)]

genetic_algorithm_for_all_currencies(currency_files, population_size=150, generations=40, mutation_rate=0.02, param_space=param_space)
