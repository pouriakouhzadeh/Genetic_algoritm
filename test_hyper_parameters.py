import csv
import random
import numpy as np
import pandas as pd


def read_data(file_name, tail_size):
    chunk_size = 1000
    chunks = pd.read_csv(file_name, chunksize=chunk_size)
    tail_data = pd.concat([chunk for chunk in chunks]).tail(tail_size)

    if tail_data.empty:
        return None

    return tail_data


def model_prediction(parameters, currency_pair):
    depth, page, feature, iter, threshold, tail_size = parameters

    if currency_pair is None:
        return 0

    currency_data = read_data(currency_pair, tail_size)

    if currency_data is None:
        return 0

    model_results = TrainModels().Train(currency_data, depth, page, feature, iter, threshold)
    ACC, wins, loses = model_results

    if (wins + loses < 100):
        return 0
    else:
        return ACC

if __name__ == "__main__":
    file_names = ["EURUSD60.csv", "AUDCAD60.csv", "AUDCHF60.csv", "AUDNZD60.csv", "AUDUSD60.csv",
                  "EURAUD60.csv", "EURCHF60.csv", "EURGBP60.csv", "GBPUSD60.csv", "USDCAD60.csv", "USDCHF60.csv"]

    currency_pair = file_names[0]  # انتخاب یک جفت ارز برای بهینه‌سازی

