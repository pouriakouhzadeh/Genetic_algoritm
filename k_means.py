import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preparing_data import PREPARE_DATA
from Clean_data import CLEAN_DATA
from sklearn.cluster import KMeans
from seprate_data import SEPRATE_DATA
from model_tester import ModelTester
from FEATURESELECTION import FeatureSelection
from PAGECREATOR import PageCreator
import logging
import warnings
# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")
# Making objects -------------------------------------------
# Information of initial parameters ------------------------
# 1 - count > count_data 
# 2 - count_page_creator > count page page creator
# 3 - count_k_means > count k_means
# 4 - count_feature > count feature
# 5 - i > cluster number
# 6 - counter 
#-----------------------------------------------------------
# Reading initial parameters from file ---------------------
with open('initial_parameters.txt', 'r') as file:
    content = file.read()

numbers_list = content.split('\n')

count_init, count_page_creator_init, count_k_means_init, count_feature_init, i_init, counter = map(int, numbers_list[:6])
#-----------------------------------------------------------
logging.basicConfig(filename= "result.log", level=logging.INFO)
data_source = pd.read_csv('EURUSD_M15.csv')
data = 0
target = 0
for count in range (count_init,0 ,-10000) :
    len_data = 200000 - count
    print("Len data : {len_data}")
    data1 = data_source[count:]
    print("Start clean data ....")
    data1 = CLEAN_DATA().clear(data1)
    print("End of clean data")

    print("Start preparing data ....")
    data1, target1 = PREPARE_DATA().ready(data1)
    print("End of preparing data")
    for count_page_creator in range(count_page_creator_init, 10):
        print("Start creating pages ....")
        del data
        del target
        data, target = PageCreator().create(data1, target1, count_page_creator)
        print("Crating pages finish")
        for count_k_means in range(count_k_means_init, 7) :
            print("Start KMeans....")
            kmeans = KMeans(n_clusters = count_k_means, init = 'k-means++', max_iter = 3000, random_state = 42)
            kmeans.fit(data)
            print("End KMeans")

            print("Start seprate data ....")
            result_data, result_target = SEPRATE_DATA().start(data, target, kmeans.labels_, count_k_means)
            print("End seprate data")
            for count_feature in range (count_feature_init, 200, 2) :
                print("Start feature selection ....")
                feature_selector = FeatureSelection()
                for i in range(i_init, count_k_means) :
                    variable_name = f"data_{i}"
                    globals()[variable_name] = result_data[i].copy()
                    variable_name_target = f"target_{i}"
                    globals()[variable_name_target] = result_target[i].copy()
                    variable_name_select = f"data_{i}_select"
                    globals()[variable_name_select] = feature_selector.select(eval(variable_name), eval(variable_name_target), count_feature).copy()
                    print("End feature selection")        
                    print(f"Start testing all models on {i}st data ....")    
                    ModelTester(eval(variable_name_select), eval(variable_name_target)).test_models(f" {i}st Cluster ", count_feature, len_data, count_k_means, i, count_page_creator)
                    print("End testet models on {i}st data")
                    counter = counter + 1
                    print(f"|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                    print(f"Stage : {counter} - number of features : {count_feature} - number of clusters : {count_k_means} - current cluster : {i} - number of Data : {200000-count}, number of pages : {count_page_creator}")
                    print(f"|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                     
                    if count_page_creator == 9 :
                        count_page_creator_init = 1
                    if count_k_means == 6 :
                        count_k_means_init = 2
                    if count_feature == 198 :
                        count_feature_init = 2
                    if i == count_k_means-1 :
                        i_init = 0
                    with open('initial_parameters.txt', 'w') as file:
                        file.write(f"{count}\n{count_page_creator}\n{count_k_means}\n{count_feature}\n{i}\n{counter}")

