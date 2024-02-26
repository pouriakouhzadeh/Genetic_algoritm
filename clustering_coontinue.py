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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import logging
import warnings
# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")
# Making objects -------------------------------------------
def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

target = pd.read_csv("target200000_page5")
data = pd.read_csv("data200000_page5.csv")
target = target[100000:]
data = data[100000:]
X_scaled = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(data)

ks = KShape(n_clusters=6, n_init=1, verbose=True, max_iter=10000)
y_pred = ks.fit_predict(X_scaled)

data['cluster'] = y_pred  # افزودن ستون خوشه به دیتافریم داده‌ها
target['cluster'] = y_pred  # افزودن ستون خوشه به دیتافریم تارگت

# ایجاد 6 دیتافریم جداگانه برای هر خوشه

# دیتا
clustered_data = {}
for i in range(6):
    clustered_data[i] = data[data['cluster'] == i].drop('cluster', axis=1)

# تارگت
clustered_target = {}
for i in range(6):
    clustered_target[i] = target[target['cluster'] == i].drop('cluster', axis=1)

for i in range(6):
    clustered_data[i].to_csv(f"Data{i}.csv")
    clustered_target[i].to_csv(f"Target{i}.csv")

# model = CatBoostClassifier(iterations=5000, depth=16, learning_rate=0.01, loss_function='Logloss', verbose=True)
# data1 = normalize_data(data_1)
# X_train, X_test, y_train, y_test = train_test_split(data_1, target, test_size=0.2, random_state=42)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# pred = model.predict_proba(X_test)
# accuracy = accuracy_score(y_test, predictions)
# y_test.reset_index(inplace = True , drop = True)
# wins = 0
# loses = 0
# for i in range(len(y_test)) :
#     if pred[i][1] > 0.6 :
#         if y_test.loc[i] == 1 :
#             wins = wins + 1
#         else :
#             loses = loses + 1    
#     if pred[i][0] > 0.6 :
#         if y_test.loc[i] == 0 :
#             wins = wins + 1
#         else :
#             loses = loses + 1       

# ACC_proba = (wins * 100) / (wins + loses)                 
# print(f"ACC = {accuracy}")
# print(f"ACC_proba = {ACC_proba} , wins = {wins} , loses = {loses}")

# # test1 0.2  itraition 1000 number of ddata : 3000
# # test 0.1 itration 5000 

print("finish")