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
import logging
import warnings
# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")
# Making objects -------------------------------------------
def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

data_source = pd.read_csv('EURUSD_M15.csv')
data = data_source[195000:]
data = CLEAN_DATA().clear(data)
data, target = PREPARE_DATA().ready(data)
data, target = PageCreator().create(data, target, 4)
kmeans = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 3000, random_state = 42)
kmeans.fit(data)
result_data, result_target = SEPRATE_DATA().start(data, target, kmeans.labels_, 6)
feature_selector = FeatureSelection()
variable_name = f"data_{2}"
globals()[variable_name] = result_data[2].copy()
variable_name_target = f"target_{2}"
globals()[variable_name_target] = result_target[2].copy()
variable_name_select = f"data_{2}_select"
globals()[variable_name_select] = feature_selector.select(eval(variable_name), eval(variable_name_target), 142).copy()

model = CatBoostClassifier(iterations=5000, depth=1, learning_rate=0.01, loss_function='Logloss', verbose=True)
globals()[variable_name_select] = normalize_data(eval(variable_name_select))
X_train, X_test, y_train, y_test = train_test_split(eval(variable_name_select), eval(variable_name_target), test_size=0.2, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
pred = model.predict_proba(X_test)
accuracy = accuracy_score(y_test, predictions)
y_test.reset_index(inplace = True , drop = True)
wins = 0
loses = 0
for i in range(len(y_test)) :
    if pred[i][1] > 0.6 :
        if y_test.loc[i][0] == 1 :
            wins = wins + 1
        else :
            loses = loses + 1    
    if pred[i][0] > 0.6 :
        if y_test.loc[i][0] == 0 :
            wins = wins + 1
        else :
            loses = loses + 1       

ACC_proba = (wins * 100) / (wins + loses)                 
print(f"ACC = {accuracy}")
print(f"ACC_proba = {ACC_proba} , wins = {wins} , loses = {loses}")

# test1 0.2  itraition 1000 number of ddata : 3000
# test 0.1 itration 5000 

