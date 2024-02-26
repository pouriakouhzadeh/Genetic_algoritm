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
data = data_source[199000:200000]
data = CLEAN_DATA().clear(data)
data, target = PREPARE_DATA().ready(data)
data, target = PageCreator().create(data, target, 4)
feature_selector = FeatureSelection()
data_1 = feature_selector.select(data, target, 14).copy()

model = CatBoostClassifier(iterations=5000, depth=16, learning_rate=0.1, loss_function='Logloss', verbose=True)
data1 = normalize_data(data_1)
X_train, X_test, y_train, y_test = train_test_split(data_1, target, test_size=0.1, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
pred = model.predict_proba(X_test)
accuracy = accuracy_score(y_test, predictions)
y_test.reset_index(inplace = True , drop = True)
treshhold = 0.55
wins = 0
loses = 0
for i in range(len(y_test)) :
    if pred[i][1] > treshhold :
        if y_test.loc[i] == 1 :
            wins = wins + 1
        else :
            loses = loses + 1    
    if pred[i][0] > treshhold :
        if y_test.loc[i] == 0 :
            wins = wins + 1
        else :
            loses = loses + 1       

ACC_proba = (wins * 100) / (wins + loses)                 
print(f"ACC = {accuracy}")
print(f"ACC_proba = {ACC_proba} , wins = {wins} , loses = {loses}")

# test1 0.2  itraition 1000 number of ddata : 3000
# test 0.1 itration 5000 

