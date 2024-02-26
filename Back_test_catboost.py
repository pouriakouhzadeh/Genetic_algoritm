import pandas as pd
import numpy as np
from Clean_data import CLEAN_DATA
from FEATURESELECTION import FeatureSelection
from PAGECREATOR import PageCreator
from preparing_data import PREPARE_DATA
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from ACC_CALC import Acc_Calculator
import logging
import warnings
# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")

def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

logging.basicConfig(filename="Cat_boost_test.log", level=logging.INFO)
ACC_CALC = Acc_Calculator()
feature_selector = FeatureSelection()
data_source = pd.read_csv('EURUSD_M15.csv')
model = CatBoostClassifier(iterations=5000, depth=16, learning_rate=0.01, loss_function='Logloss', verbose=True)
Stage = 0
for i in range ( int( ( len(data_source) / 8 ) * 7 ) ,len(data_source) - 101, 100 ) :
    Stage = Stage + 1
    data = data_source[i-2000 : i+100]
    data = CLEAN_DATA().clear(data)
    data, target = PREPARE_DATA().ready(data)
    data, target = PageCreator().create(data, target, 11)
    data = feature_selector.select(data, target, 100).copy()
    data = normalize_data(data)
    X_train = data[:2000]
    X_test = data[2000:]
    y_train = target[:2000]
    y_test = target[2000:]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)
    A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test, predictions)
    A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test, predictions_proba, 0.65)
    A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test, predictions_proba, 20)
    print(f"Stage : {Stage}, ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
    logging.info(f"Stage : {Stage}, ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
    del data, target, X_test, X_train, y_test, y_train, predictions, predictions_proba