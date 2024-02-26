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
from xgboost import XGBClassifier
from ACC_CALC import Acc_Calculator
import logging
import logging
import warnings
# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")
# Making objects -------------------------------------------
def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

logging.basicConfig(filename="Cat_boost_test.log", level=logging.INFO)
data_source = pd.read_csv('EURUSD_M15.csv')
data = data_source[190000:]
data = CLEAN_DATA().clear(data)
data, target = PREPARE_DATA().ready(data)
data, target = PageCreator().create(data, target, 11)
feature_selector = FeatureSelection()
data = feature_selector.select(data, target, 300).copy()
kmeans = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 3000, random_state = 42)
kmeans.fit(data)
labels = kmeans.labels_
result_data, result_target = SEPRATE_DATA().start(data, target, kmeans.labels_, 6)

for i in range(1, 7):
    globals()[f'data_{i}'] = result_data[i-1].copy()
    globals()[f'target_{i}'] = result_target[i-1].copy()


model_1 = CatBoostClassifier(iterations=5000, depth=16, learning_rate=0.01, loss_function='Logloss', verbose=True)
model_2 = CatBoostClassifier(iterations=5000, depth=16, learning_rate=0.01, loss_function='Logloss', verbose=True)
model_3 = CatBoostClassifier(iterations=5000, depth=16, learning_rate=0.01, loss_function='Logloss', verbose=True)
model_4 = CatBoostClassifier(iterations=5000, depth=16, learning_rate=0.01, loss_function='Logloss', verbose=True)
model_5 = CatBoostClassifier(iterations=5000, depth=16, learning_rate=0.01, loss_function='Logloss', verbose=True)
model_6 = CatBoostClassifier(iterations=5000, depth=16, learning_rate=0.01, loss_function='Logloss', verbose=True)

data_1 = normalize_data(data_1)
data_2 = normalize_data(data_2)
data_3 = normalize_data(data_3)
data_4 = normalize_data(data_4)
data_5 = normalize_data(data_5)
data_6 = normalize_data(data_6)

X_train1, X_test1, y_train1, y_test1 = train_test_split(data_1, target_1, test_size=0.1, random_state=1234)
X_train2, X_test2, y_train2, y_test2 = train_test_split(data_2, target_2, test_size=0.1, random_state=1234)
X_train3, X_test3, y_train3, y_test3 = train_test_split(data_3, target_3, test_size=0.1, random_state=1234)
X_train4, X_test4, y_train4, y_test4 = train_test_split(data_4, target_4, test_size=0.1, random_state=1234)
X_train5, X_test5, y_train5, y_test5 = train_test_split(data_5, target_5, test_size=0.1, random_state=1234)
X_train6, X_test6, y_train6, y_test6 = train_test_split(data_6, target_6, test_size=0.1, random_state=1234)

ACC_CALC = Acc_Calculator()

model_1.fit(X_train1, y_train1)
predictions_1 = model_1.predict(X_test1)
predictions_1_proba = model_1.predict_proba(X_test1)
A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test1, predictions_1)
A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test1, predictions_1_proba, 0.6)
A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test1, predictions_1_proba, 20)
print(f"ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.6 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
logging.info(f"ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.6 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")

model_2.fit(X_train2, y_train2)
predictions_2 = model_2.predict(X_test2)
predictions_2_proba = model_2.predict_proba(X_test2)
A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test2, predictions_2)
A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test2, predictions_2_proba, 0.6)
A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test2, predictions_2_proba, 20)
print(f"ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.6 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
logging.info(f"ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.6 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")

model_1.fit(X_train3, y_train3)
predictions_3 = model_3.predict(X_test3)
predictions_3_proba = model_3.predict_proba(X_test3)
A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test3, predictions_3)
A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test3, predictions_3_proba, 0.6)
A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test3, predictions_3_proba, 20)
print(f"ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.6 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
logging.info(f"ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.6 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")

model_4.fit(X_train4, y_train4)
predictions_4 = model_1.predict(X_test4)
predictions_4_proba = model_4.predict_proba(X_test4)
A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test4, predictions_4)
A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test4, predictions_4_proba, 0.6)
A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test4, predictions_4_proba, 20)
print(f"ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.6 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
logging.info(f"ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.6 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")

model_5.fit(X_train5, y_train5)
predictions_5 = model_5.predict(X_test5)
predictions_5_proba = model_5.predict_proba(X_test5)
A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test5, predictions_5)
A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test5, predictions_5_proba, 0.6)
A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test5, predictions_5_proba, 20)
print(f"ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.6 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
logging.info(f"ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.6 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")

model_6.fit(X_train6, y_train6)
predictions_6 = model_1.predict(X_test6)
predictions_6_proba = model_6.predict_proba(X_test6)
A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test6, predictions_6)
A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test6, predictions_6_proba, 0.6)
A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test6, predictions_6_proba, 20)
print(f"ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.6 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
logging.info(f"ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.6 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")




model_2.fit(X_train2, y_train2)
predictions_2 = model2.predict(X_test2)
accuracy_2 = accuracy_score(y_test2, predictions_2)
print(f"ACC2 = {accuracy_2}")
logging.info(f"ACC2 = {accuracy_2}")

model_3.fit(X_train3, y_train3)
predictions_3 = model3.predict(X_test3)
accuracy_3 = accuracy_score(y_test3, predictions_3)
print(f"ACC3 = {accuracy_3}")
logging.info(f"ACC3 = {accuracy_3}")

model_4.fit(X_train4, y_train4)
predictions_4 = model4.predict(X_test4)
accuracy_4 = accuracy_score(y_test4, predictions_4)
print(f"ACC4 = {accuracy_4}")
logging.info(f"ACC4 = {accuracy_4}")

model_5.fit(X_train5, y_train5)
predictions_5 = model5.predict(X_test5)
accuracy_5 = accuracy_score(y_test5, predictions_5)
print(f"ACC5 = {accuracy_5}")
logging.info(f"ACC5 = {accuracy_5}")

model_6.fit(X_train6, y_train6)
predictions_6 = model6.predict(X_test6)
accuracy_6 = accuracy_score(y_test6, predictions_6)
print(f"ACC6 = {accuracy_6}")
logging.info(f"ACC6 = {accuracy_6}")




