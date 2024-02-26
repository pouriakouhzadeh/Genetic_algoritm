import pandas as pd
import numpy as np
from Clean_data import CLEAN_DATA
from FEATURESELECTION import FeatureSelection
from PAGECREATOR import PageCreator
from preparing_data import PREPARE_DATA
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from ACC_CALC import Acc_Calculator
from bestcatboost import BestCatBoost
from timeconvert import TimeConvert
from selecttimetodelete import SelectTimeToDelete
from deleterow import DeleteRow
from sklearn.model_selection import train_test_split
import warnings
import logging
# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")

logging.basicConfig(filename="Best_models_ACC.log", level=logging.INFO)

def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

ACC_CALC = Acc_Calculator()
feature_selector = FeatureSelection()
data_source = pd.read_csv('EURUSD_M15.csv')
# model = CatBoostClassifier(iterations=500, depth=2, learning_rate=0.01, loss_function='Logloss', verbose=True)
Stage = 0
TotalWins = 0
TotalLoses = 0
for i in range ( int( ( len(data_source) / 5 ) * 4 ) ,len(data_source) - 101, 100 ) :
    Stage = Stage + 1
    data = data_source[i-800 : i+100]
    data = CLEAN_DATA().clear(data)
    data = TimeConvert().exec(data)
    data = SelectTimeToDelete().exec(data, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,0])
    data, target = PREPARE_DATA().ready(data)
    data, target = PageCreator().create(data, target, 6)
    # data2 = PageCreator().create_dataset(data, 12)
    
    data, target = DeleteRow().exec(data, target)
    data = feature_selector.select(data, target, 50).copy()
    data = normalize_data(data)
    # X_train = data[:len(data)-100]
    # X_test = data[len(data)-100:]
    # y_train = target[:len(data)-100]
    # y_test = target[len(data)-100:]
    X_train, X_test, y_train, y_test = train_test_split(
                    data, target, test_size=0.05, random_state=1234)
    print("Starting to Train best models ....")
    wins, loses = BestCatBoost().TrainModels(X_train, y_train, X_test, y_test, Stage, i)
    TotalWins = TotalWins + wins
    TotalLoses = TotalLoses + loses
    try:
        logging.info(f"TotalWins = {TotalWins}, TotalLoses = {TotalLoses}, TotalACC= = {(TotalWins * 100) / (TotalWins + TotalLoses)}")
    except:
        logging.info("Division by zero")
    del data, target, X_test, X_train, y_test, y_train