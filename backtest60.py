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
from Train_model import TrainModels
import os
import pickle
import warnings
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import logging
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler
from Clean_data import CLEAN_DATA
from timeconvert import TimeConvert
from selecttimetodelete import SelectTimeToDelete
from preparing_data import PREPARE_DATA
from PAGECREATOR import PageCreator
from deleterow import DeleteRow
from FEATURESELECTION import FeatureSelection
import pickle
from Desision_class import DesisionClass
# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")

logging.basicConfig(filename="backtest55.log", level=logging.INFO)

def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled


# ------------Parameters-------------
depth = 2
Thereshhold = 60
page =60
feature = 30
QTY = 1000
iter = 100
# ------------Parameters-------------


Data_EURUSD =  pd.read_csv("EURUSD60.csv")
Data_AUDCAD =  pd.read_csv("AUDCAD60.csv")
Data_AUDCHF =  pd.read_csv("AUDCHF60.csv")
Data_AUDNZD =  pd.read_csv("AUDNZD60.csv")
Data_AUDUSD =  pd.read_csv("AUDUSD60.csv")
Data_EURAUD =  pd.read_csv("EURAUD60.csv")
Data_EURCHF =  pd.read_csv("EURCHF60.csv")
Data_EURGBP =  pd.read_csv("EURGBP60.csv")
Data_GBPUSD =  pd.read_csv("GBPUSD60.csv")
Data_USDCAD =  pd.read_csv("USDCAD60.csv")
Data_USDCHF =  pd.read_csv("USDCHF60.csv")

Data_EURUSD_for_train =  pd.read_csv("EURUSD60_for_train.csv")
Data_AUDCAD_for_train =  pd.read_csv("AUDCAD60_for_train.csv")
Data_AUDCHF_for_train =  pd.read_csv("AUDCHF60_for_train.csv")
Data_AUDNZD_for_train =  pd.read_csv("AUDNZD60_for_train.csv")
Data_AUDUSD_for_train =  pd.read_csv("AUDUSD60_for_train.csv")
Data_EURAUD_for_train =  pd.read_csv("EURAUD60_for_train.csv")
Data_EURCHF_for_train =  pd.read_csv("EURCHF60_for_train.csv")
Data_EURGBP_for_train =  pd.read_csv("EURGBP60_for_train.csv")
Data_GBPUSD_for_train =  pd.read_csv("GBPUSD60_for_train.csv")
Data_USDCAD_for_train =  pd.read_csv("USDCAD60_for_train.csv")
Data_USDCHF_for_train =  pd.read_csv("USDCHF60_for_train.csv")


def win_los(Data_EURUSD_for_train_input,Data_EURUSD_for_train_target,Data_EURUSD_for_answer):
    Data_EURUSD_for_train_input.reset_index(inplace=True, drop=True)
    Data_EURUSD_for_train_target.reset_index(inplace=True, drop=True)
    Data_EURUSD_for_answer.reset_index(inplace=True, drop=True)
    wins = 0
    loses = 0
    model = TrainModels().Train(Data_EURUSD_for_train_input ,depth = 2 ,page = 2 ,feature = 30 ,QTY = 1000 ,iter = 100)
    position, Answer = DesisionClass().Desision(Data_EURUSD_for_answer ,model ,page = 2 ,feature = 30 ,QTY = 1000 ,depth = 2, Thereshhold = 60)
    if position == "BUY":
        if Data_EURUSD_for_train_target["close"][1] >  Data_EURUSD_for_train_target["close"][0] :
            wins =wins + 1
        else :
            loses = loses + 1
    if position == "SELL":
        if Data_EURUSD_for_train_target["close"][1] <  Data_EURUSD_for_train_target["close"][0] :
            wins =wins + 1
        else :
            loses = loses + 1

    return wins, loses


wins = 0 
loses = 0

for i in range (22000,24900):
   Time =  pd.to_datetime(Data_EURUSD.iloc[i-1]["time"])
   print(Time.hour)
   if Time.hour > 17 and Time.hour != 0 :
        W,L =  win_los(Data_EURUSD_for_train[i-3000:i], Data_EURUSD[i-1:i+1], Data_EURUSD[i-3000:i])
        wins = wins + W
        loses = loses + L 
        #------------------------------------------------------------------
        W,L =  win_los(Data_AUDCAD_for_train[i-3000:i], Data_AUDCAD[i-1:i+1], Data_AUDCAD[i-3000:i])
        wins = wins + W
        loses = loses + L 
        #------------------------------------------------------------------
        W,L =  win_los(Data_AUDCHF_for_train[i-3000:i], Data_AUDCHF[i-1:i+1], Data_AUDCHF[i-3000:i])
        wins = wins + W
        loses = loses + L 
        #------------------------------------------------------------------
        W,L =  win_los(Data_AUDNZD_for_train[i-3000:i], Data_AUDNZD[i-1:i+1], Data_AUDNZD[i-3000:i])
        wins = wins + W
        loses = loses + L 
        #------------------------------------------------------------------
        W,L =  win_los(Data_AUDUSD_for_train[i-3000:i], Data_AUDUSD[i-1:i+1], Data_AUDUSD[i-3000:i])
        wins = wins + W
        loses = loses + L 
        #------------------------------------------------------------------
        W,L =  win_los(Data_EURAUD_for_train[i-3000:i], Data_EURAUD[i-1:i+1], Data_EURAUD[i-3000:i])
        wins = wins + W
        loses = loses + L 
        #------------------------------------------------------------------
        W,L =  win_los(Data_EURCHF_for_train[i-3000:i], Data_EURCHF[i-1:i+1], Data_EURCHF[i-3000:i])
        wins = wins + W
        loses = loses + L 
        #------------------------------------------------------------------
        W,L =  win_los(Data_EURGBP_for_train[i-3000:i], Data_EURGBP[i-1:i+1], Data_EURGBP[i-3000:i])
        wins = wins + W
        loses = loses + L 
        #------------------------------------------------------------------
        W,L =  win_los(Data_GBPUSD_for_train[i-3000:i], Data_GBPUSD[i-1:i+1], Data_GBPUSD[i-3000:i])
        wins = wins + W
        loses = loses + L 
        #------------------------------------------------------------------
        W,L =  win_los(Data_USDCAD_for_train[i-3000:i], Data_USDCAD[i-1:i+1], Data_USDCAD[i-3000:i])
        wins = wins + W
        loses = loses + L 
        #------------------------------------------------------------------
        W,L =  win_los(Data_USDCHF_for_train[i-3000:i], Data_USDCHF[i-1:i+1], Data_USDCHF[i-3000:i])
        wins = wins + W
        loses = loses + L 
        #------------------------------------------------------------------
        try :
            print(f"wins = {wins} ----- loses = {loses} ---- ACC = {wins/(wins+loses)}")    
            logging.info(f"BT60 : wins = {wins} ----- loses = {loses} ---- ACC = {wins/(wins+loses)}") 
        except:
            print(f"wins = {wins} ----- loses = {loses}")    
            logging.info(f"BT60 : wins = {wins} ----- loses = {loses}")     
            

