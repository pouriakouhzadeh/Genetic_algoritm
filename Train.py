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
# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")

def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled
directory_path = '/home/pouria/Train/CSVFILES'
directory_path_trained = '/home/pouria/main_expert/TRAINED_MODELS/'
extension = '.csv'
extension_model = '.pkl'
feature_selector = FeatureSelection()
# ------------Parameters-------------
depth = 2
Thereshhold = 65
page = 2
feature = 30
QTY = 1000
iter = 100
# ------------Parameters-------------
def time_until_next_hour():
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    time_left = next_hour - now
    return time_left.total_seconds()


while True:
    files = os.listdir(directory_path)
    print(len(files))
    if len(files) == 11 :   
        print("Sleeping for 120 secoand while main expert generating answer ... ") 
        time.sleep(120)
        files_with_creation_time = []
        for filename in os.listdir(directory_path):
                if filename.endswith(extension):
                    file_path = os.path.join(directory_path, filename)
                    creation_time = os.path.getctime(file_path)
                    files_with_creation_time.append((file_path, creation_time))
        files_with_creation_time.sort(key=lambda x: x[1])
        for file_path, creation_time in files_with_creation_time:
            try :
                data = pd.read_csv(file_path)

                model = TrainModels().Train(data ,depth = 2 ,page = 2 ,feature = 30 ,QTY = 1000 ,iter = 100)
                
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"The file at {file_path} has been deleted.")
                else:
                    print(f"The file at {file_path} does not exist.")

                file_path_model = os.path.splitext(file_path)[0] + extension_model
                file_path_model_final = directory_path_trained + file_path_model[-22:]
                with open(file_path_model_final, 'wb') as file:
                    pickle.dump(model, file)
                print("Trained model save by pikle successfully")    
            except :
                print("Can not read file")        
    
    remaining_time = time_until_next_hour()
    print(f"Training Expert showing you the progress of next job ...")
    with tqdm(total=remaining_time, desc="Time until next hour", unit="s") as pbar:
        while remaining_time > 0 :
            files = os.listdir(directory_path)
            if len(files) == 11 :   
                print("New files for training just recived progress broen to coninue proccecing")
                break
            time.sleep(1)
            remaining_time -= 1
            pbar.update(1)
            files = os.listdir(directory_path)
            print(f"Number of file remaining if CSV directory is : {len(files)}")

    print("Next hour has started!")
    