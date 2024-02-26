import time
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import pandas as pd
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

def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled
directory_path = '/home/pouria/main_expert/CSVfiles'
directory_path_models = '/home/pouria/main_expert/TRAINED_MODELS/'
directory_path_answer = '/home/pouria/main_expert/ANSWER/'
extension = '.csv'
extension_answer = '.txt'
extension_acn = '.acn'
extension_model = '_for_train.pkl'
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


def delete_acns():
    files = os.listdir(directory_path_answer)
    if len(files) == 11 :
         time.sleep(1)
         for count in range (11) :
            temp = files[count]
            print(temp[-3:])
            if temp[-3:] == "acn" :
                os.remove(directory_path_answer+files[count])
                print(f" Deleting ACN files : {count}")

def delete_TXT_files():
    files = os.listdir(directory_path_answer)
    print(f"Number of file remaining if ANSWER directory is : {len(files)}")
    files1 = os.listdir(directory_path_answer)

    if len(files) == 22  :
        time.sleep(1)
        while len(files1) >= 1 :
            os.remove(directory_path_answer+files[len(files1)-1])
            # time.sleep(2)
            files1 = os.listdir(directory_path_answer)
            print(f" Deleting ACN and TXT files {len(files1)}")


while True:
    files = os.listdir(directory_path)
    print(f" Number of CSV files wating for making answer is :  {len(files)}")
    if len(files) == 11 :
            
        files_with_creation_time = []
        for filename in os.listdir(directory_path):
                if filename.endswith(extension):
                    file_path = os.path.join(directory_path, filename)
                    creation_time = os.path.getctime(file_path)
                    files_with_creation_time.append((file_path, creation_time))
        files_with_creation_time.sort(key=lambda x: x[1])
        for file_path, creation_time in files_with_creation_time:
            file_path_answer = os.path.splitext(file_path)[0] + extension_answer
            file_path_model = file_path[:-21] + 'TRAINED_MODELS/' +file_path[-12:-4] + extension_model
            data = pd.read_csv(file_path)
            # file_path_akn = file_path[:-3] + "akn"
            # with open(file_path_akn, 'w') as file:
            #     file.write('aknowladgement')
            print(f"Start to predicting {file_path[-12:-6]} :")
            position, Answer = DesisionClass().Desision(data ,file_path_model ,page = 2 ,feature = 30 ,QTY = 1000 ,depth = 2  )

            print(file_path[-12:-6]+" ------> "+position)

            with open(directory_path_answer + file_path[-12:-4] + '.txt', 'w') as file:
                file.write(position+"\n")
                file.write(str(Answer[0, 0])+"\n")
                file.write(str(Answer[0, 1]))

            os.remove(file_path)



              
    time.sleep(1)
    delete_acns()

    remaining_time = time_until_next_hour()
    print(f"Main Expert showing you the progress of next job ...")
    with tqdm(total=remaining_time, desc="Time until next hour", unit="s") as pbar:
        while remaining_time > 0 :
            files = os.listdir(directory_path_answer)
            if len(files) == 11 :   
                print("New files for calculating just recived progress broken to coninue proccecing")
                delete_acns()
                break
            files = os.listdir(directory_path_answer)            
            if len(files) == 22 :   
                print("New files for calculating just recived progress broken to coninue proccecing")
                delete_TXT_files()
                break     
            
            files = os.listdir(directory_path)
            if len(files) == 11 :   
                break       
            time.sleep(1)
            remaining_time -= 1
            pbar.update(1)
            files = os.listdir(directory_path_answer)
            print(f"Number of file remaining if ANSWER directory is : {len(files)}")

    print("Next hour has started!")
