import datetime
import pandas as pd
import numpy as np
import os
import time
import logging
import warnings
from preparing_data2 import PREPARE_DATA
from filter import FILTER
from models_answer import MODELS_ANSWER
logging.basicConfig(filename='result.log', level=logging.INFO)

# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")
# Making objects -------------------------------------------
prepare = PREPARE_DATA()
filter = FILTER()
answer = MODELS_ANSWER()
# Body -----------------------------------------------------
while True :
    if os.path.exists('data.csv'):
        print('Start reading file ...')
        data = pd.read_csv('data.csv')
        print('File read successfully')
        print('Starting to prepare data for first model ...')
        data1, target1 = prepare.ready(data)
        print('Data for first model is ready')
        print('Starting to prepare data for second model ...')
        data2, target2 = prepare.ready(data)
        print('Data for secoand model is ready')
        print('Start to filter Data for first model ...')
        data1, target1 = filter.ready(data1, target1, 1, 0)
        print('First models Data filterd successfully')
        print('Start to filter Data for second model ...')
        data2, target2 = filter.ready(data2, target2, 1, 0)
        print('Secoand models Data filterd successfully')
        print('Starting tarin for first model ...')
        y_pred1, y_pred_proba1 = answer.ready(data1 , target1)
        print('First model trained successfully and answer is ready')
        print('Starting tarin for secoand model ...')
        y_pred2, y_pred_proba2 = answer.ready(data2, target2)
        print('Secoand model trained successfully and answer is ready')
        print(f'First model predict : {y_pred1[-1]} and predict proba :{y_pred_proba1[0][1]}')
        print(f'Secoand model predict : {y_pred2[-1]} and predict proba :{y_pred_proba2[0][1]}')
        print('Start checking conditions ...')
        if y_pred1[0] == 1 and y_pred2[0] == 0 and y_pred_proba1[0][1] > 0.6 and y_pred_proba2[0][1] < 0.4 :
            print('Oriantation is UP with 70 percent probablity')
        else:
            if y_pred1[0] == 0 and y_pred2[0] == 1 and y_pred_proba2[0][1] > 0.6 and y_pred_proba1[0][1] < 0.4 :
                print('Oriantation is DOWN with 70 percent probablity')          
            else:
                print('Expert has no answer in current time')

    for i in range(4):
        print('*')
        time.sleep(1) 
    print('Waiting for next candle ...')                
    time.sleep(1)                