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

data_source = pd.read_csv('EURUSD_M15.csv')
# data = data_source[180000:]
data = CLEAN_DATA().clear(data_source)
data, target = PREPARE_DATA().ready(data)
data, target = PageCreator().create(data, target, 5)
# feature_selector = FeatureSelection()
# data_1 = feature_selector.select(data, target, 142).copy()

