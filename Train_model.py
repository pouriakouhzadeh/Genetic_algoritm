from timeconvert import TimeConvert
from selecttimetodelete import SelectTimeToDelete
from preparing_data_for_train import PREPARE_DATA_FOR_TRAIN
from PAGECREATOR import PageCreator
from deleterow import DeleteRow
from FEATURESELECTION import FeatureSelection
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled


def ACC_BY_THRESHHOLD(self, y_test, predictions_proba, TH):
    predictions_proba = pd.DataFrame(predictions_proba)
    predictions_proba.reset_index(inplace = True ,drop =True)
    y_test.reset_index(inplace = True ,drop =True)
    TH = TH / 100
    try :
        wins = 0
        loses = 0
        for i in range(len(y_test)) :
            if predictions_proba[1][i] > TH :
                if y_test['close'][i] == 1 :
                    wins = wins + 1
                else :
                    loses = loses + 1    
            if predictions_proba[0][i] > TH :
                if y_test['close'][i] == 0 :
                    wins = wins + 1
                else :
                    loses = loses + 1       
        # logging.info(f"Thereshhold wins = {wins}, Thereshhold loses = {loses}")
        return ( (wins * 100) / (wins + loses) , wins, loses)  
    except :
        return 0, 0, 0
    
class TrainModels:
    def Train(self, data ,depth ,page ,feature ,iter, Thereshhold):
        page = int(page)
        data = TimeConvert().exec(data)
        # data = data[-QTY:]
        data.reset_index(inplace = True, drop = True)
        # Forbidden_list = SelectTimeToDelete().exec(data, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,0])
        data, target = PREPARE_DATA_FOR_TRAIN().ready(data)
        data, target = PageCreator().create_dataset(data, target, page)
        # Forbidden_list = Forbidden_list[page:]
        # data, target = DeleteRow().exec(data, target,Forbidden_list)
        data = FeatureSelection().select(data, target, feature).copy()
        data = normalize_data(data)
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=1234)
        print(f"iter:{iter},depth:{depth} ,page:{page} ,feature:{feature} ,Thereshhold:{Thereshhold},tail:{len(data)}")
        model = CatBoostClassifier(
            iterations=iter,
            depth=depth,
            learning_rate=0.01,
            loss_function='Logloss',
            verbose=False,
            task_type= 'CPU'
            )
        model.fit(X_train, y_train)
        predictions_proba = model.predict_proba(X_test)
        ACC , wins , loses = ACC_BY_THRESHHOLD(self, y_test, predictions_proba, Thereshhold)

        return ACC , wins , loses
