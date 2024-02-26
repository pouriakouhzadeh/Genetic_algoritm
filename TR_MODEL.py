from timeconvert import TimeConvert
from selecttimetodelete import SelectTimeToDelete
from preparing_data import PREPARE_DATA
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
    def Train(self, data, depth, page, feature, QTY, iter, Thereshhold, primit_hours=[]):
        print(f"depth:{depth}, page:{page}, features:{feature}, QTY:{QTY}, iter:{iter}, Thereshhold:{Thereshhold}, primit_hours:{primit_hours}")
        data = TimeConvert().exec(data)
        data = data[-QTY:]
        data.reset_index(inplace=True, drop=True)
        primit_hours = SelectTimeToDelete().exec(data, primit_hours)
        data, target, primit_hours = PREPARE_DATA().ready(data, primit_hours)
        data, target = PageCreator().create_dataset(data, target, page)
        primit_hours = primit_hours[page:]
        data, target = DeleteRow().exec(data, target, primit_hours)
        data = FeatureSelection().select(data, target, feature).copy()
        data = normalize_data(data)
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=1234)
        # print(f"primit_hours  = {primit_hours}")
        model = CatBoostClassifier(
            iterations=iter,
            depth=depth,
            learning_rate=0.01,
            loss_function='Logloss',
            verbose=False,
            task_type='CPU'
        )
        model.fit(data, target)
        predictions_proba = model.predict_proba(X_test)
        ACC , wins , loses = ACC_BY_THRESHHOLD(self, y_test, predictions_proba, Thereshhold)

        return ACC , wins , loses

