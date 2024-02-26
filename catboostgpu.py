from catboost import CatBoostClassifier
from ACC_CALC import Acc_Calculator
import logging
import warnings

# Silent alerts
warnings.filterwarnings("ignore")

class CatBoostGpu:
    def TrainModels(self, X_train, y_train, X_test, y_test, Stage, i):
        logging.basicConfig(filename="Best_models_ACC.log", level=logging.INFO)

        # حذف پارامتر task_type به منظور استفاده از CPU
        model_1 = CatBoostClassifier(iterations=5000, depth=16, learning_rate=0.01, loss_function='Logloss', verbose=True)

        ACC_CALC = Acc_Calculator()
        
        logging.info("*****************************************************************************")
        logging.info(f"Stage = {Stage}, Data number is : {i}")
        
        model_1.fit(X_train, y_train)
        predictions = model_1.predict(X_test)
        predictions_proba = model_1.predict_proba(X_test)
        
        A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test, predictions)
        A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test, predictions_proba, 0.65)
        A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test, predictions_proba, 20)
        
        print(f"model_1 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
        logging.info(f"model_1 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")