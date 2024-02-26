from catboost import CatBoostClassifier
import logging
import warnings
# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")

class BestCatBoost :
    def TrainModels (self, X_train, y_train) :
        model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.01, loss_function='Logloss', verbose=True)
        model.fit(X_train, y_train)
        return model
 