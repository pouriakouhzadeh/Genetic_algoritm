import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LogisticRegression, Perceptron, Ridge, SGDClassifier,
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier


import logging
import warnings

# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")

# Making objects -------------------------------------------

class ModelTester:
    def __init__(self, data, target, test_size=0.2, random_state=42):
        self.data = data
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.models = {
            'LogisticRegression': LogisticRegression(n_jobs=-1),
            'RandomForestClassifier': RandomForestClassifier(n_jobs=-1),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'BaggingClassifier': BaggingClassifier(),
            'ExtraTreesClassifier': ExtraTreesClassifier(n_jobs=-1),
            'VotingClassifier': VotingClassifier(estimators=[('lr', LogisticRegression(n_jobs=-1)), ('rf', RandomForestClassifier(n_jobs=-1)), ('gnb', GaussianNB())]),
            'SVC': SVC(probability=True),
            'LinearSVC': LinearSVC(),
            'NuSVC': NuSVC(probability=True),
            'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1),
            'RadiusNeighborsClassifier': RadiusNeighborsClassifier(n_jobs=-1),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'ExtraTreeClassifier': ExtraTreeClassifier(),
            'GaussianNB': GaussianNB(),
            'BernoulliNB': BernoulliNB(),
            'MLPClassifier': MLPClassifier(),
            'XGBClassifier': XGBClassifier(n_jobs=-1),
            'LGBMClassifier': LGBMClassifier(),
            'Perceptron': Perceptron(),
            'RidgeClassifier': RidgeClassifier(),
            'SGDClassifier': SGDClassifier(),
            'CalibratedClassifierCV': CalibratedClassifierCV(),
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
            'GaussianProcessClassifier': GaussianProcessClassifier(kernel=RBF()),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'MLPClassifier': MLPClassifier(),
            'SVC': SVC(probability=True),
            'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1),
            'RadiusNeighborsClassifier': RadiusNeighborsClassifier(n_jobs=-1),
            'VotingClassifier': VotingClassifier(estimators=[('lr', LogisticRegression(n_jobs=-1)), ('rf', RandomForestClassifier(n_jobs=-1)), ('gnb', GaussianNB()), ('svc', SVC(probability=True))]),
            'RandomForestClassifier_2': RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=self.random_state, n_jobs=-1),
            'GradientBoostingClassifier_2': GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=self.random_state),
            'AdaBoostClassifier_2': AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=self.random_state),
            'BaggingClassifier_2': BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=self.random_state),
            'ExtraTreesClassifier_2': ExtraTreesClassifier(n_estimators=100, criterion='gini', random_state=self.random_state, n_jobs=-1),
            'VotingClassifier_2': VotingClassifier(estimators=[('lr', LogisticRegression(n_jobs=-1)), ('rf', RandomForestClassifier(n_jobs=-1)), ('gnb', GaussianNB()), ('svc', SVC(probability=True))]),
            'SVC_2': SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=self.random_state),
            'SVC_3': SVC(kernel='poly', degree=3, C=1, gamma='scale', probability=True, random_state=self.random_state),
            'RandomForestClassifier_3': RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, random_state=self.random_state, n_jobs=-1),
            'GradientBoostingClassifier_3': GradientBoostingClassifier(learning_rate=0.05, n_estimators=50, max_depth=3, random_state=self.random_state),
            'AdaBoostClassifier_3': AdaBoostClassifier(n_estimators=100, learning_rate=0.5, algorithm='SAMME.R', random_state=self.random_state),
            'BaggingClassifier_3': BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=100, random_state=self.random_state),
            'ExtraTreesClassifier_3': ExtraTreesClassifier(n_estimators=50, criterion='entropy', max_depth=3, random_state=self.random_state, n_jobs=-1),
            'VotingClassifier_3': VotingClassifier(estimators=[('lr', LogisticRegression(n_jobs=-1)), ('rf', RandomForestClassifier(n_jobs=-1)), ('gnb', GaussianNB()), ('svc', SVC(probability=True))]),
            'LGBMClassifier_2': LGBMClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, random_state=self.random_state),
            'XGBClassifier_2': XGBClassifier(learning_rate=0.1, n_estimators=50, max_depth=3, random_state=self.random_state, n_jobs=-1),
            'MLPClassifier_2': MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, activation='relu', solver='adam', random_state=self.random_state),

           # Original models kept here...
            'LogisticRegression': LogisticRegression(n_jobs=-1),
            'RandomForestClassifier': RandomForestClassifier(n_jobs=-1),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),

            # Proposed new models added below
            'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=self.random_state, n_jobs=-1),
            'LGBMClassifier': LGBMClassifier(random_state=self.random_state, n_jobs=-1),
            'LogisticRegression_Pipeline': make_pipeline(StandardScaler(), LogisticRegression(random_state=self.random_state)),
            'VotingClassifier_Hard': VotingClassifier(estimators=[
                ('lr', LogisticRegression(n_jobs=-1, random_state=self.random_state)),
                ('rf', RandomForestClassifier(n_jobs=-1, random_state=self.random_state)),
                ('gb', GradientBoostingClassifier(random_state=self.random_state))
            ], voting='hard'),
            'VotingClassifier_Soft': VotingClassifier(estimators=[
                ('lr', LogisticRegression(n_jobs=-1, random_state=self.random_state)),
                ('rf', RandomForestClassifier(n_jobs=-1, random_state=self.random_state)),
                ('gb', GradientBoostingClassifier(random_state=self.random_state))
            ], voting='soft'),        
        
            'CatBoostClassifier': CatBoostClassifier(iterations=5000, depth=7, learning_rate=0.1, loss_function='Logloss', verbose=True),
            'StackingClassifier': StackingClassifier(estimators=[('rf', RandomForestClassifier(n_jobs=-1)), ('xgb', XGBClassifier(n_jobs=-1)), ('lgbm', LGBMClassifier(n_jobs=-1))], final_estimator=LogisticRegression()),    
            'LSTM': self.create_lstm_model(),
        
        }

    def create_lstm_model(self):
            """
            Creates and compiles an LSTM model
            """
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(self.data.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(25))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            return model


    def normalize_data(self):
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

    def test_models(self, cluster, count_feature, len_data, count_k_means, cluster_count, count_page_creator, log_file="result.log"):
        self.normalize_data()
        logging.basicConfig(filename=log_file, level=logging.INFO)
        max_acc = 0
        best_model = ""

        for model_name, model in self.models.items():
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    self.data, self.target, test_size=self.test_size, random_state=self.random_state
                )
                if model_name == 'LSTM':
                    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                if accuracy > max_acc:
                    max_acc = accuracy
                    best_model = model_name
            except Exception as e:
                print(f"Error in model : {model_name} ")

        if max_acc > 0.6:
            logging.info(f"best model: {best_model} and the acc: {max_acc:.4f}")
            logging.info(f"Len Data : {len_data} and number of features : {count_feature}, Total cluster : {count_k_means}, cluster number : {cluster_count}")
            logging.info(f"Number of pages : {count_page_creator}")
            logging.info("************************************************************************************************")
