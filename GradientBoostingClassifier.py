import pandas as pd
import numpy as np
# Removed unnecessary imports to declutter code
from sklearn.model_selection import train_test_split, GridSearchCV
from Clean_data import CLEAN_DATA
from preparing_data import PREPARE_DATA
from PAGECREATOR import PageCreator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Assuming that preprocessing functions from the initial code block are valid and can be used here

# Preprocessing data
data_source = pd.read_csv('EURUSD_M15.csv')
data = data_source[190000:]
data = CLEAN_DATA().clear(data)
data, target = PREPARE_DATA().ready(data)
data = PageCreator().create(data, target, 4)

# Feature Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.2, random_state=42)

# Feature selection
feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
feature_selector.fit(X_train, y_train)
selector = SelectFromModel(feature_selector, prefit=True)
X_train_select = selector.transform(X_train)
X_test_select = selector.transform(X_test)

# Model tuning and Cross-Validation
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
clf = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
clf.fit(X_train_select, y_train)

print(f"Best parameters: {clf.best_params_}")
best_model = clf.best_estimator_

# Prediction and Evaluation
predictions = best_model.predict(X_test_select)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy}")