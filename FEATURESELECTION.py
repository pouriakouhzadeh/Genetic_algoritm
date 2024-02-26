from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import warnings

# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")

# Making objects -------------------------------------------
class FeatureSelection :

    def select(self, data, target, n) :
        n = int(n)
        model = RandomForestClassifier()
        sfm = SelectFromModel(estimator = model, threshold=-np.inf, max_features = n)
        X_selected = sfm.fit_transform(data, target)

        return X_selected
