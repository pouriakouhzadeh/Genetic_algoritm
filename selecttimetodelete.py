import pandas as pd
import numpy as np
class SelectTimeToDelete :
    
    def exec(self, data, n=[]):
        Forbidden = pd.DataFrame()

        for i in range(len(data)):
            if data.at[i, "Hour"] in n:
                Forbidden.at[i, 'Marked'] = 1
            else:
                Forbidden.at[i, 'Marked'] = np.nan

        return Forbidden

