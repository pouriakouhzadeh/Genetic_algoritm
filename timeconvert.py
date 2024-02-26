import pandas as pd
import numpy as np
class TimeConvert :
    
    def exec(self, data) :
        # Hour = pd.DataFrame()
        Hour = []
        for i in range (len(data)) :
            temp =  pd.to_datetime(data.iloc[i]["time"])
            Hour.append(temp.hour)

        Hour = pd.DataFrame(Hour)   
        data.reset_index(inplace = True, drop = True)
        data["Hour"] = Hour[0]
        data.drop(columns = 'time', inplace = True)
        return data
