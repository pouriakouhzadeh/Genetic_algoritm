import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class PageCreator():
    def create(self, data, target, n, num_threads=1):
        num_rows, num_columns = data.shape
        new_df_list = []

        def process_row(i):
            new_row = data[i - n + 1:i + 1].values.flatten(order='F')
            return pd.DataFrame([new_row])

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(executor.map(process_row, range(n, num_rows)), total=num_rows - n, desc="Processing Rows", unit="row"))

        new_df = pd.concat(results, ignore_index=True)

        new_target = target[n:]
        new_target.reset_index(drop=True, inplace=True)

        return new_df, new_target



    def create_dataset(self, dataset, target, time_step=10):
        data = []
        time_step = int(time_step)
        for i in range(time_step, len(dataset)):
            a = dataset[(i - time_step + 1) : (i + 1)]
            data.append(a)
        data = np.array(data)    
        data = data.reshape(data.shape[0], -1)
        target = target[time_step:]
        return data, target

# مثال استفاده
# page_creator = PageCreator()
# new_df, new_target = page_creator.create(data, target, n)
