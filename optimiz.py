from scipy.optimize import minimize
import pandas as pd
from Train_model import TrainModels
import logging
from tqdm import tqdm

logging.basicConfig(filename='optimazation.log', level=logging.INFO, format='%(asctime)s - %(message)s')

                    #depth ,page ,feature ,QTY ,iter, Thereshhold
def model_prediction(depth ,page ,feature ,iter, Thereshhold):

    Data_EURUSD =  pd.read_csv("EURUSD60.csv")
    Data_AUDCAD =  pd.read_csv("AUDCAD60.csv")
    Data_AUDCHF =  pd.read_csv("AUDCHF60.csv")
    Data_AUDNZD =  pd.read_csv("AUDNZD60.csv")
    Data_AUDUSD =  pd.read_csv("AUDUSD60.csv")
    Data_EURAUD =  pd.read_csv("EURAUD60.csv")
    Data_EURCHF =  pd.read_csv("EURCHF60.csv")
    Data_EURGBP =  pd.read_csv("EURGBP60.csv")
    Data_GBPUSD =  pd.read_csv("GBPUSD60.csv")
    Data_USDCAD =  pd.read_csv("USDCAD60.csv")
    Data_USDCHF =  pd.read_csv("USDCHF60.csv")

    tail_size = 3000

    Tail_EURUSD = Data_EURUSD.tail(tail_size)
    Tail_AUDCAD = Data_AUDCAD.tail(tail_size)
    Tail_AUDCHF = Data_AUDCHF.tail(tail_size)
    Tail_AUDNZD = Data_AUDNZD.tail(tail_size)
    Tail_AUDUSD = Data_AUDUSD.tail(tail_size)
    Tail_EURAUD = Data_EURAUD.tail(tail_size)
    Tail_EURCHF = Data_EURCHF.tail(tail_size)
    Tail_EURGBP = Data_EURGBP.tail(tail_size)
    Tail_GBPUSD = Data_GBPUSD.tail(tail_size)
    Tail_USDCAD = Data_USDCAD.tail(tail_size)
    Tail_USDCHF = Data_USDCHF.tail(tail_size)

    currency_pairs = [Tail_EURUSD, Tail_AUDCAD, Tail_AUDCHF, Tail_AUDNZD, Tail_AUDUSD, Tail_EURAUD, Tail_EURCHF, Tail_EURGBP, Tail_GBPUSD, Tail_USDCAD, Tail_USDCHF]

    # Initialize lists to store results
    all_ACC = []
    all_wins = []
    all_loses = []

    # Loop through each currency pair
    for tail_data in currency_pairs:
                                                        # depth ,page ,feature ,QTY ,iter, Thereshhold
        ACC, wins, loses = TrainModels().Train(tail_data, depth, page, feature, iter, Thereshhold)
        
        # Append results to the lists
        all_ACC.append(ACC)
        all_wins.append(wins)
        all_loses.append(loses)

    total_ACC = sum(all_ACC)/len(currency_pairs)
    total_wins = sum(all_wins)
    total_loses = sum(all_loses)

    logging.info(f"total ACC = {total_ACC}, total wins = {total_wins}, total loses = {total_loses}")
    if (total_loses+total_wins < 100):
        return (0)
    else :
        return total_ACC


# تابع هدف برای minimize کردن
def objective_function(parameters):
    # مقدار بازگشتی تابع هدف برابر با معکوس عملکرد مدل می‌باشد، زیرا minimize کرده‌ایم
    return -model_prediction(*parameters)

# ,depth = 2 ,page = 2 ,feature = 30 ,QTY = 1000 ,iter = 100, Thereshhold=65
            # depth ,page ,feature ,  iter, Thereshhold
min_values = [2,     2,    30,        100,  55]
max_values = [16,    10,   200,       10000,70]




with tqdm(total=100, desc="Optimizing", position=0, leave=True) as pbar:
    def callback(xk):
        pbar.update()


# بهینه‌سازی
result = minimize(objective_function, x0=[2,     2,    30,        100,  55], bounds=list(zip(min_values, max_values)), method='SLSQP')

# نمایش نتیجه
print("Optimal Parameters:", result.x)
print("Optimal Performance:", -result.fun)  # منفی گرفته‌ام چرا که minimize کرده‌ایم

# ذخیره نتیجه ی نهایی در یک فایل
final_result = {"Optimal Parameters": result.x, "Optimal Performance": -result.fun}
final_result_df = pd.DataFrame.from_dict(final_result, orient='index', columns=['Values'])
final_result_df.to_csv("final_result_optimaization.csv")