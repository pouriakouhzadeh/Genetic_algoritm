from scipy.optimize import minimize
import pandas as pd
from Train_model import TrainModels
import logging
from tqdm import tqdm
import concurrent.futures

logging.basicConfig(filename='optimization.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def read_data(file_name, min_tail_size):
    data = pd.read_csv(file_name)
    
    # Ensure tail_size is not less than 1000
    tail_size = max(min_tail_size, 1000)
    
    tail_data = data.tail(tail_size)
    if tail_data.empty:
        # Handle the case of an empty DataFrame, you may choose to return None or take other actions
        return None
    return tail_data

def model_prediction(parameters, currency_pair, min_required_tail_size):
    depth, page, feature, iter, Thereshhold, _ = map(int, parameters)  # Cast parameters to integers
    
    if currency_pair is None:
        return 0  # Handle the case of an empty DataFrame
    
    # Read the content of currency_pair from the file
    currency_data = read_data(currency_pair, min_required_tail_size)
    
    if currency_data is None:
        return 0  # Handle the case of an empty DataFrame
    
    # Log the parameters and currency pair
    logging.info(f'Model Prediction - Parameters: {parameters}, Currency Pair: {currency_pair}')
    
    # Train the model and get results
    model_results = TrainModels().Train(currency_data, depth, page, feature, iter, Thereshhold)
    
    # Extract necessary information from the results
    ACC, wins, loses = model_results
    
    # Log the model accuracy
    logging.info(f'Model Prediction - Accuracy: {ACC}, Wins: {wins}, Loses: {loses}')
    
    if (wins + loses < 100):
        return 0
    else:
        return ACC

def objective_function(parameters, currency_pairs):
    total_ACC = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda args: model_prediction(parameters, *args), currency_pairs))
        total_ACC = sum(results)
    return -total_ACC / len(currency_pairs)

def optimize_parameters(file_names, min_required_tail_size, bounds, bounds_dtype):
    with tqdm(total=100, desc="Optimizing", position=0, leave=True) as pbar:
        def callback(xk):
            pbar.update()

        # Load currency pairs data within the optimization loop
        currency_pairs = [(file_name, min_required_tail_size) for file_name in file_names]

        # Perform optimization with bounds specifying integer constraints
        result = minimize(objective_function, x0=[2, 2, 30, 100, 55, 3000],  # Initial values for parameters
                          bounds=bounds,
                          method='SLSQP', args=(currency_pairs,), callback=callback,
                          options={'disp': True},
                          constraints=({'type': 'eq', 'fun': lambda x: [dtype(x[i]) for i, dtype in enumerate(bounds_dtype)]}))

    # Display and save results
    print("Optimal Parameters:", result.x)
    print("Optimal Performance:", -result.fun)
    
    final_result = {"Optimal Parameters": result.x, "Optimal Performance": -result.fun}
    final_result_df = pd.DataFrame.from_dict(final_result, orient='index', columns=['Values'])
    final_result_df.to_csv("final_result_optimization.csv")

if __name__ == "__main__":
    # Adjust file names as needed
    file_names = ["EURUSD60.csv", "AUDCAD60.csv", "AUDCHF60.csv", "AUDNZD60.csv", "AUDUSD60.csv",
                  "EURAUD60.csv", "EURCHF60.csv", "EURGBP60.csv", "GBPUSD60.csv", "USDCAD60.csv", "USDCHF60.csv"]
    
    # Set the minimum required tail size (e.g., 1000)
    min_required_tail_size = 1000

    # Specify integer bounds for the last three parameters (iter, Thereshhold, tail_size)
    bounds = [(2, 16), (2, 10), (30, 200), (100, 10000), (55, 70), (1000, 5000)]  # Adjust the upper and lower bounds for tail_size
    
    # Specify the data type for each parameter (int for integer variables)
    bounds_dtype = [int, int, int, int, int, int]

    optimize_parameters(file_names, min_required_tail_size, bounds, bounds_dtype)
