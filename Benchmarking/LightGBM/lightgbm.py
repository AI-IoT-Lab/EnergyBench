import os
import glob
import time
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import islice
import argparse

from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg

import warnings
warnings.filterwarnings('ignore') 

# Data pipelining
def get_batched_data_fn(sub_df,
    batch_size: int = 128, 
    context_len: int = 168, 
    horizon_len: int = 24):
    
    examples = defaultdict(list)
    num_examples = 0
    for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
      num_examples += 1
      #examples["country"].append(country)
      examples["inputs"].append(sub_df["y"][start:(context_end := start + context_len)].tolist())
      #examples["gen_forecast"].append(sub_df["gen_forecast"][start:context_end + horizon_len].tolist())
      #examples["week_day"].append(sub_df["week_day"][start:context_end + horizon_len].tolist())
      examples["outputs"].append(sub_df["y"][context_end:(context_end + horizon_len)].tolist())
      examples['inputs_ts'].append(sub_df.index[start:(context_end := start + context_len)])
      examples["outputs_ts"].append(sub_df.index[context_end:(context_end + horizon_len)])

    return examples


# Benchmark
batch_size = 32
context_len = 168
horizon_len = 24

def process_building(df):
   #  input_data = get_batched_data_fn(df, batch_size=32)
    building_name = df.columns[0]
    df.columns = ['y']
    input_data = get_batched_data_fn(df, batch_size=500)
    # print(input_data)
    
    windows_all = []
    counter = 1
    for inputs_ts, inputs, outputs_ts, outputs in zip(input_data['inputs_ts'], 
                                                      input_data['inputs'], 
                                                      input_data['outputs_ts'], 
                                                      input_data['outputs']):
        
        input_df = pd.DataFrame({'timestamp': inputs_ts, 
                                 'target': inputs})
        
        output_df = pd.DataFrame({'timestamp': outputs_ts, 
                                 'target': outputs})
        combined = pd.concat([input_df, output_df], axis=0)
        combined['item_id'] = str(building_name) + '_' + str(counter)
        combined['item_id_no'] = counter
        counter += 1
        windows_all.append(combined)
        
    windows_all_df = pd.concat(windows_all)
    windows_all_df.timestamp = pd.to_datetime(windows_all_df.timestamp)
    windows_all_df.set_index('timestamp', inplace=True)

    return windows_all_df


def process_file(filename):
    df = pd.read_parquet(filename)
    save_filename = os.path.basename(filename).split(".")[0]


    print(datetime.now(), df.shape, flush=True)
            

    # if df.shape[1] < 2:
    #     return None
        
    print(datetime.now(), df.shape, flush=True)

    results_all = []
    c = 0
    j = 0
    lag = 168 
    for building_name in df.columns:
        print(f'{datetime.now()} {c} / {len(df.columns)} {building_name}', 'Dataset: ', dataset, flush=True)
        df1 = df[[building_name]]
        print(datetime.now(), c, '/', len(df.columns), building_name, df1.shape, "Dataset: ", save_filename, flush=True)
        df1 = df1.loc[df1.first_valid_index():]
        print(datetime.now(), c, '/', len(df.columns), building_name, df1.shape, "Dataset: ", save_filename, flush=True)
        
        mid = len(df1) // 2
        training_set = df1.iloc[:mid, :]
        test_set = df1.iloc[mid:, :]
        training_set = training_set.fillna(0)
        test_set = test_set.fillna(0)

        print(f'fine-tune set date range: {training_set.index[0]} {training_set.index[-1]}, '
        f'test set date range: {test_set.index[0]} {test_set.index[-1]}')

        if (len(training_set) <= 192) and (len(test_set) <= 192):
            c += 1
            continue
        if (c % 500 == 0) and (c != len(df.columns) - 1):
            results_all = []
            j += 1

        windowed_df_train = process_building(training_set)
        windowed_df_test = process_building(test_set)

        forecaster = ForecasterAutoreg(
                    regressor        = LGBMRegressor(max_depth=-1, n_estimators=100, n_jobs=24, verbose=-1),
                    lags             = 168
                )
        forecaster.fit(y= windowed_df_train['target'],
    # exog = training_set[[key for key in training_set.keys() if key != 'target' and key != 'item_id']]
            )

        p = []
        for i in windowed_df_test.item_id_no.unique():#(pred_days):
            # i -= 1           
            seq_ptr =lag + 24 * i
        
            df_test = windowed_df_test[windowed_df_test.item_id_no == i]
            last_window  = df_test.iloc[0:168]#windowed_df_test.iloc[i*192:(i*192)+168]#[seq_ptr - lag : seq_ptr]
            ground_truth = df_test.iloc[168:192]#windowed_df_test.iloc[i*168:(i*168)+24]#[seq_ptr : seq_ptr + 24]

        
            predictions = forecaster.predict(
                steps       = 24,
                last_window = last_window['target'],
                # exog        = ground_truth[[key for key in test_set.keys() if key != 'target' and key != 'item_id']]
            )
            # p.append(predictions)
            res = ground_truth.copy()
            res = res[['target']]
            # print(res)
            res.columns = ['y_true']
            res = res.reset_index()
            res.insert(2, 'y_pred', predictions.reset_index()['pred'])
            res.set_index('timestamp', inplace=True)
            # res['y_pred'] = predictions
            p.append(res)
        res = pd.concat(p)
        res['building'] = building_name
        results_all.append(res)
        c += 1 
        if c % 5 == 0:
            print(datetime.now(), 'Saving...')
            results_all_df = pd.concat(results_all)
            results_all_df.to_csv(f'Results/lightgbm/forecasts/{dataset}/{save_filename}_{j}.csv')          
        # if i == 2:
        #    break
        #break
        
    results_all_df = pd.concat(results_all)
    results_all_df.to_csv(f'Results/lightgbm/forecasts/{dataset}/{save_filename}_{j}.csv')
    return results_all_df


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = "zero-shot using lightgbm")

    parser.add_argument("--dataset", type = str, help = "Dataset name")

    args = parser.parse_args()
    dataset = args.dataset

    files_list = glob.glob(f'Residential/{dataset}/*.parquet')
    os.makedirs(f'Results/lightgbm/forecasts/{dataset}/', exist_ok = True)
    os.makedirs(f'Results/lightgbm/results/{dataset}/', exist_ok = True)
    
    for filename in files_list:
        print(datetime.now(), filename)
        results = process_file(filename)
        if results is not None:
            # results.to_csv(f'forecasts/{dataset}/{os.path.basename(filename)}')
            print("Done")
        print('')
