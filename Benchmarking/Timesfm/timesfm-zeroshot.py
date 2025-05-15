import os
import glob
import time
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
# import jax
# jax.config.update('jax_platform_name', 'cpu')
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import timesfm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
import argparse


tfm = timesfm.TimesFm(
    context_len=512 ,
    horizon_len=24,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend='cpu'
)
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

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
      examples['inputs_ts'].append(sub_df["ds"][start:(context_end := start + context_len)].tolist())
      examples["outputs_ts"].append(sub_df["ds"][context_end:(context_end + horizon_len)].tolist())

    #print(num_examples)
  
    def data_fn():
        for i in range(1 + (num_examples - 1) // batch_size):
            yield {k: v[(i * batch_size) : ((i + 1) * batch_size)] for k, v in examples.items()}
  
    return data_fn

# # Benchmark
# batch_size = 32
# context_len = 168
# horizon_len = 24

def process_building(df):
   #  input_data = get_batched_data_fn(df, batch_size=32)
    input_data = get_batched_data_fn(df, batch_size=500)

    metrics = defaultdict(list)
    results_all = []
    for i, example in enumerate(input_data()):
        #print(datetime.now(), i)
        raw_forecast, _ = tfm.forecast(inputs=example["inputs"], freq=[0] * len(example["inputs"]))

        #print(f"Batch {i+1}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, Time: {end_time - start_time:.2f}s")
        for ts, y_true, y_pred in zip(example['outputs_ts'], example['outputs'], raw_forecast):
            res_df = pd.DataFrame({'ts': ts, 'y_true': y_true,'y_pred': y_pred})
            results_all.append(res_df)
        
    results_all_df = pd.concat(results_all)
    return results_all_df

def process_file(filename, dataset):
    df = pd.read_parquet(filename)

    print(datetime.now(), df.shape, flush=True)

    results_all = []
    i = 0
    j = 0
    for building_name in df.columns:
        # print(datetime.now(), building_name, flush=True)
        # df1 = df[[building_name]]
        print(datetime.now(), i, '/', len(df.columns), building_name, "Dataset: ", dataset, flush=True)
        df1 = df[[building_name]]#.head(24*200)
        print(datetime.now(), i, '/', len(df.columns), building_name, "Dataset: ", dataset, df1.shape, flush=True)
        df1 = df1.loc[df1.first_valid_index():]
        print(datetime.now(), i, '/', len(df.columns), building_name, "Dataset: ", dataset, df1.shape, flush=True)

        if len(df1) <= 192:
            continue
        df1 = df1.reset_index()
        df1.columns = ['ds', 'y']

        #df1['y'] = MinMaxScaler().fit_transform(df1.y)
        df1['y'] = minmax_scale(df1['y'])

        if (i % 500 == 0) and (i != len(df.columns) - 1):
            results_all = []
            j += 1

         

        res = process_building(df1)
        res['building'] = building_name
        results_all.append(res)
        i+=1
        if i % 5 == 0:
            print(datetime.now(), 'Saving...')
            results_all_df = pd.concat(results_all)
            results_all_df.to_csv(f'Results/timesfm/forecasts/{dataset}/{save_filename}_{j}.csv', index=False)  
        
        
        # if i == 2:
        #    break
        #break
        
    results_all_df = pd.concat(results_all)
    results_all_df.to_csv(f'Results/timesfm/forecasts/{dataset}/{save_filename}_{j}.csv', index=False) 
    print("Zero-shot completed for ", dataset, ".") 
    return results_all_df


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = "zero-shot using timesfm")

    parser.add_argument("--dataset", type = str, help = "Dataset name")

    args = parser.parse_args()
    dataset = args.dataset


    files_list = glob.glob(f'Residential/{dataset}/*.parquet')
    # filename = '/home/user/New_Buildings_Datasets/Mathura_and_Bareilly/dataverse_files/processed/Mathura/Mathura_2019.csv'
    
    
    os.makedirs(f'Results/timesfm/forecasts/{dataset}/', exist_ok = True)
    os.makedirs(f'Results/timesfm/results/{dataset}/', exist_ok = True)
    
    for filename in files_list:
        save_filename = os.path.basename(filename).split(".")[0]
        print(datetime.now(), filename)
        results = process_file(filename, dataset)
        if results is not None:
            # results.to_csv(f'Results/timesfm/forecasts/{dataset}/{save_filename}.csv')
            print('Done')
        print('')
