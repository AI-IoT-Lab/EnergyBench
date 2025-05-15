import os
import torch
import matplotlib.pyplot as plt
import glob
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from tqdm.autonotebook import tqdm
import matplotlib.dates as mdates
from itertools import islice
from collections import defaultdict
import gluonts
from datetime import datetime
import sklearn.metrics
import numpy as np
import argparse

from gluonts.evaluation import make_evaluation_predictions, Evaluator
from autogluon.timeseries.metrics import TimeSeriesScorer
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

TimeSeriesScorer.greater_is_better_internal = True

class NRMSE(TimeSeriesScorer):
   greater_is_better_internal = True
   optimum = 0.0

   def compute_metric(self, data_future, predictions, target, **kwargs):
      # return sklearn.metrics.root_mean_squared_error(y_true=data_future[target], y_pred=predictions["mean"]) / data_future[target].mean()
       return np.sqrt(np.mean(np.square(data_future[target] - predictions["mean"]))) / data_future[target].mean()

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

def forecast_building(df):
    # torch.cuda.empty_cache()
    # Set numerical columns as float32
    for col in df.columns:
        # Check if column is not of string type
        if df[col].dtype != 'object' and pd.api.types.is_string_dtype(df[col]) == False:
            df[col] = df[col].astype('float32')
    
    # Create the Pandas
    # dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")
    dataset = TimeSeriesDataFrame(df.reset_index())

    backtest_dataset = dataset
    prediction_length = 24  # Define your prediction length. We use 24 here since the data is of hourly frequency
    num_samples = 100

    train_data, test_data = backtest_dataset.train_test_split(prediction_length)

    predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data,
    hyperparameters={
        "Naive": {}
    },
    skip_model_selection=True,
    verbosity=0)
    predictions = predictor.predict(train_data)
    # agg_metrics = predictor.evaluate(backtest_dataset, metrics=["RMSE", "MSE", "MAE", "MSE", "MAPE", "SMAPE", NRMSE(), "SQL"])
    try:
        agg_metrics = predictor.evaluate(backtest_dataset, metrics=["RMSE", "MSE", "MAE", "MSE", "MAPE", "SMAPE", NRMSE(), "SQL"])
    except Exception as E:
        agg_metrics = None

    res_all = pd.DataFrame(test_data[test_data.index.isin(predictions.index)].target)
    res_all.columns = ['y_true']
    res_all.insert(1, 'y_pred', predictions['mean'])
    res_all_df = res_all.reset_index().drop('item_id', axis = 1).sort_values('timestamp')
    
    # return res_all_df, agg_metrics, ts_metrics 
    return res_all_df, agg_metrics

def process_building(df): 
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
    # windows_all_df.to_csv('test.csv')

    # res, agg_metrics, ts_metrics = forecast_building(windows_all_df)
    res, agg_metrics = forecast_building(windows_all_df)
    # return res, agg_metrics, ts_metrics
    return res, agg_metrics

def process_file(filename, dataset):
    df = pd.read_parquet(filename)
    save_filename = os.path.basename(filename).split(".")[0]

        
    print(datetime.now(), df.shape, flush=True)

    res_all = []
    agg_metrics_all = []
    # ts_metrics_all = []
    
    i = 0
    j = 0
    for building_name in df.columns:
        print(datetime.now(), i, '/', len(df.columns), building_name, "Dataset: ", dataset, flush=True)
        df1 = df[[building_name]]#.head(24*200)
        print(datetime.now(), i, '/', len(df.columns), building_name, df1.shape, "Dataset: ", save_filename, flush=True)
        df1 = df1.loc[df1.first_valid_index():]
        print(datetime.now(), i, '/', len(df.columns), building_name, df1.shape, "Dataset: ", save_filename, flush=True)

        if len(df1) <= 192:
            i += 1
            continue

        if (i % 500 == 0) and (i != len(df.columns) - 1):
            res_all = []
            agg_metrics_all = []
            j += 1

        # res, agg_metrics, ts_metrics = process_building(df1)
        res, agg_metrics = process_building(df1)
        res['building'] = building_name
        res['filename'] = filename
        res_all.append(res)
        if agg_metrics is None:
            i += 1
            continue
        
        agg_metrics_df = pd.DataFrame([agg_metrics])
        agg_metrics_df.insert(0, 'building', building_name)
        agg_metrics_df.insert(0, 'filename', filename)
        agg_metrics_all.append(agg_metrics_df)

        i += 1
        if i % 5 == 0:
            print(datetime.now(), 'Saving...')
            res_all_df = pd.concat(res_all).round(6)
            res_all_df = res_all_df.reset_index()
            res_all_df = res_all_df.rename(columns={res_all_df.columns[0]: "timestamp" })
            res_all_df.to_csv(f'/media/user/DATA/Results/naive/forecasts/{dataset}/{save_filename}_{j}.csv', index=False)            
   

            agg_metrics_all_df = pd.concat(agg_metrics_all).round(6)            
            agg_metrics_all_df.to_csv(f'/media/user/DATA/Results/naive/results/{dataset}/agg_metrics_{save_filename}_{j}.csv', index=False)            
    
    
    res_all_df = pd.concat(res_all).round(6)
    res_all_df = res_all_df.reset_index()
    res_all_df = res_all_df.rename(columns={res_all_df.columns[0]: "timestamp" })
    res_all_df.to_csv(f'/media/user/DATA/Results/naive/forecasts/{dataset}/{save_filename}_{j}.csv', index=False)            
         

    agg_metrics_all_df = pd.concat(agg_metrics_all).round(6)   
    agg_metrics_all_df.to_csv(f'/media/user/DATA/Results/naive/results/{dataset}/agg_metrics_{save_filename}_{j}.csv', index=False)         
    print("zero-shot completed for - ", dataset)  

    return res_all_df, agg_metrics_all_df



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = "zero-shot using naive")

    parser.add_argument("--dataset", type = str, help = "Dataset name")

    args = parser.parse_args()
    dataset = args.dataset

    files_list = glob.glob(f'/media/user/DATA/Dataset_V0.0/Energy-Load-Profiles/Hourly/Residential/{dataset}/*.parquet')

    os.makedirs(f'/media/user/DATA/Results/naive/forecasts/{dataset}/', exist_ok = True)
    os.makedirs(f'/media/user/DATA/Results/naive/results/{dataset}/', exist_ok = True)

    for filename in files_list:

	print(datetime.now(), filename)
	results = process_file(filename, dataset)

        print('')
