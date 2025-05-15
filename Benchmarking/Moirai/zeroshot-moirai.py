import os, sys
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
import argparse

sys.path.insert(0, os.path.dirname(os.path.join(os.getcwd(), "uni2ts")))

from uni2ts.eval_util.plot import plot_single
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.eval_util.evaluation import evaluate_model

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

    # Set numerical columns as float32
    for col in df.columns:
        # Check if column is not of string type
        if df[col].dtype != 'object' and pd.api.types.is_string_dtype(df[col]) == False:
            df[col] = df[col].astype('float32')
    
    # Create the Pandas
    dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")

    backtest_dataset = dataset
    prediction_length = 24  # Define your prediction length. We use 24 here since the data is of hourly frequency
    num_samples = 100

    model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
    prediction_length=prediction_length,
    context_length=168,
    patch_size='auto',
    target_dim=1,
    feat_dynamic_real_dim=backtest_dataset.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=backtest_dataset.num_past_feat_dynamic_real,
)
    predictor = model.create_predictor(batch_size=32, device="cuda")

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=backtest_dataset,
        predictor=predictor,
        num_samples=num_samples
    )



    forecasts = list(forecast_it)
    tss = list(ts_it)

    evaluator = Evaluator()
    agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))
    # agg_metrics, ts_metrics = evaluator(tss, forecasts)

    res_all = []
    for ts, fc in zip(tss, forecasts):
        res = ts[ts.index.isin(fc.index)]
        res.columns = ['y_true']
        res.insert(1, 'y_pred', fc.median)        
        res.sort_index()
        res_all.append(res)
        #print(gt.shape)
        #break
    res_all_df = pd.concat(res_all)#.sort_index()
    return res_all_df, agg_metrics, ts_metrics 

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
    #windows_all_df.to_csv('test.csv')

    res, agg_metrics, ts_metrics = forecast_building(windows_all_df)
    return res, agg_metrics, ts_metrics

def process_file(filename, dataset):
    df = pd.read_parquet(filename)
    save_filename = os.path.basename(filename).split(".")[0]

        
    print(datetime.now(), df.shape, flush=True)

    res_all = []
    agg_metrics_all = []
    ts_metrics_all = []
    
    i = 0
    j = 0
    for building_name in df.columns:
        print(datetime.now(), i, '/', len(df.columns), building_name, "Dataset: ", dataset, flush=True)
        df1 = df[[building_name]]#.head(24*200)
        print(datetime.now(), i, '/', len(df.columns), building_name, "Dataset: ", save_filename, df1.shape, flush=True)
        df1 = df1.loc[df1.first_valid_index():]
        print(datetime.now(), i, '/', len(df.columns), building_name, "Dataset: ", save_filename, df1.shape, flush=True)

        if len(df1) <= 192:
            continue

        if (i % 500 == 0) and (i != len(df.columns) - 1):
            res_all = []
            agg_metrics_all = []
            ts_metrics_all = []
            j += 1

        res, agg_metrics, ts_metrics = process_building(df1)
        res['building'] = building_name
        res['filename'] = filename
        res_all.append(res)
        
        ts_metrics.insert(0, 'building', building_name)
        ts_metrics.insert(0, 'filename', filename)
        ts_metrics = ts_metrics.sort_values(['forecast_start'])
        ts_metrics_all.append(ts_metrics)
        
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
            res_all_df.to_csv(f'/home/user/energygpt/Results/moirai/forecasts/{dataset}/{save_filename}_{j}.csv', index=False)            

            ts_metrics_all_df = pd.concat(ts_metrics_all).round(6)
            ts_metrics_all_df.to_csv(f'/home/user/energygpt/Results/moirai/results/{dataset}/ts_metrics_{save_filename}_{j}.csv', index=False)            

            agg_metrics_all_df = pd.concat(agg_metrics_all).round(6)            
            agg_metrics_all_df.to_csv(f'/home/user/energygpt/Results/moirai/results/{dataset}/agg_metrics_{save_filename}_{j}.csv', index=False)            
    
    
    res_all_df = pd.concat(res_all).round(6)
    res_all_df = res_all_df.reset_index()
    res_all_df = res_all_df.rename(columns={res_all_df.columns[0]: "timestamp" })
    res_all_df.to_csv(f'/home/user/energygpt/Results/moirai/forecasts/{dataset}/{save_filename}_{j}.csv', index=False)            

    ts_metrics_all_df = pd.concat(ts_metrics_all).round(6)    
    ts_metrics_all_df.to_csv(f'/home/user/energygpt/Results/moirai/results/{dataset}/ts_metrics_{save_filename}_{j}.csv', index=False)            

    agg_metrics_all_df = pd.concat(agg_metrics_all).round(6)   
    agg_metrics_all_df.to_csv(f'/home/user/energygpt/Results/moirai/results/{dataset}/agg_metrics_{save_filename}_{j}.csv', index=False)    
    print("Zero-shot completed for ", dataset, ".")            

    return res_all_df, ts_metrics_all_df, agg_metrics_all_df


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = "zero-shot using moirai")

    parser.add_argument("--dataset", type = str, help = "Dataset name")

    args = parser.parse_args()
    dataset = args.dataset

    files_list = glob.glob(f'/media/user/DATA2/Dataset_V0.0/Energy-Load-Profiles/Hourly/Residential/{dataset}/*.parquet')

    
    os.makedirs(f'/home/user/energygpt/Results/moirai/forecasts/{dataset}/', exist_ok = True)
    os.makedirs(f'/home/user/energygpt/Results/moirai/results/{dataset}/', exist_ok = True)

    for filename in files_list:
        print(datetime.now(), filename)
        results = process_file(filename, dataset)
        # if results is not None:
        #     results.to_csv(f'../forecasts/{dataset}/{os.path.basename(filename)}', index=False)
        print('')

    torch.cuda.empty_cache()
