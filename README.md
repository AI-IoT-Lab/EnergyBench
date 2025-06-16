# EnergyBench
EnergyBench: A Large-Scale Energy Meter Dataset and Benchmark for Load Forecasting

## Overview
Energy data analytics is crucial for optimizing energy consumption, improving operational efficiency, and supporting sustainability in power systems. Despite its importance, the lack of large-scale, diverse, and publicly available energy datasets poses a significant challenge for training and evaluating advanced AI models. To address this, we introduce **EnergyBench**, a comprehensive dataset designed to support **Short-Term Load Forecasting (STLF)** and to support a variety of energy analytics applications, including load profile analysis, and load forecasting. EnergyBench comprises a real-world subset containing **78,037** electricity time series from commercial and residential buildings across the globe, with sampling intervals ranging from 15 minutes to 1 hour, and a synthetic subset of 120 million building time series generated using energy simulation models. 

Along with the dataset , we also benchmark a variety of traditional and foundation models for STLF and analyze their performance across different regions.
We provide an Hugging Face Dataset for multiple real building and synthetic energy consumption datasets as Pandas DataFrames in Parquet, simple (naive) to advanced (TSFM) baselines, metrics management, and more.

## Getting Started
We recommend using [Anaconda](https://www.anaconda.com/download) to run the experiments. Create the separate conda environment using the (modelname)_environment.yml found
under each model directory in the Notebooks. 
```
conda env create -f <modelname>_environment.yml
```
For TimesFM, kindly follow the respective [READEME.md](https://github.com/google-research/timesfm/blob/master/README.md) for the installation. For more information about each model, kindly look in 
their Github repositories.

## Benchmarking
**STLF**:

The univariate short-term load forecasting (STLF) problem can be defined
as follows: Given *H* past load values *x*<sub>*t* − *H* : *t*</sub>, we
aim to predict the conditional distribution of *T* unobserved future
load values *y*<sub>*t* + 1 : *t* + *T*</sub>. For our analysis, we
considered the STLF problem of forecasting the next day’s load values (T
= 24) using data from the past week (H = 168).

**Protocol**:
- We evaluate a range of models on our real 69,560 residential buildings, from simple baseline statistical models to advanced TSFM under a zero-shot setting for STLF.
- To enusre the fair comparison, we have used only real residential buildings, since some of the real commercial buildings have been included in the pre-training corpora of existing open-source TSFMs.
- Also, it is to be noted that, all real residential buildings, except LCL, in our EnergyBench have not been included in the pre-training datasets of existing open-source TSFMs.
- This highlights the potential of EnergyBench as a valuable resource for evaluating generalization and transfer capabilities of TSFMs across previously unseen building energy consumption domains

**Evaluation metrics**:
- NRMSE - Normalized Root Mean Squared Error (%)

## Dataset Repository
EnergyBench is hosted in the Hugging Face and is available for download with a permissive CC-4.0 license.
[HuggingFace Datasets](https://huggingface.co/datasets/ai-iot/EnergyBench)

## Overview of Dataset

It compiles over 60 detailed electricity consumption datasets encompassing approximately 78,037 real buildings representing the global building stock, offering insights into the temporal and spatial variations in energy consumption. The buildings are classified into two categories: Commercial, which includes **2,916** buildings, and Residential, which includes **75,121** buildings, with a total of 65,000 building-days of time-series data. These buildings span diverse geographies, climatic zones, and operational profiles, providing a broad spectrum of usage patterns. Additional to the real buildings, it also includes synthetic and simulated buildings electricity consumption data.

All processed datasets contain hourly data, with some offering additional 15-minute and 30-minute resolution. To ensure efficient storage, we saved the datasets in Parquet format rather than CSV. For datasets involving a large number of buildings, we partitioned the data into multiple Parquet files.

Table 1: Summary of EnergBench dataset (Real Buildings)
| Category    | # Datasets | # Buildings | # Days | # Hourly Obs |
|-------------|-------------|--------------|---------|--------|
| Commercial  | 20          | 2,916        | 16,706  | 62M    |
| Residential | 47          | 75,121       | 43,852  | 1.2B   |

Table 2: Summary of EnergBench dataset (Synthetic and Simulated Buildings)
| Category    | # Datasets | # Buildings | # Days | # Hourly Obs |
|-------------|-------------|--------------|---------|--------|
| Commercial  | 1           | 207,559      | 365     | ~1.7B  |
| Residential | 4           | ~120M        | 966     | ~1044B |

To know more about dataset, please visit [HuggingFace Datasets](https://huggingface.co/datasets/ai-iot/EnergyBench).

## Benchmarking Leaderboard

Table 3: Comparison of the performance of traditional models (naive, auto-ARIMA, linear regression, lightgbm), and TSFMs using median NRMSE across different residential datasets. 
(Bold - Best, Italics - $2^{nd}$ Best)

|		Dataset	    | Naive  | Auto-ARIMA | Linear Regression | LightGBM | Moirai |Chronos| TimesFM |Time-MoE|
|---------------------------|--------|--------|--------|--------|--------|-------------------|----------------|--------|
| AMPD             | 76.08  | 61.61  | 66.23  | 64.33  | 56.77  | 59.73             | 82.84          | 52.41  |
| CEEW             | 100.74 | 78.27  | 129.24 | 132.07 | 188.76 | 92.14             | 60.06          | 96.20  |
| DCB              | 43.31  | 34.30  | 15.06  | 12.99  | 17.89  | 13.33             | 13.22          | 12.09  |
| DESM             | 55.77  | 46.54  | 60.41  | 58.03  | 84.73  | 42.67             | 38.94          | 42.41  |
| DTH              | 107.22 | 79.57  | 36.04  | 36.53  | 38.73  | 32.16             | 33.79          | 30.79  |
| ECCC             | 104.10 | 80.5   | 87.07  | 83.73  | 69.34  | 76.48             | 64.02          | 63.47  |
| ECWM             | 48.69  | 40.22  | 53.52  | 50.39  | 38.84  | 39.45             | 37.70          | 39.68  |
| ENERTALK        | 75.69  | 60.65  | 382.28 | NaN    | 71.23  | 56.76             | 51.17          | 115.49 |
| flEECe           | 85.84  | 56.58  | 67.37  | 64.13  | 53.09  | 55.76             | 63.58          | 50.63  |
| GoiEner          | 142.20 | 115.04 | 129.24 | 126.80 | 119.95 | 118.10            | 108.25         | 176.72 |
| HES              | 118.23 | 97.58  | 112.78 | 118.36 | 95.67  | 100.29            | 91.46          | 94.18  |
| Honda SMART Home | 33.01  | 23.24  | 15.33  | 14.96  | 13.99  | 12.82             | 13.66          | 12.59  |
| HUE              | 99.69  | 85.00  | 212.78 | 220.58 | 83.23  | 84.50             | 75.06          | 131.82 |
| iFlex            | 45.65  | 36.60  | 43.34  | 46.11  | 35.18  | 34.89             | 43.20          | 75.95  |
| IHEPC            | 100.48 | 78.59  | 77.68  | 77.59  | 73.07  | 76.88             | 68.90          | 66.16  |
| IPC-Residential  | 51.20  | 34.76  | 50.27  | 52.93  | 106.48 | 14.23             | 11.87          | 28.26  |
| IRH              | 112.69 | 96.80  | 110.27 | 105.67 | 93.52  | 101.57            | 88.57          | 87.10  |
| LEC              | 144.14 | 109.87 | 148.53 | 142.08 | 172.42 | 128.87            | 87.78          | 120.93 |
| MFRED            | 37.84  | 33.31  | 34.58  | 33.21  | 26.01  | 24.26             | 33.74          | 25.03  |
| MIHEC            | 154.21 | 144.78 | 226.99 | 199.14 | 148.75 | 164.6             | 138.33         | 158.10 |
| NEEA             | 102.05 | 77.34  | 82.78  | 80.50  | 76.05  | 74.96             | 68.00          | 80.42  |
| NESEMP          | 137.4  | 102.7  | 151.02 | 168.79 | 103.63 | 96.31             | 81.66          | 142.77 |
| Norwegian       | 61.67  | 49.13  | 60.66  | 59.18  | 47.36  | 48.58             | 49.33          | 44.00  |
| PES              | 6.04   | 6.08   | 103.85 | NaN    | 2.68   | 3.10              | 2.86           | 4.38   |
| Plegma           | 186.82 | 153.26 | 182.59 | 178.07 | 165.49 | 165.71            | 151.61         | 171.28 |
| Prayas           | 96.50  | 77.08  | 89.71  | 89.35  | 139.70 | 74.76             | 68.42          | 96.22  |
| RHC              | 76.08  | 61.61  | 66.23  | 64.33  | 56.57  | 59.73             | 82.84          | 52.41  |
| RSL              | 101.80 | 81.49  | 123.07 | 138.23 | 103.44 | 81.99             | 66.02          | 125.01 |
| SAVE             | 1.00   | 1.02   | 2.25   | 37.87  | 1.09   | 1.37              | 2.79           | 34.70  |
| SFHG             | 108.52 | 90.18  | 97.28  | 95.78  | 96.60  | 92.58             | 93.94          | 97.06  |
| SGSC             | 128.87 | 101.44 | 110.16 | 107.25 | 96.86  | 101.93            | 91.09          | 118.67 |
| SMART-Star       | 91.81  | 71.35  | 107.14 | 104.59 | 120.06 | 76.14             | 69.74          | 81.88  |
| SRSA             | 59.52  | 46.29  | 53.30  | 52.52  | 151.53 | 42.53             | 39.31          | 50.10  |
| UKST             | 110.54 | 89.41  | 102.61 | 129.40 | 89.55  | 84.44             | 84.25          | 103.60 |
| WED              | 104.37 | 84.66  | 102.69 | 64.13  | 101.84 | 85.36             | 9.90           | 73.19  |
| **Average**          | 88.85  | 71.05  | 99.84  | 91.20  | 84.00  | *69.11* | **61.94** | 91.96  |

## Directory structure
```
.
└── Benchmarking/
    ├── Arima             <- STLF using Auto-ARIMA
    ├── LightGBM          <- STLF using LightGBM
    ├── Linear-Regression <- STLF using Linear Regression
    ├── Naive             <- STLF using Naive model
    ├── TimesFM           <- Zero-shot using TimesFM
    ├── Moirai            <- Zero-shot using Moirai
    └── Chronos           <- Zero-shot using Chronos
└── Data-preprocessing/
    ├── HES               <- Data-preprocessing for HES
    ├── METER             <- Data-preprocessing for METER
    ├── NEEA              <- Data-preprocessing for NEEA
    ├── NESEMP            <- Data-preprocessing for NESEMP
    ├── SAVE              <- Data-preprocessing for SAVE
    └── UKST              <- Data-preprocessing for UKST
└── README.md
└── LICENSE
```



### Note: This repository is under development.
