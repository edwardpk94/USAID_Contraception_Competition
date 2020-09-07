# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 22:56:27 2020

@authors: anbarry, Eddie Kunkel
"""


import warnings 
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
#%matplotlib inline
import matplotlib.pyplot as plt  
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from itertools import product
from pdb import set_trace

PRINT_VERBOSE = False

def MASE(training_series, testing_series, prediction_series):
    """
    Computes the mean-absolute scaled error forcast error for univariate time series prediction.
    
    See "Another look at measures of forecast accuracy", Rob J Hyndman
    
    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.
    
    """
    n = training_series.shape[0]
    d = np.abs(np.diff(training_series)).sum()/(n-1)
    
    errors = np.abs(testing_series - prediction_series )
    return errors.mean()/d

color = sns.color_palette()
sns.set_style('darkgrid')

### import data and adjust to only necessary cols
full_df = pd.read_csv(r'../data/ProcessingOutput/contraceptive_logistics_data_clean.csv')
full_df['date'] = pd.to_datetime(full_df['date'], format="%Y-%m-%d")
full_df = full_df[[
    'site_code',
    'product_code',
    'stock_distributed',
    'date'
    ]]

full_df['year'] = full_df['date'].dt.year
full_df['month'] = full_df['date'].dt.month
full_df['day'] = full_df['date'].dt.dayofyear
full_df['weekday'] = full_df['date'].dt.weekday
full_df = full_df.set_index('date', drop = False)
full_df.index.names = ['date_index']
full_df = full_df[['site_code','product_code','stock_distributed']]

### Creates the iteration of hyperparamater list
# Ande's original parameter space
# AR = [0,6,12]
# S=[0]
# MA= [0,6,12]

AR = [0,1,4]
S=[0,1]
MA= [0,1,4]

a = [AR,S,MA]
arima_params_combinations = list(product(*a))

import time

t0 = time.time()
t1 = time.time()

total = t1-t0

unique_site_codes = full_df['site_code'].unique()
unique_product_codes = full_df['product_code'].unique()

# !#EPK DEBUG Code to randomly sample some of the site codes for testing
unique_site_codes = np.random.choice(unique_site_codes, 1, replace=False)
unique_product_codes = np.random.choice(unique_product_codes, 2, replace=False)

# Record MASE and model parameters. Will be used for model selection
eval_df_list = []

# List of dataframe which each contain every date for a respective site/product code combination
combo_df_list = []

for site_code_dx, site_code in enumerate(unique_site_codes):
    t0 = time.time()
    print('on site_code_dx {} of {} ({})'.format(site_code_dx, len(unique_site_codes)-1, site_code))

    for product_code_dx, product_code in enumerate(unique_product_codes):
        product_code_start_time = time.time()
        if PRINT_VERBOSE:
            print('on product_code_dx {} of {} ({})'.format(product_code_dx, len(unique_product_codes)-1, product_code))

        # Select the current site/product code combination and fill in missing dates
        all_dates = pd.date_range(start=datetime(2016,1,1), end=datetime(2019,9,1), freq='MS')

        current_df = full_df[(full_df['product_code']==product_code) & (full_df['site_code']==site_code)]
        current_df = current_df.reindex(all_dates, fill_value=None)
        current_df.loc[:,'site_code'] = site_code
        current_df.loc[:,'product_code'] = product_code
        current_df.index.names = ['date_index']

        # Ensure all NaNs are set to np.nan to play nice with StatsModels
        current_df.loc[:,'stock_distributed'] = current_df['stock_distributed'].fillna(np.nan)

        train_df_p = current_df[current_df.index < '2019-07-01']
        test_df_p = current_df[current_df.index > '2019-06-01']

        # Test df must have no NaNs so error metrics can be calculated
        test_df_p.loc[:,'stock_distributed'] = test_df_p.loc[:,'stock_distributed'].fillna(0)

        # Calculate the naive solution by averaging the 3 most recent months
        test_df_p['naive_pred'] = train_df_p[~train_df_p['stock_distributed'].isna()]['stock_distributed'].tail(3).mean()
        naive_MASE = MASE(train_df_p['stock_distributed'].fillna(0), test_df_p['stock_distributed'], test_df_p['naive_pred'])

        best_ARIMA_MASE = None
        best_ARIMA_pred = None
        best_ARIMA_params = None
        
        # INSERT ADDITIONAL RULES FOR WHEN NAIVE MODEL SHOULD BE USED, REGARDLESS OF OTHER METRICS
        num_non_nan_train_dates = (~train_df_p['stock_distributed'].isna()).sum()

        if num_non_nan_train_dates > 5:
            for arima_params in arima_params_combinations:
                if PRINT_VERBOSE:
                    print('fitting ARIMA model using params {}'.format(arima_params))

                try:
                    arima_model = ARIMA(train_df_p.stock_distributed, order=arima_params).fit()

                    arima_pred = arima_model.predict(start=datetime(2019,7,1), end=datetime(2019,9,1))
                    model_MASE = MASE(train_df_p['stock_distributed'].fillna(0), test_df_p['stock_distributed'], arima_pred)

                    # Store the parameters of the best-performing ARIMA model
                    if (not best_ARIMA_MASE) or (model_MASE < best_ARIMA_MASE):
                        best_ARIMA_MASE = model_MASE
                        best_ARIMA_pred = arima_pred
                        best_ARIMA_params = arima_params
                except:
                    pass

        curr_eval_df = pd.DataFrame({'site_code' : [site_code],
                                     'product_code' : [product_code],
                                     'ARIMA_params' : [best_ARIMA_params],
                                     'ARIMA_MASE' : [best_ARIMA_MASE],
                                     'naive_MASE' : [naive_MASE],
                                     'num_non_nan_train_dates' : [num_non_nan_train_dates]})

        eval_df_list.append(curr_eval_df)

        test_df_p['ARIMA_pred'] = best_ARIMA_pred
        test_df_p = test_df_p[['site_code','product_code','naive_pred','ARIMA_pred']]
        current_df = pd.merge(current_df, test_df_p, how='outer', on=['date_index','site_code','product_code'])

        # concatenating the individual site/product combination dataframes and doing a large merge later is
        #     much faster than doing an outer merge every loop
        combo_df_list.append(current_df)

        if PRINT_VERBOSE:
            print('execution of this site/product combination took {} seconds'.format(time.time()-product_code_start_time))


# DF with data and predictions for all site/product codes run
combo_df = pd.concat(combo_df_list)
combo_df.drop(columns=['stock_distributed'], inplace=True)

eval_df = pd.concat(eval_df_list)
eval_df['ARIMA_performed_better'] = eval_df['ARIMA_MASE'] < eval_df['naive_MASE']


# !#EPK Can remove this when testing all product/site codes. Should be the same number of entries as the original full_df
full_df = pd.merge(full_df, combo_df, how='outer', on=['date_index','site_code','product_code'])

print(eval_df)

# Draft final predictions using the best performing model
predictions_df_list = []
prediction_dates = pd.date_range(start=datetime(2019,10,1), end=datetime(2019,12,1), freq='MS')

for site_code_dx, site_code in enumerate(unique_site_codes):
    for product_code_dx, product_code in enumerate(unique_product_codes):
        curr_df = full_df.loc[(full_df['site_code'] == site_code) & (full_df['product_code'] == product_code),:]
        curr_eval = eval_df.loc[(eval_df['site_code'] == site_code) & (eval_df['product_code'] == product_code),:]
        curr_prediction = pd.DataFrame(index=prediction_dates)
        curr_prediction['site_code'] = site_code
        curr_prediction['product_code'] = product_code

        # Naive model prediction
        curr_naive_pred = curr_df[~curr_df['stock_distributed'].isna()]['stock_distributed'].mean()
        
        if np.isnan(curr_naive_pred):
            curr_naive_pred = 0

        # ARIMA model prediction
        if (curr_eval['ARIMA_performed_better'].iloc[0]) & (curr_eval['num_non_nan_train_dates'].iloc[0] > 0):
            try:
                arima_model = ARIMA(curr_df['stock_distributed'], order=curr_eval['ARIMA_params']).fit()
                arima_pred = arima_model.predict(start=datetime(2019,10,1), end=datetime(2019,12,1))
                curr_prediction['prediction'] = arima_pred
            except:
                # Use naive model if ARIMA fails to train
                curr_prediction['prediction'] = curr_naive_pred
        else:
            curr_prediction['prediction'] = curr_naive_pred

        predictions_df_list.append(curr_prediction)

predictions = pd.concat(predictions_df_list)
set_trace()


### shows MAPE SMAP and MASE for entire test file
def smape_kun(training, y_true, y_pred):
    mase = MASE(training, y_true, y_pred)
    mape = np.mean(abs((y_true-y_pred)/y_true))*100
    smape = np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))).fillna(0))
    print('MAPE: %.2f %% \nSMAPE: %.2f %% \nMASE: %.2f'% (mape,smape,mase), "%")

### Creates a list of individual MASEs per SITE/Product
# mase_list = []
# for site_code in list(test['site_code'].unique()):
#     print("on store "+str(y)+" out of "+str(len(list(test['site_code'].unique()))))
#     test_df = test[test['site_code']==site_code]
#     train_df = train[train['site_code']==site_code]
#     for product_code in list(test_df['product_code'].unique()):
#         train_df_p = train_df[train_df['product_code']==product_code]
#         test_df_p = test_df[test_df['product_code']==product_code]
#         mase = MASE(train_df_p['stock_distributed'], test_df_p['stock_distributed'], test_df_p['ARIMA_Pred'])
#         mase_list.append(mase)

### Shows just the individual MASE
print(MASE(train['stock_distributed'], test['stock_distributed'], test['ARIMA_Pred']))
