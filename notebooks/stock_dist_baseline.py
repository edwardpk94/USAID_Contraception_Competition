# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:28:37 2020

@author: jdumiak

Simple baseline using last 3 month average per site per product and projecting forward

"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from pdb import set_trace


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


# Read in the contraceptive logistics data
logistics_data = pd.read_csv("contraceptive_logistics_data.csv")[['year','month','site_code','product_code','stock_distributed']]

# Set year and months to split
train_test_year = 2019
start_month_train = 4
end_month_train = 6

# We will take the average by site by product for the last 3 months and project 
# That forward three months and compare to actuals
train = logistics_data.loc[(logistics_data.year == train_test_year) & (logistics_data.month >= start_month_train) & (logistics_data.month <= end_month_train)]
test = logistics_data.loc[(logistics_data.year == train_test_year) & (logistics_data.month >= end_month_train + 1) & (logistics_data.month <= end_month_train + 3)]

# Group by site code and product code to get the average of stock dist for the last three months
train_avg_baseline = train.groupby(['site_code', 'product_code'])['stock_distributed'].mean().reset_index()
train_avg_baseline.rename(columns={'stock_distributed': 'pred_stock_distributed'}, inplace=True)

set_trace()
# Join on to test set as prediction
predictions = pd.merge(test, train_avg_baseline, how = "left", on = ['site_code', 'product_code'])

# Few don't have info for the last 3 months - make them zero
predictions.isnull().sum(axis = 0)
predictions = predictions.fillna(0)

# Calculate the MSE, RMSE, MAE, R2, MASE
print("MSE = " + str(mean_squared_error(predictions['stock_distributed'], predictions['pred_stock_distributed']))) # MSE
print("RMSE = " + str(np.sqrt(mean_squared_error(predictions['stock_distributed'], predictions['pred_stock_distributed'])))) # RMSE
print("MAE = " + str(mean_absolute_error(predictions['stock_distributed'], predictions['pred_stock_distributed']))) # MAE
print("R2 = " + str(r2_score(predictions['stock_distributed'], predictions['pred_stock_distributed']))) # R2
print("MASE = " + str(MASE(train['stock_distributed'], predictions['stock_distributed'], predictions['pred_stock_distributed']))) # MASE



