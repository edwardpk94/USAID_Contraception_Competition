# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:34:33 2020

@author: jdumiak
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import itertools
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')
import matplotlib.backends.backend_pdf
import datetime as dt

date = dt.datetime.today().strftime('%Y%m%d')

# Create pdf saving
pdf = matplotlib.backends.backend_pdf.PdfPages('C:/Users/jdumiak/Documents/CfA/USAID_Forecast/' + str(date) + '_sarima_output.pdf')


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


# Read in data
logistics_data = pd.read_csv("~/Documents/CfA/USAID_Forecast/Data/USAID_Intelligent_Forecasting/contraceptive_logistics_data.csv")[['year','month','site_code','product_code','stock_distributed']]

# Create date field from year and month
logistics_data['date'] = logistics_data['month'].map(str)+ '-' + logistics_data['year'].map(str)
logistics_data['date'] = pd.to_datetime(logistics_data['date'], format='%m-%Y').dt.to_period('M')

# Sort by date
logistics_data = logistics_data.sort_values(by = 'date')

# Set year and months to split
test_year = 2019
start_month_test = 5
end_month_test = 9

# First 40 months train, last 5 test (90/10 split)
test = logistics_data.loc[(logistics_data.year == test_year) & (logistics_data.month >= start_month_test) & (logistics_data.month <= end_month_test)]
train = logistics_data[~logistics_data['date'].isin(test['date'])]

uniq_site = logistics_data['site_code'].unique()#[0:2]
uniq_prod = logistics_data['product_code'].unique()#[0:2]

# Initialize all pred df
all_predictions = []
all_actuals = []

# Loop through site and product code - store forecast and output all plots
for site in uniq_site:
    for prod in uniq_prod:
        # Initialize 40 x 1 training data frame with the unique dates as the index
        init_df = pd.DataFrame(index=train.date.unique().to_timestamp())
        
        # Subset to relevant data
        df_train = train[(train['site_code'] == site) & (train['product_code'] == prod)][['date','stock_distributed']]
        df_test = test[(test['site_code'] == site) & (test['product_code'] == prod)][['date','stock_distributed']]
        
        df_train = df_train.set_index(['date'])[['stock_distributed']]
        df_train.index=df_train.index.to_timestamp()
        
        if sum(df_train.values) < 1:
            print('All zero values for ' + site + ' ' + prod)
            continue
        
        df_test = df_test.set_index(['date'])[['stock_distributed']]
        df_test.index=df_test.index.to_timestamp()
        
        if len(df_test) < 5:
            print('No testing data for ' + site + ' ' + prod)
            continue
        
        # Merge into 40 x 1 shape with nans for missing data
        df_train = pd.merge(init_df, df_train, how = "left", left_index=True, right_index=True)
        
        all_actuals.append(df_test.values)
        
        # Plot the ACF and PACF
        # plot_pacf(df['stock_distributed']); #significant peak at lag 1 use MA(1)
        # plot_acf(df['stock_distributed']); #significant peak at lag 1 use AR(1)
        
        # ad_fuller_result = adfuller(df['stock_distributed'])
        # print(f'ADF Statistic: {ad_fuller_result[0]}')
        # print(f'p-value: {ad_fuller_result[1]}')
        
        
        # p = d = q = range(0, 2)
        # pdq = list(itertools.product(p, d, q))
        # seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        # print('Examples of parameter for SARIMA...')
        # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
        # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
        # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
        # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
        
        
        # for param in pdq:
        #     for param_seasonal in seasonal_pdq:
        #         mod = SARIMAX(df_train['stock_distributed'], order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
        #         results = mod.fit()
        #         print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
        
        # SARIMA(0, 1, 1)x(0, 1, 1, 12)12 
              
        mod = SARIMAX(df_train['stock_distributed'], order=(0, 1, 1),seasonal_order=(0, 1, 1, 12), enforce_stationarity=False,enforce_invertibility=False)
        results = mod.fit()
        
        print(results.summary().tables[1])
        
        # Create pdf figure for each combination
        fig = plt.figure()
        
        results.plot_diagnostics(figsize=(18, 8))
        plt.show()
        pdf.savefig(fig)
        
        # Forecast out and store preds
        forecast_values = results.forecast(steps = df_test.shape[0])
        all_predictions.append(forecast_values.ravel())
        
        # Get predictions and CI 
        pred = results.get_prediction(start=pd.to_datetime('2019-05-01'), end=pd.to_datetime('2019-09-01'),dynamic=False)
        pred_ci = pred.conf_int()
        pred.summary_frame()
        pred.predicted_mean
        
        # Plot previous data and forecast
        # Combine train and test
        all_df = df_train.append(df_test)
        fig = plt.figure()
        
        ax = all_df.plot()
        pred.predicted_mean.plot(ax = ax, label = "Forecasts")
        
        ax.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color = 'g', alpha = .5)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Contraceptive Volume")
        ax.set_title("SARIMA Forecast " + site + " " + prod)
        plt.show()
        pdf.savefig(fig)
        
# Output all plots to a PDF to view
pdf.close()

# Create final DF
final_predictions  = pd.DataFrame([val for sublist in all_predictions for val in sublist])
final_predictions.columns = ['stock_distributed_forecast']

final_actuals = pd.DataFrame([val for sublist in all_actuals for val in sublist])
final_actuals.columns = ['stock_distributed_actual']

# final_df = pd.concat([final_actuals, final_predictions], axis = 1)

# Calculate the MSE, RMSE, MAE, R2, MASE
print("MSE = " + str(mean_squared_error(final_actuals['stock_distributed_actual'], final_predictions['stock_distributed_forecast']))) # MSE
print("RMSE = " + str(np.sqrt(mean_squared_error(final_actuals['stock_distributed_actual'], final_predictions['stock_distributed_forecast'])))) # RMSE
print("MAE = " + str(mean_absolute_error(final_actuals['stock_distributed_actual'], final_predictions['stock_distributed_forecast']))) # MAE
print("R2 = " + str(r2_score(final_actuals['stock_distributed_actual'], final_predictions['stock_distributed_forecast']))) # R2
print("MASE = " + str(MASE(df_train['stock_distributed'], final_actuals['stock_distributed_actual'], final_predictions['stock_distributed_forecast']))) # MASE


