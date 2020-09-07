# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 22:56:27 2020

@author: anbarry
"""


import warnings 
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
#%matplotlib inline
import matplotlib.pyplot as plt  
import seaborn as sns
import statsmodels.api as sm
import datetime

color = sns.color_palette()
sns.set_style('darkgrid')

### import data and adjust to only necessary cols
train = pd.read_csv(r'C:\Users\anbarry\Documents\USAID_Forecast\USAID_intel_forecast\data\ProcessingOutput\contraceptive_logistics_data_clean.csv')
train['date'] = pd.to_datetime(train['date'], format="%Y-%m-%d")
train = train[[
    'site_code',
    'product_code',
    'stock_initial',
    'stock_received',
    'stock_distributed',
    'stock_adjustment',
    'average_monthly_consumption',
    'stock_stockout_days',
    'stock_ordered',
    'date'
    ]]

train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.dayofyear
train['weekday'] = train['date'].dt.weekday

### Training set was everything before 07-01-2019

test = train.loc[train['date']>='2019-07-01']
train = train.loc[train['date']<'2019-07-01']



# per 1 store, 1 item for ploting and getting hyperparameter estimates
# gathering a general understanding of what one site/product looks like
train_df = train[train['site_code']=='C4001']
train_df = train_df[train['product_code']=='AS27000']

sns.lineplot(x="date", y="stock_distributed",legend = 'full' , data=train_df)

sns.lineplot(x="date", y="stock_distributed",legend = 'full' , data=train_df[:28])

train_df = train_df.set_index('date')
train_df['sales'] = train_df['sales'].astype(float)


### plot seasonal decompose
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(train_df['stock_distributed'], model='additive', freq=12)

fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(15, 12)

### Testing for Stationarity
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries, window = 12, cutoff = 0.01):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
    
    print(dfoutput)

test_stationarity(train_df['stock_distributed'])

# =============================================================================
# #if not stationary
# first_diff = train_df.sales - train_df.sales.shift(1)
# first_diff = first_diff.dropna(inplace = False)
# test_stationarity(first_diff, window = 12)
# =============================================================================


### Plot ACF and PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_df.stock_distributed, ax=ax1) # 
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_df.stock_distributed, ax=ax2)# , lags=40



### Creates the iteration of hyperparamater list
from itertools import product
AR = [0,6,12]
S=[0]
MA= [0,6,12]
a = [AR,S,MA]
prod = list(product(*a))

### creates the list of training dates
date_list = pd.date_range(train['date'].min(),train['date'].max(), 
              freq='MS').tolist()
dates = pd.DataFrame(date_list, columns = ['date'])

import time

t0 = time.time()
t1 = time.time()

total = t1-t0

## Training the arima model on each store, and each location for each hyperparameter
predictions = pd.DataFrame()
y=1
for store in list(train['site_code'].unique()):
    t0 = time.time()
    print("on store "+str(y)+" out of "+str(len(list(train['site_code'].unique())))+' took '+str(total)+ ' time')
    train_df = train[train['site_code']==store]
    for pduct in list(train['product_code'].unique()):
        train_df_p = train_df[train_df['product_code']==pduct]
        ##check's if the site/product data is less than the expect number of dates
        ## if so, it move on to the next site/product
        if len(train_df_p)<42:
            continue
        lowAIC=1000000000
        for x in prod:
            try:
                ## checks if the AIC of the model is lower than the lowest AIC
                arima_mod6 = sm.tsa.ARIMA(train_df_p.stock_distributed, x).fit(disp=False)
                aic = arima_mod6.aic
                if aic < lowAIC:
                    lowAIC = aic
                    bestarima = arima_mod6
            except:
                pass

        try:
            ### grabs the next 3 forecast steps
            out = pd.DataFrame(bestarima.forecast(steps = 3)[0], columns = ['ARIMA_Pred'])
            out['product'] =pduct
            out['site'] = store
            out['date'] =['2019-07-01','2019-08-01','2019-09-01']
            predictions = predictions.append(out)
        except:
            continue
    y+=1
    t1 = time.time()
    total = t1-t0
    
## Write predictions to save after the long run
predictions.to_csv(r'C:\Users\anbarry\Documents\USAID_Forecast\USAID_intel_forecast\data\ProcessingOutput\arima_predictions.csv')

### Merge Predictions with test data to get actuals
predictions['date'] = pd.to_datetime(predictions['date'])
predictions = predictions.rename({'product':'product_code', 'site':'site_code'}, axis='columns')

test = pd.merge(test, predictions, how ='left', on = ['site_code', 'product_code', 'date'])
test = test[test.ARIMA_Pred.notnull()]


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

### shows MAPE SMAP and MASE for entire test file
def smape_kun(training, y_true, y_pred):
    mase = MASE(training, y_true, y_pred)
    mape = np.mean(abs((y_true-y_pred)/y_true))*100
    smape = np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))).fillna(0))
    print('MAPE: %.2f %% \nSMAPE: %.2f %% \nMASE: %.2f'% (mape,smape,mase), "%")

### Creates a list of individual MASEs per SITE/Product
mase_list = []
for store in list(test['site_code'].unique()):
    print("on store "+str(y)+" out of "+str(len(list(test['site_code'].unique()))))
    test_df = test[test['site_code']==store]
    train_df = train[train['site_code']==store]
    for pduct in list(test_df['product_code'].unique()):
        train_df_p = train_df[train_df['product_code']==pduct]
        test_df_p = test_df[test_df['product_code']==pduct]
        mase = MASE(train_df_p['stock_distributed'], test_df_p['stock_distributed'], test_df_p['ARIMA_Pred'])
        mase_list.append(mase)


### Shows jsut the individual MASE
print(MASE(train['stock_distributed'], test['stock_distributed'], test['ARIMA_Pred']))
