import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pdb

####### EXTRACTING AND CLEANING
# logistics_path = '../data/PrimaryData/contraceptive_logistics_data.csv'
logistics_path = '../data/ProcessingOutput/contraceptive_logistics_data_clean.csv'
df = pd.read_csv(logistics_path)

### rows where all fields are zero
empty_rows = (df[['stock_initial', 'stock_received', 'stock_distributed',
                    'stock_adjustment', 'stock_end', 'average_monthly_consumption',
                    'stock_stockout_days']] == 0).all(axis = 1)

df = df[~empty_rows]

### rows where initial stock exists and none are distributed even with historical demand
invalid_zeros = (df.stock_initial > 0) & (df.stock_distributed == 0) \
                & (df.average_monthly_consumption > 5)

df = df[~invalid_zeros]

# Create date index from year/month columns
df['date_col'] = df['month'].map(str)+ '-' +df['year'].map(str)
df['date_col'] = pd.to_datetime(df['date_col'], format='%m-%Y')
df = df[['date_col','year','month','site_code','product_code','stock_distributed']]
df.sort_values(by=['date_col'], inplace=True)
df.reset_index(inplace=True, drop=True)
df.index = df['date_col']

###### MODELING
num_months_to_predict = 3

end_predict_month = df['date_col'].iloc[-1].to_pydatetime()
start_predict_month = (end_predict_month - relativedelta(months=+num_months_to_predict-1))
end_predict_month_int = end_predict_month.month
start_predict_month_int = start_predict_month.month
predict_year = 2019

PLOT = False

# dictionary of dictionaries containing predictions
# {site_code : {product_code1 : df1, product_code2 : df2}}
# pred_dict = dict()

unique_product_codes = df['product_code'].unique()
unique_site_codes = df['site_code'].unique()

# Note that this won't work in the event that predictions cross over a year
pred_df = pd.DataFrame()
train_df = df.loc[(df.year != predict_year) & (df.month < start_predict_month_int)]
test_df = df.loc[(df.year == predict_year) & (df.month >= start_predict_month_int) & (df.month <= end_predict_month_int)]

for site_dx, site_code in enumerate(unique_site_codes):
    print('On site_code {} of {}'.format(site_dx,len(unique_site_codes)-1))
    # pred_dict[site_code] = dict()

    for product_dx, product_code in enumerate(unique_product_codes):
        print('On product_code {} of {}'.format(product_dx,len(unique_product_codes)-1))

        # The current training slice
        curr_train = train_df.loc[(train_df['site_code'] == site_code) & (train_df['product_code'] == product_code)]
        curr_test =  test_df.loc[(test_df['site_code'] == site_code) & (test_df['product_code'] == product_code)]
        curr_train = curr_train.sort_index()

        if len(curr_train) < 1:
            print('looks like we got a blank one!!!')
            break

        pdb.set_trace()
        model = ARIMA(curr_train['stock_distributed'], order=(1, 1, 1))
        model_fit = model.fit()
        model_fit.predict(start=datetime(predict_year, start_predict_month_int, 1))
        curr_test['pred_stock_distributed'] = forecast.tolist()
        test_df = pd.merge(test_df, curr_test, how='left', on=['site_code', 'product_code'])

        # pred_dict[site_code][product_code] = predict

test_df = predictions = predictions.fillna(0)
set_trace()
print("MSE = " + str(mean_squared_error(pred_df['stock_distributed'], pred_df['pred_stock_distributed']))) # MSE
print("RMSE = " + str(np.sqrt(mean_squared_error(pred_df['stock_distributed'], pred_df['pred_stock_distributed'])))) # RMSE
print("MAE = " + str(mean_absolute_error(pred_df['stock_distributed'], pred_df['pred_stock_distributed']))) # MAE
print("R2 = " + str(r2_score(pred_df['stock_distributed'], pred_df['pred_stock_distributed']))) # R2
# print("MASE = " + str(MASE(train['stock_distributed'], predictions['stock_distributed'], predictions['pred_stock_distributed']))) # MASE
    
pdb.set_trace()