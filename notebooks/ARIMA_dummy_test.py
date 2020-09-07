import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import numpy as np
from pdb import set_trace

index_months = [1,2,3,4,5,6,7,8]
index = ['2018-{}-1'.format(index_month) for index_month in index_months]
print(index)
df = pd.DataFrame({'date' : index, 'input' : [0,1,np.nan,3,3,np.nan,np.nan,10]})
df['datetime'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df.set_index(df['datetime'])
df = df[['input']]
df.index = df.index.to_period('M')
set_trace()

model_fit = ARIMA(df['input'], order=(1, 1, 1)).fit()
predict = model_fit.predict(start=datetime(2018,9,1),end=datetime(2018,11,1)) 

set_trace()