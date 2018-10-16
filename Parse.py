import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model, metrics 

import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import statsmodels.api as sm
#from numpy import percentile

input_file = "C:/Users/user/Documents/Python Scripts/parsed.csv"

df = pd.read_csv(input_file, header = 0)

# # of rows
    #https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe
split=int(len(df.index))/2

#DCOILWTICO was not significant, most likely due to present of cpiaucsl
X = df.loc[108:, ['date','CPIAUCSL','PSAVERT','GDPC1','DGS10','UMCSENT','EMRATIO','POPTOTUSA647NWDB','TTLHH','MEHOINUSA672N']]

y = df.loc[108:, ['SPCS20RSA']]

#108 to just shy of last

#training
tx = X.iloc[0:int(split), 1:10]

#df1 = df.loc[:, ['date','PSAVERT','GDPC1','DGS10','UMCSENT','EMRATIO','POPTOTUSA647NWDB','TTLHH','MEHOINUSA672N']]


ty = y.iloc[0:int(split), ]

#validation
vx = X.iloc[int(split+1):,1:10]

#print
vy = y.iloc[int(split+1):,1:10]

#print
vx
ty
tx
#split
#ty = df.iloc[0:int(split), -1]

#tX = X.iloc[0:(split+1):, 0:52]

#y

#offset
tx_lagged = tx.shift(+1)
vx_lagged = X.shift(+1)
ty_lagged = ty.shift(+1)
vy_lagged = ty.shift(+1)
ty_future = ty.shift(-1)
vy_future = ty.shift(-1)

#yields

tx_yield = (tx-tx_lagged)/tx_lagged
vx_yield = (vx-vx_lagged)/vx_lagged
ty_yield = (ty-ty_lagged)/ty_lagged
vy_yield = (vy-vy_lagged)/vy_lagged
ty_future_yield = (ty_future-ty)/ty
vy_future_yield = (vy_future-vy)/vy

#validate
#ty_yield

#tx_lagged

#df1 = df.loc[:, ['date', 'CPIAUCSL']]df1 = df.loc[

XNew = pd.concat([tx, tx_yield, ty, ty_yield], axis=1)
vXNew = pd.concat([vx, vx_yield, vy, vy_yield], axis=1)
YNew = ty_future_yield
vYNew = vy_future_yield

#training
#model = sm.OLS(ty, tx).fit()
modelWLag = sm.OLS(YNew.loc[109:270], XNew.loc[109:270,]).fit()

reg = linear_model.LinearRegression() 
  
# train the model using the training sets 
#reg.fit(YNew.loc[109:270], tx.loc[109:270,])

#model.summary()
#ty_future_yield

modelWLag.summary()
#modelWLag.params

modelWLag.predict(XNew.loc[271:327])
#vx_lagged
vx_yield
#modelWLag.fit()
#results = model.fit()
#results

#modelWLag.fit()
#YNew.loc[109:,]
#XNew.loc[109:,]
