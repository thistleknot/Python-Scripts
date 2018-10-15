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
ty_lagged = ty.shift(+1)
ty_future = ty.shift(-1)


#yields

tx_yield = (tx-tx_lagged)/tx_lagged
ty_yield = (ty-ty_lagged)/ty_lagged
ty_future_yield = (ty_future-ty)/ty

#validate
#ty_yield


#tx_lagged

#df1 = df.loc[:, ['date', 'CPIAUCSL']]df1 = df.loc[

XNew = pd.concat([tx, tx_yield, ty, ty_yield], axis=1)
YNew = ty_future_yield

model = sm.OLS(ty, tx).fit()
modelWLag = sm.OLS(YNew.loc[109:270], XNew.loc[109:270,]).fit()

model.summary()
#ty_future_yield

modelWLag.summary()
#YNew.loc[109:,]
#XNew.loc[109:,]
