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
df
X = df.iloc[108:, 1:48]

y = df.iloc[108:, -1]

#108 to just shy of last

#training
tx = X.iloc[0:int(split), 1:48]

df1 = df.loc[:, ['date', 'CPIAUCSL']]


ty = y.iloc[0:int(split), ]

#validation
vx = X.iloc[int(split+1):,1:48]

#print
tx

#print
vx
ty
tx
#split
#ty = df.iloc[0:int(split), -1]

#tX = X.iloc[0:(split+1):, 0:52]

#y

model = sm.OLS(ty, tx).fit()

model.summary()
#df1 = df.loc[:, ['date', 'CPIAUCSL']]df1 = df.loc[:, ['date', 'CPIAUCSL']]
df1
#split
#y
#tX
#df.iloc[:, 0:52]
#df.loc[0:100,'date']