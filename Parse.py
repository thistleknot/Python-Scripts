import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import statsmodels.api as sm
#from numpy import percentile

input_file = "C:/Users/user/Documents/Python Scripts/parsed.csv"



df = pd.read_csv(input_file, header = 0)


# # of rows
    #https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe
split=int(len(df.index)-1)

tX = df.iloc[1:split+1:, 1:52]

ty = df.iloc[1:split+1:, -1]

#tX = X.iloc[0:(split+1):, 0:52]

#ty

model = sm.OLS(ty, tX).fit()

model.summary()
#split

#tX
#df.iloc[:, 0:52]
#df.loc[0:100,'date']