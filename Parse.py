import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import statsmodels.api as sm
#from numpy import percentile

input_file = "C:/Users/user/Documents/Python Scripts/parsed.csv"



df = pd.read_csv(input_file, header = 0)
df

# # of rows
    #https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe
split=int((len(df.index)-1)/2)

X, y = df.iloc[:, 1:52], df.iloc[:, -1]

X

y

model = sm.OLS(y, X).fit()

model.summary()
split
#df.iloc[:, 0:52]
#df.loc[0:100,'date']