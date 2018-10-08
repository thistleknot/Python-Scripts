import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import statsmodels.api as sm
#from numpy import percentile

input_file = "C:/Users/user/Documents/plots/parsed.csv"

df = pd.read_csv(input_file, header = 0)
df

X, y = df.iloc[:, :-2], df.iloc[:, -2]

x = df.iloc[:, -3]

x

X

y

model = sm.OLS(y, x).fit()

model.summary()