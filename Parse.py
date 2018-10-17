import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split

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
x = df.loc[0:, ['date','CPIAUCSL','PSAVERT','GDPC1','DGS10','UMCSENT','EMRATIO','POPTOTUSA647NWDB','TTLHH','MEHOINUSA672N']]

y = df.loc[0:, ['CSUSHPINSA']]
#y

#offset
x_lagged = x.shift(+1)
y_lagged = y.shift(+1)

y_future = y.shift(-1)

# skip na's
    # 1: skip 1st row
    # :-1 skip last row
    # ,1: skip 1st column
    
x_yield = (x.iloc[1:-1,1:]-x_lagged.iloc[1:-1,1:])/x_lagged.iloc[1:-1,1:]

y_yield = (y.iloc[1:-1,]-y_lagged.iloc[1:-1,])/y_lagged.iloc[1:-1,]
#y_future_yield = ()

y_future_yield = (y_future.iloc[1:-1,]-y.iloc[1:-1,])/y_future.iloc[1:-1,]
y_future_yield

x_and_y_with_yields = pd.concat([x.iloc[1:-1,1:], x_yield, y.iloc[1:-1,], y_yield], axis=1)
x_and_y_with_yields

# .938 Adj R^2
model = sm.OLS(y_future_yield.loc[1:int(split+1)], x_and_y_with_yields.loc[1:int(split+1)]).fit()

#note: has to be +2
    #x_and_y_with_yields.loc
results = pd.concat([model.predict(x_and_y_with_yields.loc[int(split+2):]), y_future_yield[int(split+1):]], axis=1)
results

model.params
model.summary()

#scikit
#https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
X_train, X_test, y_train, y_test = train_test_split(x_and_y_with_yields.loc[1:int(split+1)], y_future_yield.loc[1:int(split+1)], test_size=0.2)

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

predictions.shape[0]
y_test.shape[0]
#pd.concat([predictions,y_test],axis=1)
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")

print ("Score:", model.score(X_test, y_test))

#type(y_test)
#type(predictions)
#predictions[0:,0:].tolist()
#model.params
#model.summary()
