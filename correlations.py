from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

input_file = "output.csv"

#https://stackoverflow.com/questions/36519086/pandas-how-to-get-rid-of-unnamed-column-in-a-dataframe/36519122
df = pd.read_csv(input_file, header = 0, index_col=0)


#x = df.drop(columns=['date','CSUSHPINSA']).sample(20, axis=1)
date = df.loc[0:,['test2_z.date']]

#append if desire a sample
#.sample(50, axis=1)
#attempts at trying to count 0's and drop columns have failed in R, so I'm hard coding the column drop here
x = df.drop(columns=['test2_z.date','CSUSHPINSA','FGCCSAQ027S','future'])
y = df.loc[0:,['CSUSHPINSA']]

xLagged = x.shift(+1)
yLagged = y.shift(+1)
yFuture = y.shift(-1)
xYield = (x/xLagged)
yYield = (y/yLagged)
yFYield = (yFuture/y)

xInteraction = (xLagged*x)
yInteraction = (yLagged*y)
yFInteraction = (yFuture*y)

symbols = [date, x, xYield,y, yYield, yFYield]

#https://pandas.pydata.org/pandas-docs/stable/merging.html
#Set logic on the other axes¶

#https://stackoverflow.com/questions/17477979/dropping-infinite-values-from-dataframes-in-pandas
xYield.columns = ['xYield_' + str(col) for col in xYield.columns]
y.columns = ['y_' + str(col) for col in y.columns]
x.columns = ['x_' + str(col) for col in x.columns]
yYield.columns = ['yYield_' + str(col) for col in yYield.columns]
yFYield.columns = ['yFYield_' + str(col) for col in yFYield.columns]

#https://chrisalbon.com/python/data_wrangling/pandas_create_column_using_conditional/
#used for quarters
df_x = pd.DataFrame(date)

#df_x['Q1'] = np.where(==12, 1, 0)




#x['Q1'] = np.where(date['date']>=50, 1, 0)

result = pd.concat([date, df_x, xYield, y, yYield, yFYield], axis=1, sort=False)
#replace inf's with 0
result.replace(np.inf, 0, inplace=True)

result.to_csv("output.csv", sep=',')
#plt.matshow(result.corr())
#https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas

corr = result.corr()
#matches corr formula in excel
#matches formula here: https://libguides.library.kent.edu/SPSS/PearsonCorr
corr.to_csv("corr.csv", sep=',')



#print(corr)
#corr.style.background_gradient(highlight_rows, axis = 0)
#corr.style.apply(highlight_rows, axis = 0)
#corr.style.background_gradient()



#dateLagged = date.shift(+1)
#bins = [ 1, 5, 10, 25, 50, 100]
#df['binned'] = pd.cut(df['percentage'], bins)


#pandas.Series.dt.month(date)


