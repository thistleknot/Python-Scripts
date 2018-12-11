from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime

input_file = "output_test.csv"

#https://stackoverflow.com/questions/36519086/pandas-how-to-get-rid-of-unnamed-column-in-a-dataframe/36519122
df = pd.read_csv(input_file, header = 0, index_col=0)


#x = df.drop(columns=['date','CSUSHPINSA']).sample(20, axis=1)
#date = df.loc[0:,['test2_z.date']]

#quarters = (date.month-1)//3


#append if desire a sample
#.sample(50, axis=1)
#attempts at trying to count 0's and drop columns have failed in R, so I'm hard coding the column drop here
x = df.drop(columns=['test2_z.date','CSUSHPINSA','future'])
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
#df_x = pd.DataFrame(date)

#df_x['Q1'] = np.where(==12, 1, 0)

#x['Q1'] = np.where(date['date']>=50, 1, 0)

result = pd.concat([date, xYield, y, yYield, yFYield], axis=1, sort=False)
#replace inf's with 0
result.replace(np.inf, 0, inplace=True)

result.to_csv("output.csv", sep=',')
#plt.matshow(result.corr())
#https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas

fieldOfInterest='yFYield_CSUSHPINSA'

corr = result.corr()
#matches corr formula in excel
#matches formula here: https://libguides.library.kent.edu/SPSS/PearsonCorr
corr.to_csv("corr.csv", sep=',')
#abs(corr)
upperLimit = max(abs(corr[fieldOfInterest]).mean(),abs(corr[fieldOfInterest]).median())
#upperLimit

#https://stackoverflow.com/questions/42613467/how-to-select-all-rows-which-contain-values-greater-than-a-threshold
#df[df['column'] == value]

#drop yFYield_CSUSHPINSA from output, but, filter on this column by upperLimit
#upper=corr.drop(columns=['yFYield_CSUSHPINSA'])[abs(corr['yFYield_CSUSHPINSA']) > upperLimit]
upper=corr[fieldOfInterest][abs(corr[fieldOfInterest]) > upperLimit]

#https://stackoverflow.com/questions/26640145/python-pandas-how-to-get-the-row-names-from-index-of-a-dataframe
upperList = upper.axes[0].tolist()

#filters columns
upperSet=corr[upperList]
#upper.loc[upperList]
#upperSet[abs(corr[fieldOfInterest]) > upperLimit]

#column means
lowerLimit = min(abs(upperSet).mean().mean(),abs(upperSet).median().median())

#lower1 = upperSet[[]]

#lowerLimit
LowerList1 = abs(upperSet).mean()[abs(upperSet).mean() < lowerLimit].axes[0].tolist()
LowerList2 = abs(upperSet).median()[abs(upperSet).median() < lowerLimit].axes[0].tolist()
#LowerList2
#LowerList2

#final List
finalSet=set(LowerList1)&set(LowerList2)

corr.loc[finalSet][list(finalSet)]
#common

#list(set(list1).intersection(list2))

#lowerList = lower1.axes[0].tolist()
#lowerList
#lowerSet=upperSet[lowerList]

#upper=corr[fieldOfInterest][abs(corr[fieldOfInterest]) > upperLimit]

#abs(upperSet).mean().mean()
#print(corr)
#corr.style.background_gradient(highlight_rows, axis = 0)
#corr.style.apply(highlight_rows, axis = 0)
#corr.style.background_gradient()



#dateLagged = date.shift(+1)
#bins = [ 1, 5, 10, 25, 50, 100]
#df['binned'] = pd.cut(df['percentage'], bins)

#dir(date)

#date.to_datetime(date.date)
#pandas.Series.dt.month(date)

#https://stackoverflow.com/questions/26105804/extract-month-from-date-in-python/26105888
#datetime.datetime.strptime(date, "%Y-%m-%d")


#result