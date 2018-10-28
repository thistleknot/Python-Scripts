import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split
import pylab

import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import statsmodels.api as sm
from scipy.stats import zscore

input_file = "parsed.csv"

df = pd.read_csv(input_file, header = 0)

#be sure to strip off date, else litter later calculations with [1:-1,1:]

x = df.loc[0:,[
'CSUSHPISA',
'CUUR0000SETB01',
'DCOILBRENTEU',
'RECPROUSM156N',
'CPIHOSNS',
'CPALTT01USM661S',
'PAYNSA',
'CUUR0000SEHA',
'CPIAUCSL',
'LNS12300060',
'GS5',
'CUUR0000SETA01',
'CPILFESL',
'CPILFENS',
'PCECTPICTM'
]]
xLagged = x.shift(+1)

y = df.loc[0:,['CSUSHPINSA']]
yLagged = y.shift(+1)
yFuture = y.shift(-1)
yYield = (yLagged-y)/yLagged
yFutureYield = (yFuture-y)/y

set1 = pd.concat([x,xLagged,x*xLagged,y,yYield], axis=1)
#set1 = pd.concat([x,xLagged,x*xLagged], axis=1)

#this model doesn't require this subsetting
#set1.loc[1:,][:-1]
model = sm.OLS(yFutureYield,set1,missing = 'drop').fit()
model.summary()

#output model records
#pd.concat([set1,yFutureYield],axis=1)