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
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

input_file = "parsed.csv"

df = pd.read_csv(input_file, header = 0)

#had to strip off yields because some values have 0... and interactions work regardless of 0 (or -)

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

#[1:,][:-1]
#remove top and bottom row
X_train, X_test, y_train, y_test = train_test_split(set1.loc[1:,][:-1], yFutureYield.loc[1:,][:-1], test_size=0.2, random_state=0)

#should be checking and flagging both columns if na is found in any
model_scikit = lm.fit(X_train.dropna(axis=1, how='all'), y_train)
predictions = lm.predict(X_test.dropna(axis=1, how='all'))

model_training = sm.OLS(y_train,X_train,missing = 'drop').fit()
print(model_training.summary())

print(pd.concat([model_training.predict(X_test),y_test],axis=1))

print ("Train Score:", model_scikit.score(X_train.dropna(axis=1, how='all'), y_train))
print ("Test Score:", model_scikit.score(X_test.dropna(axis=1, how='all'), y_test))
dir(model_training)
model_testing = sm.OLS(model_training.predict(X_test),y_test,missing='drop').fit()
model_testing.summary()

plt.scatter(y_test, model_training.predict(X_test))
plt.xlabel("True Values")
plt.ylabel("Predictions")

#http://scikit-learn.org/stable/modules/cross_validation.html

#np.array(X_train).flatten()
#np.array(y_train).flatten()

#clf = svm.SVC(kernel='linear', C=1).fit(np.array(X_train).flatten(), np.array(y_train).flatten())
