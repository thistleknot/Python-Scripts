#https://cran.r-project.org/web/packages/olsrr/vignettes/variable_selection.html
#https://www.statsmodels.org/dev/examples/notebooks/generated/formulas.html

from __future__ import print_function
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

from sklearn.datasets import load_boston
import pandas as pd
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split

input_file = "prepped.csv"

fieldOfInterest='yFYield_CSUSHPINSA'

#df = pd.read_csv(input_file, header = 0, index_col=0)

df = pd.read_csv(input_file, header = 0)

x = df.drop(columns=['test2_z.date','yFYield_CSUSHPINSA','BL_yFYield_CSUSHPINSA'])
y = df.loc[0:,['yFYield_CSUSHPINSA']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.50)

X_train

#https://stackoverflow.com/questions/5251507/how-to-succinctly-write-a-formula-with-many-variables-from-a-data-frame
#Yes of course, just add the response y as first column in the dataframe and call lm() on it:

frames = pd.concat([pd.DataFrame(y_train),pd.DataFrame(X_train)], axis=1)
#lm(frames)


#s1 = pd.Series(X_train, name = s1)

#result = pd.concat([y_train, s1], axis=1)
#result
#X_train
#print(MR_data.index)
#lm(fieldOfInterest ~ , data=frames)

#https://www.statsmodels.org/dev/examples/notebooks/generated/formulas.html
#(sm.OLS(y, X).fit().summary())

list(frames)
' '.join(list(y_train)) + ' ~ ' + ' + '.join(list(X_train))
#mod = ols(formula=,data=frames)
#model <- lm(

#X_train,X_test