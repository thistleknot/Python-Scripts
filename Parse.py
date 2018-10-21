import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import statsmodels.api as sm

#stepwise
#https://datascience.stackexchange.com/questions/24405/how-to-do-stepwise-regression-using-sklearn/24447#24447
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


#model_with_interactions.summary()
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import statsmodels.api as sm

#from numpy import percentile

input_file = "parsed.csv"

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
x_interaction = (x.iloc[1:-1,1:]*x_lagged.iloc[1:-1,1:])

y_yield = (y.iloc[1:-1,]-y_lagged.iloc[1:-1,])/y_lagged.iloc[1:-1,]
y_interaction = (y.iloc[1:-1,]*y_lagged.iloc[1:-1,])
#y_future_yield = ()

#new column, no matching x (this isn't validation data, this is offset data)
y_future_yield = (y_future.iloc[1:-1,]-y.iloc[1:-1,])/y.iloc[1:-1,]
y_future_interaction = (y_future.iloc[1:-1,]*y.iloc[1:-1,])

x_and_y_with_yields = pd.concat([x.iloc[1:-1,1:], x_yield, y.iloc[1:-1,], y_yield], axis=1)
x_and_y_with_interactions = pd.concat([x.iloc[1:-1,1:], x_lagged.iloc[1:-1,1:], x_interaction, y.iloc[1:-1,], y_lagged.iloc[1:-1,], y_interaction], axis=1)

# .938 Adj R^2
model = sm.OLS(y_future_yield.loc[1:int(split+1)], x_and_y_with_yields.loc[1:int(split+1)]).fit()
model_with_interactions = sm.OLS(y_future_yield.loc[1:int(split+1)], x_and_y_with_interactions.loc[1:int(split+1)]).fit()

#note: has to be +2
    #x_and_y_with_yields.loc
#results_with_interactions = pd.concat([model.predict(x_and_y_with_interactions.loc[int(split+1):]), y_future_yield[int(split+1):]], axis=1)
results = pd.concat([model.predict(x_and_y_with_yields.loc[int(split+1):]), y_future_yield[int(split+1):]], axis=1)


model.params


#scikit
#https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
X_train, X_test, y_train, y_test = train_test_split(x_and_y_with_yields.loc[1:int(split+1)], y_future_yield.loc[1:int(split+1)], test_size=0.2)

lm = linear_model.LinearRegression()

model_scikit = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

predictions.shape[0]
y_test.shape[0]
#pd.concat([predictions,y_test],axis=1)
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")

print ("Score:", model_scikit.score(X_test, y_test))
results


#type(y_test)
#type(predictions)
#predictions[0:,0:].tolist()
#model.params
#model.summary()
print(model.summary())
print(results)
results.to_csv("results_1stHalf.csv")


#model_with_interactions.summary()
x_and_y_with_interactions

#stepwise regression
#result = stepwise_selection(x.loc[0:,1:],y)
#result
x.loc[0:int(split+1)]