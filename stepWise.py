from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split

input_file = "parsed.csv"

#https://stackoverflow.com/questions/36519086/pandas-how-to-get-rid-of-unnamed-column-in-a-dataframe/36519122
df = pd.read_csv(input_file, header = 0, index_col=0)

#x = df.drop(columns=['date','CSUSHPINSA']).sample(20, axis=1)
x = df.drop(columns=['date','CSUSHPINSA']).sample(20, axis=1)
y = df.loc[0:,['CSUSHPINSA']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

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
            best_feature = new_pval.idxmin()
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
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

#xLagged = x.shift(+1)
#yLagged = y.shift(+1)
#yFuture = y.shift(-1)
#yYield = (yLagged-y)/yLagged
#yFutureYield = (yFuture-y)/y

symbols = ['x', 'lagged', 'interaction', 'y', 'y_yield']

set1 = pd.concat([x,xLagged,x*xLagged,y,yYield], keys=['x', 'lagged', 'interaction', 'y', 'y_yield'], axis=1, names=['symbols'])


result = stepwise_selection(X_train, y_train)

#yFutureYield.loc[2:,][:-1]
#set1.loc[2:,][:-1]
#X_train
#y_train
#set1.loc[2:,][:-1].head(5)
#set1.loc[1:,][:-1].tail(5)

#yFutureYield.loc[1:,][:-1].head(5)
#yFutureYield.loc[1:,][:-1].tail(5)