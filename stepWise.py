import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import statsmodels.api as sm

import sklearn.feature_selection as fs

#stepwise
#https://datascience.stackexchange.com/questions/24405/how-to-do-stepwise-regression-using-sklearn/24447#24447
#Stating that OLS is just not good enough compared to other methods is misleading. For a linearly separable dataset where the Gauss-Markov assumptions are satisfied, OLS will be more efficient than any other linear or nonlinear method. It's more of a question of data and model structure than anything else. – Digio
#As far as I understand, p-values (1) are a very specific interpretation of a single OLS algorithm, and (2) are useful for inference (to decide whether a single predictor matters), but not so useful for prediction (model with lots of bad p-values may have good predictive power, and vice versa) – David Dale (author of code and answer)

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

#from numpy import percentile

input_file = "parsed.csv"

df = pd.read_csv(input_file, header = 0)

split=int(len(df.index))/2

df

xsw = df.drop(columns=['date', 'CSUSHPINSA','future'])[0:int(split+1)]

ysw = df.loc[0:,'CSUSHPINSA'][0:int(split+1)]

xsw.iloc[0:,0:]

#wout date
#result = stepwise_selection(xsw.iloc[0:,1:220],ysw)
#xsw.iloc[0:,1:220]
#result
fs.f_regression(xsw.iloc[1:,1:2],ysw[1:],center=True)

#ps = fs.f_regression(xsw.iloc[0:,1:220],ysw,center=TRUE)

#x = df(columns=[result])[0:int(split+1)]
