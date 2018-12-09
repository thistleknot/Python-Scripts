from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

input_file = "output_test.csv"

#https://stackoverflow.com/questions/36519086/pandas-how-to-get-rid-of-unnamed-column-in-a-dataframe/36519122
df = pd.read_csv(input_file, header = 0, index_col=0)

#def train_test_split2(X, y)

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

#x = df.drop(columns=['date','CSUSHPINSA']).sample(20, axis=1)
date = df.loc[0:,['test2_z.date']]

#append if desire a sample
#.sample(50, axis=1)
#attempts at trying to count 0's and drop columns have failed in R, so I'm hard coding the column drop here
x = df.drop(columns=['test2_z.date','CSUSHPINSA','FGCCSAQ027S','future'])
y = df.loc[0:,['CSUSHPINSA']]

#dateLagged = date.shift(+1)
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
result = pd.concat([date, x, xYield, y, yYield, yFYield], axis=1, sort=False)
#replace inf's with 0
result.replace(np.inf, 0, inplace=True)

result.to_csv("output.csv", sep=',')
#plt.matshow(result.corr())
#https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas

corr = result.corr()
#matches corr formula in excel
#matches formula here: https://libguides.library.kent.edu/SPSS/PearsonCorr
corr.to_csv("corr.csv", sep=',')

print(corr)
#corr.style.background_gradient(highlight_rows, axis = 0)
#corr.style.apply(highlight_rows, axis = 0)
#corr.style.background_gradient()

