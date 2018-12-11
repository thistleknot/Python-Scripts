from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split

input_file = "prepped.csv"

#https://stackoverflow.com/questions/36519086/pandas-how-to-get-rid-of-unnamed-column-in-a-dataframe/36519122
df = pd.read_csv(input_file, header = 0, index_col=0)

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

x = df.drop(columns=['test2_z.date','yFYield_CSUSHPINSA'])
y = df.loc[0:,['yFYield_CSUSHPINSA']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.50)

result = stepwise_selection(X_train, y_train)
print(result)
#get RNG list from X_train output of prior train_test_split
#train_index = X_train.index
#test_index = X_test.index

#https://stackoverflow.com/questions/19155718/select-pandas-rows-based-on-list-index
#set randomized
#set1[list2].loc[train_index]

result2 = stepwise_selection(X_test, y_test)
print(result2)

model_training = sm.OLS(y_train,X_train,missing = 'drop').fit()
#X_train.filter(items=[result], axis=0)

#x.filter(regex=result)
    

#xTrainSubset = set1[list(set(list3))].loc[train_index]
#xTestSubset = set1[list(set(list3))].loc[test_index]


#model_training = sm.OLS(y_train,xTrainSubset,missing = 'drop').fit()
#print(model_training.summary())

#model_testing = sm.OLS(y_test,xTestSubset,missing = 'drop').fit()
#print(model_testing.summary())

#problem is duplicates aren't excluded, trying to use set didn't work


#set1[list(set(list3))].loc[train_index]


#X_train.filter(items=[result], axis=0)

#x.filter(regex=result)

        