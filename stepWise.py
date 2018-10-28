from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split

input_file = "parsed.csv"

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
x = df.drop(columns=['date','CSUSHPINSA']).sample(50, axis=1)
y = df.loc[0:,['CSUSHPINSA']]

xLagged = x.shift(+1)
yLagged = y.shift(+1)
yFuture = y.shift(-1)
yYield = (yLagged-y)/yLagged
yFutureYield = (yFuture-y)/y

symbols = ['x', 'lagged', 'interaction', 'y', 'y_yield']

#without prefix, the next stepwise command gets confused on column names...
#would be best to only pass x, and then backjoin to these other derived values to do step_wise on
set1 = pd.concat([x,xLagged.add_prefix('xLag_'),(x*xLagged).add_prefix('xInter_'),y.add_prefix('y_'),yYield.add_prefix('yYield_')], names=['symbols'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(x.loc[2:,][:-1], yFutureYield.loc[2:,][:-1], test_size=0.25)

result = stepwise_selection(X_train, y_train)


print(result)

#filtered
my_list = set1[result]

#all columns
all_list = list(set1)

list2 = result
for i in range(0, len(list(result))):
    
        xInter_ = "xInter_" + result[i]
        list2.append(xInter_)
        xLag_ = "xLag_" + result[i]
        list2.append(xLag_)
        
        continue

#get RNG list from X_train output of prior train_test_split
train_index = X_train.index

print(list2)
#https://stackoverflow.com/questions/19155718/select-pandas-rows-based-on-list-index
set1[list2].loc[train_index]

result2 = stepwise_selection(set1[list2].loc[train_index], y_train)
print(result2)

list3 = result2
for i in range(0, len(list(result2))):
    
        temp = result[i]
        if temp.startswith("xInter_"):
            list3.append(temp[-7:])
            #continue
        elif temp.startswith("xLag_"):
            list3.append(temp[-5:])
            #continue
        #continue
    
#print(list3)

set(list3)

#model_training = sm.OLS(y_train,X_train,missing = 'drop').fit()
#X_train.filter(items=[result], axis=0)

#x.filter(regex=result)

        