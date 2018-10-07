import numpy as np
import pandas as pd
import sklearn.model_selection as ms
#from numpy import percentile

input_file = "C:/Users/user/Documents/plots/parsed.csv"

df = pd.read_csv(input_file, header = 0)
df

X, y = df.iloc[:, :-1], df.iloc[:, -1]

X

y

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.25, random_state=42)
