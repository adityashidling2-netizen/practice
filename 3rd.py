import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = sns.load_dataset('titanic')
print(df.head())


#quartile finding
q1 = df['fare'].quantile(0.25)
#3rd quartile
q3 = df['fare'].quantile(0.75)

iqr = q3-q1
print(q1)
print(q3)
print(iqr)

#ub
ub = q3 + (1.5*iqr)
lb = q1 - (1.5*iqr)
print("--Upper bound--")
print(ub)
print("-- Lower bound--")
print(lb)

fare = df['fare']
outlier = fare [(fare <= lb) | (fare >= ub)]
print("Outilers in age")
print(outlier)

#Removing outliers 
tr = df.copy()
ui = tr [tr['fare']>ub].index
li = tr [tr['fare']<lb].index

tr.drop(ui, inplace=True)
tr.drop(li, inplace= True)
print(df.info())
print(tr.info())
#Replacing outlier with upper cap and lowercap
tc = df.copy()
fare = tc['fare']
uc = np.percentile(fare.dropna(), 99)
lc = np.percentile(fare.dropna(), 1)
print("Upper cap ", uc)
print("Lower cap ", lc)
tc['fare'] = tc['fare'].clip(lower=lc, upper=uc)
print("Dataset after winsorization method")
print(tc['fare'])

#using mean
tme = df.copy()
meanfare = df['fare'].mean()
tme['fare'] = np.where(
    (tme['fare']< lb) | (tme['fare']>ub), meanfare, tme['fare']
)
print(tme['fare'])
#using mean
tme = df.copy()
meanfare = df['fare'].median()
tme['fare'] = np.where(
    (tme['fare']< lb) | (tme['fare']>ub), meanfare, tme['fare']
)
print(tme['fare'])