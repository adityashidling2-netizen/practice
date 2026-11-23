import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

irir = load_iris()

data = pd.DataFrame(irir.data, columns=irir.feature_names)
data['Species'] = irir.target

print(data.head())
print(data.info())
print(data.describe())
print(data.shape)
print(data.columns)
print(data.isnull().sum())

#converting 
data['Species_name'] = data['Species'].map({
    0:'Setosa',1:'Versicolor', 3:'Verginica'
})
print(data.head())
#Inp target
x = data[irir.feature_names]
y= data['Species']

#standar scaler
sc = StandardScaler()
x_new = sc.fit_transform(x)
print(x_new[:5])

#splitting
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=45
)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)