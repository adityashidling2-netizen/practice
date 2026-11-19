import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

#load dataset 
data = pd.read_csv('apples_and_oranges.csv')
print(data.head())

#getting input variables and target
x = data.drop('Class', axis=1)
y = data['Class']

#splitting
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=45
)
sns.scatterplot(x='Weight', y='Size', hue='Class', data=data)
plt.show()

#model creating
model = SVC(kernel='rbf', random_state=45, gamma='auto')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy score")
print(accuracy_score(y_pred, y_test))

cm = confusion_matrix(y_pred, y_test)
print(cm)