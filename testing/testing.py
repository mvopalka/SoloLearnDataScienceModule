from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

iris = pd.read_csv('https://sololearn.com/uploads/files/iris.csv')
iris.drop('id', axis=1, inplace=True)

X = iris[['petal_len', 'petal_wd', 'sepal_wd', 'sepal_len']]
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
print(model.predict_proba(X_test))
