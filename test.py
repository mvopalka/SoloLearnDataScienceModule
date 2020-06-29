import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from itertools import combinations


def testMethode(x, y):
    model = LinearRegression()
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)


boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
print(boston[['CHAS', 'RM', 'AGE', 'RAD', 'MEDV']].head())
print()
print(boston.describe().round(2))

# print(boston_dataset.target)
# boston['MEDV'] = boston_dataset.target
# print(boston.shape)

X = boston[['CHAS', 'RM', 'AGE', 'RAD']]
Y = boston['MEDV']

# model = LinearRegression()
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
# model.fit(X_train, Y_train)
# Y_test_predicted = model.predict(X_test)
# print()
# print(model.score(X_test, Y_test))

print(testMethode(boston[['CHAS', 'RM', 'AGE', 'RAD']], boston['MEDV']))
print(testMethode(boston[['RM']], boston['MEDV']))
print(testMethode(boston[['CHAS', 'RM']], boston['MEDV']))
