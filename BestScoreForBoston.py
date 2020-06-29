from itertools import combinations
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def test_method(x, y):
    model = LinearRegression()
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)


boston_dataset = load_boston()
boston_dataset2 = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston2 = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
stuff = list(boston2)
boston['MEDV'] = boston_dataset.target

i = 0
maxScore = 0
winSubset = 0

for L in range(1, len(stuff)):
    for subsetT in combinations(stuff, L):
        subset = list(subsetT)
        i = test_method(boston[subset], boston['MEDV'])
        if i > maxScore:
            maxScore = i
            winSubset = subset
    print(L, "from:", len(stuff), maxScore, winSubset)
print(winSubset)
