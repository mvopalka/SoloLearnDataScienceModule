import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
print(boston[['CHAS', 'RM', 'AGE', 'RAD', 'MEDV']].head())
print()
print(boston.describe().round(2))

#print(boston_dataset.target)
#boston['MEDV'] = boston_dataset.target
#print(boston.shape)
