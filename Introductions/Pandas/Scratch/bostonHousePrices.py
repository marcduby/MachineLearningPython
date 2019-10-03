

# imports
import pandas as pd

# read in the data
boston_prices = pd.read_csv("~/Data/Scratch/boston_house_prices.csv")

# print description
print(boston_prices.describe())

# print description
print(boston_prices.info())

# drop columns with no value
train_data = boston_prices.drop(columns = ['LSTAT', 'MEDV'])

# print description
print(train_data.describe())



