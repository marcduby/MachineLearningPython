

## Recap
# So far, you have loaded your data and reviewed it with the following code. Run this cell to set up your coding environment where the previous step left off.



# Code you have previously used to load data
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")



# print the list of columns in the dataset to find the name of the prediction target
home_data.columns

y = home_data.SalePrice

step_1.check()

# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# select data corresponding to features in feature_names
X = home_data[feature_names]

step_2.check()



# Review data
# print description or statistics from X
print(X.describe())

# print the top few lines
print(X.head())




from sklearn.tree import DecisionTreeRegressor
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state = 2)

# Fit the model
iowa_model.fit(X, y)

step_3.check()



predictions = iowa_model.predict(X)
print(predictions)
step_4.check()


# compare predictions
y.head()
type(predictions)
predictions[0:5]


# test accuracy with mean absolute error (avg of each absolute error)
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)



# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model

melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))






