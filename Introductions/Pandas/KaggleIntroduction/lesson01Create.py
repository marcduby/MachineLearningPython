

# imports
import pandas as pd
pd.set_option('max_rows', 5)

# create data frames
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
fruits = pd.DataFrame([[30, 21]], columns=['Apples', 'Bananas'])

# q1.check()

fruits

# create rows
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = pd.DataFrame([[35, 21], [41, 34]], columns=['Apples', 'Bananas'],
                index=['2017 Sales', '2018 Sales'])

# q2.check()

fruit_sales

# other way to create data frame with index and columns
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals



# use series
quantities = ['4 cups', '1 cup', '2 large', '1 can']
items = ['Flour', 'Milk', 'Eggs', 'Spam']
ingredients = pd.Series(quantities, index=items, name='Dinner')

# q3.check()

ingredients


# read csv
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

reviews.head()

# q4.check()

reviews


# write csv
animals.to_csv("cows_and_goats.csv")






